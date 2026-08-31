"""A collection of classes and functions for interacting with the database (Oracle or SQLite).

This module provides database abstraction layer for managing time series sensor data,
model configurations, and forecast data. Supports both Oracle and SQLite backends
through SQLAlchemy ORM.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# from warnings import deprecated (works from python 3.13 on)
from dotenv import load_dotenv
from pandas.tseries.offsets import DateOffset
from sqlalchemy import between, bindparam, create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from torch.utils.data import DataLoader, Dataset

from forecastlib.database.orm_classes import (
    Base,
    InputForecastsLong,
    InputForecastsMeta,
    Model,
    ModelSensor,
    PegelForecastsLong,
    SensorData,
)
from forecastlib.models.LightningWrapper import CustomLightningModule, EnsembleModule, UncertaintyLightningModule
from forecastlib.utils.timefeatures import get_data_stamp

# pylint: disable=unsupported-assignment-operation
# pylint: disable=unsubscriptable-object
# pylint: disable=reportAttributeAccessIssue
# pylint: disable=reportArgumentType
# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
# pyright: reportUnusedVariable=warning, reportUntypedBaseClass=error


# TODO external forecast with missing values cant work at the moment
# TODO track for which timestamps we actually made predictions


class DBModelDataset(Dataset):
    """Dataset providing batches of time series windows from forecastlib.database.

    This class loads pre-scaled sensor data from a database along with optional
    external forecast data, and produces windows compatible with time series
    forecasting models. It handles data preparation including scaling, time
    encoding, differencing, and external forecast integration.

    Important:
        This class does NOT load the model. The caller must load the model
        and pass the scaler/hparams-derived sizes and the scaled dataframe
        into this constructor.

    Args:
        engine: SQLAlchemy database engine for data access.
        columns: List of sensor column names to use as features.
        scaler: sklearn scaler object (fitted StandardScaler) for data normalization.
        timestamps: List or index of timestamps defining data range.
        target_sensor_name: Name of the primary target sensor for forecasting.
        seq_len: Length of input sequence (lookback window).
        label_len: Length of the label/encoder sequence.
        pred_len: Length of prediction horizon (forecast length).
        df_ext_meta: DataFrame with metadata about external forecasts.
        members: List of ensemble member indices to include.
        timeenc: Time encoding mode (0=integer, 1=continuous normalized).
        diff: Whether to use differencing for non-stationary data.
        max_missing: Maximum number of consecutive missing values to interpolate.
            Defaults to 200.
        freq: Frequency string for time encoding. Defaults to "h" (hourly).
        **kwargs: Additional keyword arguments passed to parent class.

    """

    def __init__(
        self,
        engine: Any,
        columns: list[str],
        scaler: Any,
        timestamps: pd.DatetimeIndex | list[pd.Timestamp],
        target_sensor_name: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        df_ext_meta: pd.DataFrame,
        members: list[int],
        timeenc: int,
        diff: bool,
        max_missing: int = 200,
        freq: str = "h",
        **kwargs: Any,
    ) -> None:
        """Initialize DBModelDataset with database connection and configuration.

        Args:
            engine: SQLAlchemy database engine instance.
            columns: List of sensor names to load from forecastlib.database.
            scaler: Fitted sklearn scaler for normalizing sensor values.
            timestamps: Timestamps defining the data range to load.
            target_sensor_name: Primary target column for forecasting.
            seq_len: Number of historical timesteps for input.
            label_len: Number of timesteps in label/encoder region.
            pred_len: Number of timesteps to forecast ahead.
            df_ext_meta: DataFrame containing external forecast metadata.
            members: List of ensemble member IDs to include.
            timeenc: Time encoding method (0=discrete, 1=continuous).
            diff: Whether to apply differencing to data.
            max_missing: Maximum gap size for interpolation. Defaults to 200.
            freq: Frequency string for time features. Defaults to "h".
            **kwargs: Additional arguments (unused, for compatibility).

        """
        self.engine = engine
        self.columns = columns
        self.scaler = scaler
        self.timestamps = timestamps
        self.target_sensor_name = target_sensor_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.df_ext_meta = df_ext_meta
        self.members = members
        self.timeenc = timeenc
        self.max_missing = max_missing
        self.diff = diff
        self.freq = freq

        self.df_sensor = self._load_input_db()
        self.x_scaled = torch.Tensor(self.scaler.transform(self.df_sensor))
        self.data_stamp = self._get_datastamp()
        self.df_ext = self._load_ext_db()
        self.df_ext_scaled = self._get_df_ext_scaled()

        self.max_ens_size = df_ext_meta["ensemble_members"].max()
        if np.isnan(self.max_ens_size):
            self.max_ens_size = 1
            self.common_indexes = self.timestamps
        else:
            self.common_indexes = sorted(
                self
                .df_ext[self.df_ext["horizon_step"] == 1]
                .groupby("sensor_name")
                .apply(lambda x: set(x.index))
                .pipe(lambda x: set.intersection(*x))
            )

    def __len__(self) -> int:
        """Return total number of samples in dataset.

        Returns:
            Total number of windows available, accounting for ensemble members.

        """
        return len(self.common_indexes) * self.max_ens_size

    def _get_datastamp(self) -> torch.Tensor:
        """Generate time encoding stamps for all timestamps.

        Creates temporal features for each timestamp based on configured
        time encoding mode. Results are cached as tensors for efficiency.

        Returns:
            Tensor of time encodings with shape (num_timestamps, num_features).

        """
        start_time = min(self.timestamps) - pd.Timedelta(self.seq_len - 1, "hours")
        end_time = max(self.timestamps) + pd.Timedelta(hours=self.pred_len + 1)
        df_stamp = pd.DataFrame(pd.date_range(start_time, end_time, freq="1H"), columns=["date"])

        data_stamp = get_data_stamp(df_stamp, self.timeenc, "1H")
        return torch.tensor(data_stamp, dtype=torch.float32)

    def _load_input_db(self) -> pd.DataFrame:
        """Load sensor data from database and apply preprocessing.

        Retrieves historical sensor data from database, performs interpolation
        for missing values, shifts external forecasts by appropriate lead times,
        and optionally applies differencing for non-stationary data.

        Returns:
            DataFrame containing preprocessed sensor data indexed by timestamp,
            with columns for each sensor. Shape: (num_timestamps, num_sensors).

        Raises:
            ValueError: If missing values remain after interpolation.
            AssertionError: If inferred frequency is not hourly.

        """
        start_time = min(self.timestamps) - pd.Timedelta(self.seq_len - 1, "hours")
        end_time = max(self.timestamps)

        stmt = select(SensorData).where(
            between(SensorData.tstamp, bindparam("start_time"), bindparam("end_time")),
            SensorData.sensor_name.in_(bindparam("model_columns", expanding=True)),
        )

        df_main = pd.read_sql(
            sql=stmt,
            con=self.engine,
            index_col="tstamp",
            params={
                "start_time": start_time,
                "end_time": end_time + pd.Timedelta(hours=self.pred_len),
                "model_columns": self.columns,
            },
        )
        df_main = df_main.pivot(columns="sensor_name", values="sensor_value")[self.columns]
        df_input = df_main[:end_time]
        df_rest = df_main[end_time + pd.Timedelta("1h") :]
        df_rest.index = df_rest.index - pd.Timedelta("48h")

        # TODO assert infered frequency
        assert df_input.index.inferred_freq == "h", "Some timestamps are missing completly."

        for col in df_input.columns:
            if df_input[col].isna().sum() > 0:
                logging.warning(
                    "Missing %s values in  %s, attempting interpolation",
                    df_input[col].isna().sum(),
                    col,
                    extra={"gauge": "Unknown in this function (todo)"},
                )

        # Interpolation
        # Precipitation is filled with 0, other values are interpolated
        # precip_cols = df_input.columns[df_input.columns.str.contains("Precip")]
        # df_input.loc[:, precip_cols].fillna(0, inplace=True)

        interpol_cols = df_input.columns.drop(list(self.df_ext_meta["sensor_name"]))
        df_input[interpol_cols] = df_input[interpol_cols].interpolate(limit=self.max_missing, limit_direction="both")

        if df_input.isna().sum().sum() > 0:
            for col in df_input.columns:
                if df_input[col].isna().sum() > 0:
                    logging.warning(
                        "Missing %s values in %s after interpolation",
                        df_input[col].isna().sum(),
                        col,
                        extra={"gauge": "Unknown in this function (todo)"},
                    )
            msg = f"Missing {df_input.isna().sum().sum()} values after interpolation"
            raise ValueError(msg)

        # Shift columns that contain external forecasts by 48 hours
        # for col in external_fcst:
        for _, row in self.df_ext_meta.iterrows():
            df_input[row.sensor_name] = df_input[row.sensor_name].shift(
                -row.forecast_length
            )  # TODO hier wird die forecast length benötigt.

        if self.diff:
            df_diff = df_input[[self.target_sensor_name]].diff()
            df_diff.iloc[0] = 0
            df_diff.columns = df_diff.columns + "_d1"
            df_input = pd.concat([df_input, df_diff], axis=1)

        return df_input

    def _load_ext_db(self) -> pd.DataFrame:

        stmst = select(InputForecastsLong).where(
            InputForecastsLong.sensor_name.in_(self.df_ext_meta["sensor_name"]),
            between(InputForecastsLong.tstamp, min(self.timestamps), max(self.timestamps)),
            InputForecastsLong.member.in_(self.members),
        )

        return pd.read_sql(sql=stmst, con=self.engine, index_col="tstamp")

    def _get_df_ext_scaled(self):
        # Set multi-index for pivoting
        df_pivot = self.df_ext.set_index(["member", "horizon_step"], append=True).pivot(
            columns="sensor_name", values="value"
        )

        # Ensure columns match df_sensor (fill missing sensors with NaN)
        for col in self.df_sensor.columns:
            if col not in df_pivot.columns:
                df_pivot[col] = np.nan

        # Reorder columns to match df_sensor
        df_pivot = df_pivot[self.df_sensor.columns]

        # Apply the scaler
        df_transformed = pd.DataFrame(self.scaler.transform(df_pivot), index=df_pivot.index, columns=df_pivot.columns)
        df_transformed = df_transformed.dropna(axis=1)

        # Melt back to original format
        df_ext_scaled = df_transformed.melt(var_name="sensor_name", value_name="value", ignore_index=False)

        # Reset the multi-index back to columns (except tstamp)
        return df_ext_scaled.reset_index(level=["member", "horizon_step"])

    def get_whole_index(self):
        rows = [(self.common_indexes[idx // self.max_ens_size], idx % self.max_ens_size) for idx in range(len(self))]
        return pd.DataFrame(rows, columns=["tstamp", "member"])

    def __getitem__(self, idx):
        if idx >= len(self):
            msg = f"Index {idx} out of range, max index is {len(self) - 1}"
            raise IndexError(msg)
        tstamp = self.common_indexes[idx // self.max_ens_size]
        member = idx % self.max_ens_size

        s_end = self.df_sensor.index.get_loc(tstamp) + 1
        s_begin = s_end - self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # base x and y as numpy arrays
        seq_x = self.x_scaled[s_begin:s_end].detach().clone()

        # x = self.x_scaled[s_begin:s_end].copy()
        # y = self.x_scaled[r_begin:r_end].copy()

        ext_fcst_all = self.df_ext_scaled.loc[tstamp]
        # insert external forecasts into the tail of x if available
        for _i, row in self.df_ext_meta.iterrows():
            col_idx = self.df_sensor.columns.get_loc(row.sensor_name)
            nan_indices = torch.where(seq_x[:, col_idx].isnan())[0] - (self.seq_len - self.pred_len)
            nan_indices = nan_indices[nan_indices >= 0]  # No idea why i did this
            nan_indices2 = nan_indices + (self.seq_len - self.pred_len)
            # ext_fcst = ext_fcst_all[(ext_fcst_all["member"] == member) & (ext_fcst_all["sensor_name"] == "112211,Precip,h.Cmd")].sort_values("horizon_step")["value"]

            temp_member = 0 if row.ensemble_members == 1 else member

            ext_fcst = ext_fcst_all[
                (ext_fcst_all["member"] == temp_member) & (ext_fcst_all["sensor_name"] == row.sensor_name)
            ].sort_values("horizon_step")["value"]
            seq_x[nan_indices2, col_idx] = torch.Tensor(ext_fcst)[nan_indices]

        if self.label_len == 0:
            seq_y = torch.zeros([self.pred_len, seq_x.shape[1]])
        else:
            seq_y = torch.zeros([self.pred_len, seq_x.shape[1]])
            seq_y = torch.cat([seq_x[-self.label_len :], seq_y], dim=0)

        # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])#.float()
        # dec_inp = torch.cat([seq_x[-2:], dec_inp], dim=0)#.float().to(self.device)

        # convert to tensors
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        assert seq_x.shape[0] == self.seq_len, "Wrong shape?"
        assert seq_y.shape[0] == self.pred_len + self.label_len, "Wrong shape?"
        assert seq_x_mark.shape[0] == self.seq_len, "Wrong shape?"
        assert seq_y_mark.shape[0] == self.pred_len + self.label_len, "Wrong shape?"

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class DatabaseConnection:
    """Class for writing/reading from the Database."""

    def __init__(self, engine, config: dict) -> None:
        self.engine = engine
        self.config = config
        self.Session = sessionmaker(self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine, checkfirst=True)

        self.members = self._get_member_list()
        self.times = self._get_times()

    def handle_model(self, model_name: str) -> None:
        """Load a model and generate forecasts for all configured timestamps and ensemble members."""
        logging.info("Starting forecast generation for model: %s", model_name)
        model, params = self._get_model_and_params(model_name)
        logging.info("Model loaded: %s (is_ensemble=%s)", model_name, isinstance(model, EnsembleModule))
        if params.get("sub_model_ids"):
            logging.info("Ensemble submodel IDs: %s", params["sub_model_ids"])

        if params["kind"] == "XGBRegressor":
            self._handle_XGBoost(model_name, model, params)

        elif params["kind"] == "PastasModel":
            self._handle_Pastas(model_name, model, params)

        else:
            ds = DBModelDataset(
                engine=self.engine, members=self.members, timestamps=self.times, scaler=model.scaler, **params
            )
            logging.info(
                "Dataset created with %d timesteps, ensemble members: %s",
                len(ds) // len(self.members),
                len(self.members),
            )
            logging.info("Making predictions for timesteps: %s ", ds.common_indexes)

            dl = DataLoader(dataset=ds, shuffle=False, batch_size=32)  # MUST NOT BE 1
            logging.debug("DataLoader initialized for batch processing")

            pred, sub_preds, std = self._get_preds(model, dl)

            # Format regular forecast for database
            df_pred = ds.get_whole_index()
            df_pred = pd.concat([df_pred, pd.DataFrame(pred)], axis=1)
            df_pred = df_pred.melt(id_vars=["tstamp", "member"], var_name="horizon_step")
            df_pred["model_id"] = params["model_id"]

            # Same for ensemble forecasts
            if isinstance(model, EnsembleModule):
                df_pred_list = []
                for i, sub_model_id in enumerate(params["sub_model_ids"]):
                    # pred = sub_preds[i*len(pred_list):(i+1)*len(pred_list)]
                    df_temp = pd.concat([ds.get_whole_index(), pd.DataFrame(sub_preds[:, i, :])], axis=1)
                    df_temp = df_temp.melt(id_vars=["tstamp", "member"], var_name="horizon_step")
                    df_temp["model_id"] = sub_model_id
                    df_pred_list.append(df_temp)
                df_pred = pd.concat([df_pred, *df_pred_list], axis=0)
            elif isinstance(model, UncertaintyLightningModule):
                # For Std forecasts format the std values and add them to df_pred
                df_std = ds.get_whole_index()
                df_std = pd.concat([df_std, pd.DataFrame(std)], axis=1)
                df_std = df_std.melt(id_vars=["tstamp", "member"], var_name="horizon_step")
                df_pred["variance"] = df_std["value"]

            df_pred["sensor_name"] = params["target_sensor_name"]

            stmt = select(
                PegelForecastsLong.tstamp,
                PegelForecastsLong.member,
                PegelForecastsLong.horizon_step,
                PegelForecastsLong.model_id,
            ).where(
                PegelForecastsLong.model_id.in_([params["model_id"]] + params["sub_model_ids"]),
                PegelForecastsLong.sensor_name == params["target_sensor_name"],
                PegelForecastsLong.tstamp.in_(ds.common_indexes),
            )

            df_already_inserted = pd.read_sql(sql=stmt, con=self.engine)
            merged = df_pred.merge(
                df_already_inserted[["tstamp", "member", "horizon_step", "model_id"]],
                on=["tstamp", "member", "horizon_step", "model_id"],
                how="left",
                indicator=True,
            )
            df_pred_create = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
            df_pred_update = merged[merged["_merge"] == "both"].drop("_merge", axis=1)
            df_pred_update["created"] = datetime.now(timezone.utc)

            logging.info("Inserting %d forecasts and updating %d forecasts", len(df_pred_create), len(df_pred_update))

            with self.Session.begin() as session:
                forecast_objs = df_pred_create.apply(lambda row: PegelForecastsLong(**row.to_dict()), axis=1).tolist()
                session.add_all(forecast_objs)
                session.execute(update(PegelForecastsLong), df_pred_update.to_dict(orient="records"))

            logging.info("Finished inserting/updating forecasts")
            if self.config["export_zrxp"]:
                self._export_zrxp(df_pred, model_name, params["model_id"])

    def _get_preds(self, model, data_loader):
        pred_list = []
        std_list = []
        sub_preds = []
        logging.info("Beginning prediction loop for %d batches", len(data_loader))
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if isinstance(model, EnsembleModule):
                    x, pred, true, sub_pred = model.predict_step(batch, batch_idx=i)
                    sub_preds.append(sub_pred)
                elif isinstance(model, UncertaintyLightningModule):
                    x, pred, std, true = model.predict_step(batch, batch_idx=i)

                    std_list.append(std)
                else:
                    _x, pred, _true = model.predict_step(batch, batch_idx=i)
                pred_list.append(pred)

        pred = torch.concat(pred_list).squeeze()

        if isinstance(model, UncertaintyLightningModule):
            std = torch.concat(std_list).squeeze()
            std = torch.sqrt(torch.cumsum(std**2, dim=1))
        else:
            std = None

        sub_preds = torch.concat(sub_preds).squeeze() if isinstance(model, EnsembleModule) else None

        return pred, sub_preds, std

    def _get_model_and_params(self, model_name):
        """Load model from database and extract hyperparameters for dataset initialization."""
        logging.debug("Loading model and parameters for: %s", model_name)
        with self.Session.begin() as session:
            model_orm = session.scalar(select(Model).where(Model.model_name == model_name))
            model_blob_orm = model_orm.blob

            if model_orm.kind == "XGBRegressor":
                from forecastlib.models.WrapperClasses import XGBoostWrapper

                model = XGBoostWrapper.from_db(model_blob_orm.yaml, model_blob_orm.artifact)

                params = {
                    "kind": model_orm.kind,
                    "model_id": model_orm.id,
                    "target_sensor_name": model_orm.target_sensor_name,
                }

                return model, params
            if model_orm.kind == "PastasModel":
                from forecastlib.models.WrapperClasses import PastasWrapper

                model = PastasWrapper.from_db(model_blob_orm.yaml, model_blob_orm.artifact)
                params = {
                    "kind": model_orm.kind,
                    "model_id": model_orm.id,
                    "target_sensor_name": model_orm.target_sensor_name,
                }
                return model, params

            if model_orm.is_ensemble:
                sub_model_ids = [ensemble_member.submodel_model.id for ensemble_member in model_orm.ensemble_members]
                blob_list = [ensemble_member.submodel_model.blob for ensemble_member in model_orm.ensemble_members]
                sub_model_names = [
                    ensemble_member.submodel_model.model_name for ensemble_member in model_orm.ensemble_members
                ]
                logging.info(
                    "Loading ensemble model '%s' with %d submodels ( %s)",
                    model_name,
                    len(sub_model_ids),
                    sub_model_names,
                )
                model_list = [CustomLightningModule.from_db(tmp_blob.yaml, tmp_blob.artifact) for tmp_blob in blob_list]
                model = EnsembleModule.from_db(model_blob_orm.yaml, model_blob_orm.artifact, model_list)

                model.set_return_subresults(True)
                embed = model_list[0].hparams.embed
            elif model_orm.is_variance:
                logging.info("Loading uncertainty model '%s'", model_name)
                model = UncertaintyLightningModule.from_db(model_blob_orm.yaml, model_blob_orm.artifact)
                embed = model.hparams.embed
                sub_model_ids = []
            else:
                logging.info("Loading single model '%s'", model_name)
                model = CustomLightningModule.from_db(model_blob_orm.yaml, model_blob_orm.artifact)
                try:
                    embed = model.hparams.embed  # model.args.embed
                except AttributeError:
                    embed = model.args.embed

                sub_model_ids = []

            model.eval()
            # assert model.hparams.freq == "h", "Only hourly data/forecasts are implemented"

            ms_query = select(ModelSensor).where(ModelSensor.model_id == model_orm.id).order_by(ModelSensor.ix)
            df_ms = pd.read_sql(sql=ms_query, con=self.engine)

            external_fcst = list(df_ms.dropna()["sensor_name"]) if not df_ms.empty else []
            ext_forecast_query = select(InputForecastsMeta).where(InputForecastsMeta.sensor_name.in_(external_fcst))
            df_ext_meta = pd.read_sql(sql=ext_forecast_query, con=self.engine)

            # paramas we need later:
            params = {
                "columns": list(df_ms["sensor_name"]),
                "model_id": model_orm.id,
                "diff": True if model_orm.is_ensemble else model.diff,
                "seq_len": model.hparams.seq_len,
                "label_len": model.hparams.label_len,
                "pred_len": model.hparams.pred_len,
                "df_ext_meta": df_ext_meta,
                "target_sensor_name": model_orm.target_sensor_name,
                "timeenc": 0 if embed != "timeF" else 1,
                "embed": embed,
                # "external_fcst": external_fcst,
                "sub_model_ids": sub_model_ids,
                "kind": model_orm.kind,
            }

        return model, params

    def _handle_Pastas(self, model_name, model, params) -> None:
        # TODO Also need to update the obs timeseries?
        # TODO update evap/prec/temp with observed values if the exists
        # TODO handly no temperature used
        # TODO also update well data
        columns = [model.hparams["prec_name"], model.hparams["temp_name"], model.hparams["evap_name"]]

        stmst = select(InputForecastsLong).where(
            InputForecastsLong.sensor_name.in_(columns),
            between(InputForecastsLong.tstamp, min(self.times), max(self.times)),
        )
        df_ext = pd.read_sql(sql=stmst, con=self.engine)

        model_most_recent_tstamp = model.model.stressmodels["recharge"].prec.series.index.max()

        forecast_dict = {}
        for (member, tstamp), df_temp in df_ext.groupby(["member", "tstamp"]):
            if tstamp > model_most_recent_tstamp - pd.Timedelta(days=215):
                continue

            cur_index = pd.date_range(start=tstamp, freq="1D", periods=215)
            temp = df_temp[df_temp["sensor_name"] == model.hparams["temp_name"]].sort_values(by="horizon_step")
            evap = df_temp[df_temp["sensor_name"] == model.hparams["evap_name"]].sort_values(by="horizon_step")
            prec = df_temp[df_temp["sensor_name"] == model.hparams["prec_name"]].sort_values(by="horizon_step")

            temp.index = cur_index
            evap.index = cur_index
            prec.index = cur_index

            if member == 0:
                warm_up = (cur_index[0] - model.model.stressmodels["recharge"].prec.series.index[0]).days
                prediction = model.predict(stress_data=None, tmin=cur_index[0], tmax=cur_index[-1], warmup=warm_up)
                forecast_dict[-1, tstamp] = pd.DataFrame(prediction)

            new_prec = model.model.stressmodels["recharge"].prec.series.copy()
            new_prec.loc[cur_index] = prec["value"]

            new_evap = model.model.stressmodels["recharge"].evap.series.copy()
            new_evap.loc[cur_index] = evap["value"]

            new_stress_data = {
                "recharge": {
                    "prec": new_prec,  # 10% higher precipitation
                    "evap": new_evap,  # 5% lower evaporation
                }
            }
            if model.model.stressmodels["recharge"].temp is not None:
                new_temp = model.model.stressmodels["recharge"].temp.series.copy()
                new_temp.loc[cur_index] = temp["value"]

                new_stress_data["recharge"]["temp"] = new_temp

            warm_up = (cur_index[0] - new_prec.index[0]).days
            prediction = model.predict(
                stress_data=new_stress_data, tmin=cur_index[0], tmax=cur_index[-1], warmup=warm_up
            )
            forecast_dict[member, tstamp] = pd.DataFrame(prediction)

        df_pred_list = []
        for (member, tstamp), df_pred in forecast_dict.items():
            df_pred = df_pred.reset_index()
            df_pred = df_pred.melt(id_vars="index")
            df_pred["sensor_name"] = params["target_sensor_name"]
            df_pred["model_id"] = params["model_id"]

            df_pred["horizon_step"] = (df_pred["index"] - tstamp).dt.days
            df_pred["tstamp"] = tstamp
            df_pred = df_pred.drop(["variable"], axis=1)
            df_pred["member"] = member
            df_pred = df_pred[["sensor_name", "tstamp", "value", "model_id", "horizon_step", "member"]]
            df_pred_list.append(df_pred)

        df_pred = pd.concat(df_pred_list)
        self._save_forecasts(df_pred, model_name, params)

    def _handle_XGBoost(self, model_name, model, params) -> None:
        stmst = select(InputForecastsLong).where(
            InputForecastsLong.sensor_name.in_(model.columns),
            between(InputForecastsLong.tstamp, min(self.times), max(self.times)),
        )
        df_ext = pd.read_sql(sql=stmst, con=self.engine)

        # with self.Session.begin() as session:
        stmst = select(SensorData).where(
            SensorData.sensor_name == model.target_sensor_name,
            between(SensorData.tstamp, min(self.times), max(self.times)),
        )
        df_y = pd.read_sql(sql=stmst, con=self.engine)

        # prepare_test_df(df_ext,num_days=num_days,ensemble_num=0)

        # df_ens_dict = {}
        forecast_dict = {}
        for ens_num in range(51):
            df_test = prepare_test_df(
                df_ext, num_days=model.num_days, ensemble_num=ens_num
            )  # ,first_time=df_x.index.min(),last_time=df_x.index.max())
            forecasts = []
            for i, x in df_test.groupby("tstamp"):
                forecasts.append(
                    pd.DataFrame(model.predict(x.drop(["horizon_step", "tstamp"], axis=1)), index=x.index, columns=[i])
                )
                forecast_dict[ens_num] = forecasts

        tstamps = [df.columns[0] for df in forecast_dict[0]]
        forecast_dict = {
            tstamp: pd.concat([forecast_dict[i][j] for i in range(51)], axis=1) for j, tstamp in enumerate(tstamps)
        }

        for k, _ in forecast_dict.items():
            forecast_dict[k].columns = list(range(51))

        df_pred_list = []
        for k, df_pred in forecast_dict.items():
            if model.hparams.diff:
                # if k not in df_y["tstamp"]:
                if k not in set(df_y["tstamp"]):
                    logging.warning("Base Groundwater for %s requested, but not available.", k)
                    continue
                base = df_y[df_y["tstamp"] == k]["sensor_value"]
                df_pred = base.item() + df_pred.cumsum()

            df_pred = df_pred.reset_index()
            df_pred = df_pred.melt(id_vars="index")
            df_pred["sensor_name"] = params["target_sensor_name"]
            df_pred["model_id"] = params["model_id"]
            df_pred["horizon_step"] = (df_pred["index"] - k).dt.days
            df_pred["tstamp"] = k
            df_pred = df_pred.rename({"variable": "member"}, axis=1)
            df_pred = df_pred[["sensor_name", "tstamp", "value", "model_id", "horizon_step", "member"]]
            df_pred_list.append(df_pred)

        df_pred = pd.concat(df_pred_list)
        self._save_forecasts(df_pred, model_name, params)

    def _save_forecasts(self, df_pred, model_name, params) -> None:
        # insert as before
        stmt = select(
            PegelForecastsLong.tstamp,
            PegelForecastsLong.member,
            PegelForecastsLong.horizon_step,
            PegelForecastsLong.model_id,
        ).where(
            PegelForecastsLong.model_id.in_([params["model_id"]]),
            PegelForecastsLong.sensor_name == params["target_sensor_name"],
            PegelForecastsLong.tstamp.in_(df_pred["tstamp"].unique()),
        )
        df_already_inserted = pd.read_sql(sql=stmt, con=self.engine)
        merged = df_pred.merge(
            df_already_inserted[["tstamp", "member", "horizon_step", "model_id"]],
            on=["tstamp", "member", "horizon_step", "model_id"],
            how="left",
            indicator=True,
        )
        df_pred_create = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
        df_pred_update = merged[merged["_merge"] == "both"].drop("_merge", axis=1)
        df_pred_update["created"] = datetime.now(timezone.utc)

        logging.info("Inserting %d forecasts and updating %d forecasts", len(df_pred_create), len(df_pred_update))

        with self.Session.begin() as session:
            forecast_objs = df_pred_create.apply(lambda row: PegelForecastsLong(**row.to_dict()), axis=1).tolist()
            session.add_all(forecast_objs)
            session.execute(update(PegelForecastsLong), df_pred_update.to_dict(orient="records"))

        logging.info("Finished inserting/updating forecasts")
        if self.config["export_zrxp"]:
            self._export_zrxp(df_pred, model_name, params["model_id"])

    def add_zrxp_data(self, zrxp_folders: list[Path]) -> None:
        """Adds zrxp data to the database.

        Args:
            zrxp_folder (str or Path): The folder containing the zrxp files

        """
        for zrxp_folder in zrxp_folders:
            for zrxp_file in zrxp_folder.iterdir():
                if zrxp_file.suffix != ".zrx":
                    continue

                vhs_gebiet = zrxp_file.name.split("_")[1]  # [6:]

                df_zrxp = pd.read_csv(zrxp_file, skiprows=3, header=None, sep=" ", parse_dates=[0])

                try:
                    if len(df_zrxp.columns) == 4:
                        zrxp_time = df_zrxp.iloc[0, 0]
                        member = int(df_zrxp.iloc[0, 2])
                        forecast = df_zrxp[3]

                    else:
                        zrxp_time = df_zrxp.iloc[0, 0] - pd.Timedelta(hours=1)
                        member = 0
                        forecast = df_zrxp[1]
                    self.insert_external_forecast(zrxp_time, vhs_gebiet, forecast, member)
                except IntegrityError as e:
                    # Try to get the error code from the original exception
                    code = None
                    if hasattr(e, "orig") and hasattr(e.orig, "args") and len(e.orig.args) > 0:
                        if hasattr(e.orig.args[0], "code"):
                            code = e.orig.args[0].code
                    if code == 2291:
                        logging.warning(
                            "%s Does the sensor_name %s exist in MODEL_SENSOR.VHS_GEBIET?",
                            str(e),
                            vhs_gebiet,
                        )
                    else:
                        logging.warning(
                            "Values for %s,%s,%s already exist in database with different values: %s",
                            zrxp_time,
                            vhs_gebiet,
                            member,
                            str(e),
                        )

    def insert_external_forecast(self, time: pd.Timestamp, vhs_gebiet: str, forecast: pd.Series, member: int) -> None:
        """Inserts an external forecast into the database.

        Args:
            time (pd.Timestamp): The timestamp of the forecast.
            vhs_gebiet (str): The name of the forecast.
            forecast (pd.Series): The forecast values.
            member (int): The member identifier.

        Returns:
            None

        """
        sensor = self.get_sensor_name(vhs_gebiet)
        if sensor is None:
            return
        sensor_name = sensor.sensor_name

        with self.Session.begin() as session:
            # Get forecast if one already exists for this model, member and time. (If oracle implements upsert this should be replaced)
            stmt = select(InputForecastsLong).where(
                InputForecastsLong.tstamp == bindparam("tstamp"),
                InputForecastsLong.sensor_name == bindparam("sensor_name"),
                InputForecastsLong.member == bindparam("member"),
            )
            params = {"tstamp": time, "sensor_name": sensor_name, "member": member}
            input_forecast = session.scalars(stmt, params=params).all()

            # Create new forecast rows for the long-format table
            inp_forecasts = [
                InputForecastsLong(
                    sensor_name=sensor_name,
                    member=member,
                    tstamp=time,
                    horizon_step=i + 1,
                    value=forecast[i],
                )
                for i in range(sensor.forecast_length)
            ]

            # If any existing rows exist for this (tstamp, sensor_name, member),
            # remove them and re-insert the fresh long-format rows. This is
            # simpler and correct for the per-horizon long representation.
            if input_forecast is not None and input_forecast:
                session.execute(
                    InputForecastsLong.__table__.delete().where(
                        InputForecastsLong.tstamp == params["tstamp"],
                        InputForecastsLong.sensor_name == params["sensor_name"],
                        InputForecastsLong.member == params["member"],
                    )
                )
                session.add_all(inp_forecasts)
                logging.debug(
                    "Input Forecast %s %s member %s already exist in database, replacing values",
                    sensor_name,
                    time,
                    member,
                )
            else:
                session.add_all(inp_forecasts)

    # def _export_zrxp(self, forecast, gauge_config, end_time, member, sensor_name, model_name):
    def _export_zrxp(self, df_pred, model_name, model_id) -> None:
        df_save = df_pred[(df_pred["member"] == 0) & (df_pred["model_id"] == model_id)]
        df_save = df_save.rename({"tstamp": "timestamp"}, axis=1)

        df_save["forecast"] = df_save["timestamp"] + pd.to_timedelta(df_save["horizon_step"] + 1, unit="h")

        sensor_name = df_save["sensor_name"].iloc[0].replace(",", "_")

        for tstamp, df_temp in df_save.groupby("timestamp"):
            df_temp = df_temp.sort_values(by="horizon_step")
            df_temp = df_temp[["timestamp", "forecast", "member", "value"]]

            t_str = tstamp.strftime("%Y%m%d%H")
            target_file = self.config["zrxp_out_folder"] / f"{t_str}_{sensor_name}_{model_name}.zrx"

            with open(target_file, "w", encoding="utf-8") as file:
                file.write("#REXCHANGEWISKI." + model_name + ".W.KNN|*|\n")
                file.write("#RINVAL-777|*|\n")
                file.write("#LAYOUT(timestamp,forecast, member,value)|*|\n")

            df_temp.to_csv(
                path_or_buf=target_file, header=False, index=False, mode="a", sep=" ", date_format="%Y%m%d%H%M"
            )
            logging.info("Exported forecast into %s", target_file)

    def maybe_update_tables(self) -> None:
        """Update database tables based on configuration settings.

        If the 'load_zrxp' flag is set to True in the configuration,
        this method loads and adds zrxp forecast data from the configured
        zrxp_folder to the database.

        Returns:
            None

        """
        if self.config.get("load_zrxp"):
            self.add_zrxp_data(self.config["zrxp_folder"])

    def _get_member_list(self) -> list[int]:
        """Get list of ensemble member IDs based on configuration.

        Returns member identifiers according to configuration flags:
        - dummy: includes [-1] for dummy forecasts
        - single: includes [0] for deterministic forecast
        - ensemble: includes [1-20] for ensemble members

        The returned list can contain a combination of these depending
        on which flags are enabled in the configuration.

        Returns:
            List of integer member identifiers. Examples:
                - [0] for single deterministic forecast
                - [1, 2, ..., 20] for ensemble only
                - [-1, 0, 1, 2, ..., 20] if all flags enabled

        """
        member_list: list[int] = []

        if self.config.get("dummy"):
            member_list += [-1]
        if self.config.get("single"):
            member_list += [0]
        if self.config.get("ensemble"):
            member_list += list(range(1, 21))
        return member_list

    def _get_times(self) -> list[pd.Timestamp]:
        """Get list of timestamps for forecast generation based on configuration.

        If 'range' flag is True, returns hourly timestamps from 'start' to 'end'.
        Otherwise returns a single timestamp at 'start' time.

        Returns:
            List of pandas Timestamp objects representing times to forecast.
            Single-element list if range=False, multiple elements if range=True.

        """
        if self.config.get("range"):
            return list(pd.date_range(self.config["start"], self.config["end"], freq="h"))
        return [pd.to_datetime(self.config["start"])]

    def get_sensor_name(self, vhs_gebiet: str) -> InputForecastsMeta | None:
        """Look up sensor name for a given vhs_gebiet.

        Queries the INPUT_FORECASTS_META table to find the sensor configuration
        associated with the specified vhs_gebiet (flood management area).

        Args:
            vhs_gebiet: Name of the flood management area (vhs_gebiet).

        Returns:
            InputForecastsMeta object containing sensor metadata, or None if not found.

        """
        stmt = select(InputForecastsMeta).where(InputForecastsMeta.vhs_gebiet == bindparam("vhs_gebiet"))
        params = {"vhs_gebiet": vhs_gebiet}
        with self.Session() as session:
            return session.scalar(statement=stmt, params=params)

    def manually_add_sensor_data(self, data_file: Path, sensor_names: list[str] | None = None) -> None:
        """Manually add sensor data from CSV file to database.

        Loads sensor data from a CSV file and inserts rows into the SensorData table,
        skipping any rows that are already present in the database to avoid duplicates.

        Warning:
            Do not use DataFrames with shifted columns (columns where a forecast is
            simulated by shifting values by 48 hours).

        Args:
            data_file: Path to CSV file containing sensor data.
                Should have a datetime index and columns for each sensor.
            sensor_names: Optional list of sensor column names. If provided, these
                will replace the column names in the CSV file. Defaults to None.

        Returns:
            None

        """
        print(
            "WARNING: Do not use dataframes with 'shifted' columns, so columns where a forecast is faked by shifting the values by 48 hours"
        )
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if sensor_names is not None:
            df.columns = sensor_names

        df = pd.melt(df, var_name="sensor_name", value_name="sensor_value", ignore_index=False)
        stmt = select(SensorData).where(
            between(SensorData.tstamp, df.index.min(), df.index.max()),
            SensorData.sensor_name.in_(df["sensor_name"].unique()),
        )
        df_already_inserted = pd.read_sql(stmt, self.engine)
        df = df.reset_index()
        merged = df.merge(
            df_already_inserted[["tstamp", "sensor_name"]], on=["tstamp", "sensor_name"], how="left", indicator=True
        )
        df_create = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
        df_create = df_create.dropna()
        with self.Session() as session:
            sensor_data_objs = df_create.apply(lambda row: SensorData(**row.to_dict()), axis=1).tolist()
            session.add_all(sensor_data_objs)
            session.commit()


def prepare_test_df(
    df_cds: pd.DataFrame,
    num_days: int = 30,
    ensemble_num: int = 0,
    first_time: pd.Timestamp | None = None,
    last_time: pd.Timestamp | None = None,
    fill_nans: bool = True,
) -> pd.DataFrame:
    """Prepare and pivot forecast data for testing/validation.

    Transforms long-format forecast data into wide format with lagged features.
    Each forecast is indexed by timestamp and horizon, with additional columns
    for lagged values from previous forecasts.

    Args:
        df_cds: DataFrame with columns: tstamp, horizon_step, member, sensor_name, value.
            Should contain raw forecast data in long format.
        num_days: Number of lag days to include as additional features. Defaults to 30.
        ensemble_num: Which ensemble member to extract (typically 0 for deterministic).
            Defaults to 0.
        first_time: Optional start time filter. Data before this is removed.
            Defaults to None (no start filter).
        last_time: Optional end time filter. Data after this is removed.
            Defaults to None (no end filter).
        fill_nans: Whether to fill NaN values with forecasts from one month prior.
            Defaults to True.

    Returns:
        DataFrame with pivoted forecasts and lagged features. Index is the
        forecast valid time (tstamp + horizon_step as timedelta).
        Rows correspond to individual forecast times and horizons.

    Notes:
        - Drops rows with NaN values at the end
        - Reshapes data from long to wide format (one column per sensor)
        - Creates lag columns for temporal pattern modeling

    """
    df_test = (
        df_cds[df_cds["member"] == ensemble_num]
        .drop("member", axis=1)
        .pivot(columns="sensor_name", index=["horizon_step", "tstamp"], values="value")
    )
    df_test.columns = df_test.columns.str.lower().str.replace(" ", "_")
    df_test.reset_index(inplace=True)
    df_test.index = df_test["tstamp"] + pd.to_timedelta(df_test["horizon_step"], unit="days")

    value_cols = df_test.columns.drop(["horizon_step", "tstamp"])

    df_list = []
    for tstamp, b in df_test.groupby("tstamp"):
        df_temp = pd.concat([b, b[value_cols].shift(periods=range(1, num_days + 1), suffix="_lag")], axis=1)
        df_list.append(df_temp)

        month_before = tstamp - DateOffset(months=1)
        if fill_nans and month_before in df_test["tstamp"]:
            prev_month = df_test[df_test["tstamp"] == month_before]

            for lag in range(1, num_days + 1):
                lag_cols = value_cols + "_lag_" + str(lag)
                lag_col = lag_cols[0]
                # Fill NaNs in current month with values from previous month at the right offset
                mask = df_temp[lag_col].isna()
                if mask.any() and len(prev_month) >= lag:
                    target_tstamp = df_temp.loc[mask,].index - pd.Timedelta(days=lag)
                    # We have problem with months of different lengths otherwise.#TODO check 2 months before
                    if prev_month.index[0] in target_tstamp:
                        target_tstamp = target_tstamp[target_tstamp >= prev_month.index[0]]
                        mask_cutoff = list(mask.index - pd.Timedelta(days=lag)).index(prev_month.index[0])
                        mask[:mask_cutoff] = False
                    # print(tstamp,lag)
                    df_temp.loc[mask, lag_cols] = prev_month.loc[target_tstamp, value_cols].values

    df_test = pd.concat(df_list, axis=0)
    # df_test = df_test.drop(["step","month"],axis=1)
    df_test["day_of_year"] = df_test.index.dayofyear
    df_test["year"] = df_test.index.year

    if first_time is not None:
        df_test = df_test[df_test.index > first_time]

    if last_time is not None:
        df_test = df_test[df_test.index < last_time]

    return df_test.dropna()


def get_engine() -> Any:
    """Create and configure database engine from environment variables.

    Reads database configuration from .env file and creates SQLAlchemy engine.
    Supports both SQLite and Oracle databases. For Oracle, optionally initializes
    thick client mode for advanced features like connection pooling.

    Environment variables:
        db_kind: Database type - "sqlite" or "oracle". Defaults to "oracle".
        db_path: Path to SQLite database file (required for SQLite).
        db_username: Oracle username (required for Oracle).
        db_password: Oracle password (required for Oracle).
        db_dsn: Oracle DSN connection string (required for Oracle).
        lib_dir: Path to Oracle client library for thick mode (optional).

    Returns:
        SQLAlchemy Engine instance configured for the specified database.

    Raises:
        RuntimeError: If required environment variables are missing for the
            selected database type.

    """
    load_dotenv(".env")
    db_kind = os.getenv("db_kind", "oracle").lower()
    lib_dir = os.getenv("lib_dir")

    if db_kind == "sqlite":
        db_path = os.getenv("db_path")
        if not db_path:
            msg = "DB path missing for SQLite. Set db_path in .env"
            raise RuntimeError(msg)
        db_url = f"sqlite:///{db_path}"
        logging.info("Using SQLite database: %s", db_url)
    else:  # oracle
        db_params = {}
        db_params["user"] = os.getenv("db_username")
        db_params["password"] = os.getenv("db_password")
        db_params["dsn"] = os.getenv("db_dsn")
        if not (db_params.get("user") and db_params.get("password") and db_params.get("dsn")):
            msg = "DB params missing in .env. Expect db_username, db_password, db_dsn, potentially also lib_dir if thick client is needed"
            raise RuntimeError(msg)
        db_url = f"oracle+oracledb://{db_params['user']}:{db_params['password']}@{db_params['dsn']}"
        if lib_dir:
            import oracledb

            logging.info("Initiating Thick mode with executable %s", lib_dir)
            oracledb.init_oracle_client(lib_dir=lib_dir)
        else:
            logging.info("Initiating Thin mode")

    return create_engine(db_url)
