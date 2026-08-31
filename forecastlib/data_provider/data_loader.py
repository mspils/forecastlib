"""Data loading utilities for time series forecasting datasets.

This module provides various PyTorch Dataset implementations for loading and
preprocessing time series data from different sources.

Supports multiple data splits (train/val/test), feature modes (univariate/multivariate),
scaling, time encoding, and data augmentation.
"""

import glob
import os
import re
import warnings
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from forecastlib.utils.augmentation import run_augmentation_single
from forecastlib.utils.timefeatures import FREQ_MAP, get_data_stamp, time_features, time_features_from_frequency_str

warnings.filterwarnings("ignore")

Features = Literal["M", "MS", "S"]
FEATURE_TYPES = get_args(Features)


class BaseForecastDataset(Dataset):
    """Shared initialization for the forecasting datasets in this module.

    Subclasses resolve their own ``size`` defaults, call ``super().__init__``
    with a ``(seq_len, label_len, pred_len)`` tuple, then perform any
    dataset-specific setup before calling ``self.__read_data__()``.
    """

    def __init__(
        self,
        args,
        root_path,
        data_path,
        flag,
        size,
        features,
        target,
        timeenc,
        freq,
    ) -> None:
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len, self.label_len, self.pred_len = size

        assert flag in {"train", "val", "test"}
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        if features not in FEATURE_TYPES:
            msg = f"Invalid feature type: {features!r}"
            raise ValueError(msg)
        self.features = features

        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path


class Dataset_Custom(BaseForecastDataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ) -> None:
        if size is None:
            size = (24 * 4 * 4, 24 * 4, 24 * 4)
        super().__init__(args, root_path, data_path, flag, size, features, target, timeenc, freq)

        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date", *cols, self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:  # "S", validated in __init__
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            msg_0 = f"Invalid timeenc value: {self.timeenc}"
            raise ValueError(msg_0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Diff(BaseForecastDataset):
    _supports_multicol = True  # default

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=(144, 48, 48),
        features="MS",
        data_path="example.csv",
        target="OT",
        timeenc=0,
        freq="h",
    ) -> None:
        if size is None:
            # TODO nicht size nehmen sonder args.seq_len etc.?
            size = (144, 48, 48)
        super().__init__(args, root_path, data_path, flag, size, features, target, timeenc, freq)

        self.date_col = args.date_col
        self.diff = args.diff

        self.is_npy = self.data_path.endswith(".npy")
        if self.is_npy:
            self.stride = 100
            self.num_time_features = (
                FREQ_MAP[args.freq] if self.args.embed == "timeF" else len(time_features_from_frequency_str(args.freq))
            )

        else:
            self.stride = 1

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.is_npy:
            data = np.load(os.path.join(self.root_path, self.data_path))
            df_raw = pd.DataFrame(data)
        elif self.data_path.endswith(".csv"):
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=[self.date_col])
        else:
            msg = f"Unsupported file format: {self.data_path}"
            raise ValueError(msg)

        # fill missing values #TODO this is only for the LFU
        max_fill = 200
        na_count = df_raw.isna().sum(axis=0)
        mask = (
            df_raw.columns.str.contains("NEW")
            | df_raw.columns.str.contains("NVh")
            | (df_raw.columns.str.startswith("N") & df_raw.columns.str.endswith("_mm"))
        )

        prec_cols = list(na_count[mask][na_count[mask] > 0].index)
        if len(prec_cols) > 0:
            df_raw.loc[:, mask] = df_raw.loc[:, mask].fillna(0)

        # interpolate data in all other columns
        df_raw = df_raw.interpolate(limit=max_fill, limit_direction="both")

        if df_raw.isna().sum().sum() > 0:
            msg = (
                f"Some columns were missing more than {max_fill} continuous values, either raise the limit or fill values manually."
                f"{df_raw.isna().sum().sum()} still missing,"
            )
            raise ValueError(msg)

        # df_raw.columns: ['date', ...(other features), target feature]
        if not self.is_npy:
            df_raw.rename(columns={self.date_col: "date"}, inplace=True)

        try:
            if self.args.borders_start is None:
                num_train = int(len(df_raw) * 0.7)
                num_test = int(len(df_raw) * 0.2)
                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
            else:
                # TODO assert compat boarders_start + dataset
                assert self.args.data_path in {"ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"}, (
                    "This option should only be used in combination with the ETT datasets"
                )
                border1s = [
                    self.args.borders_start[0],
                    self.args.borders_start[1] - self.seq_len,
                    self.args.borders_start[2] - self.seq_len,
                ]
                border2s = self.args.borders_end
        except (KeyError, AttributeError):
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.args.diff:
            if self.features == "M":
                cols = list(df_raw.columns)
                if not self.is_npy:
                    cols.remove("date")
            else:  # "MS" or "S", validated in __init__
                cols = [self.target]

            df_diff = df_raw[cols].diff()
            df_diff.iloc[0] = 0
            df_diff.columns = [str(x) + "_d1" for x in df_diff.columns]
            df_raw = pd.concat([df_raw, df_diff], axis=1)

        if not self.is_npy:
            cols = list(df_raw.columns)
            cols.remove("date")
            df_raw = df_raw[["date", *cols]]

        if self.features in {"M", "MS"}:
            if not self.is_npy:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            else:
                df_data = df_raw
        elif self.features == "S" and self.args.diff:
            df_data = df_raw[[self.target, f"{self.target}_d1"]]
        else:
            df_data = df_raw[[self.target]]

        self.data_x_raw = df_data[border1:border2]
        train_data = df_data[border1s[0] : border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        if not self.is_npy:
            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            self.index = df_stamp["date"].copy()
            data_stamp = get_data_stamp(df_stamp, self.timeenc, self.freq)
            self.data_stamp = torch.tensor(data_stamp, dtype=torch.float32)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        # TODO Alternativ könnte man auch im trainings loop auf float runter gehen. Oder das ganze in das data_module verschieben.
        self.data_x = torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y = torch.tensor(self.data_y, dtype=torch.float32)

        self.width = self.data_x.shape[1]

    def __getitem__(self, index):

        if index > len(self):
            msg = f"Index {index} out of range, max index is {len(self)}"
            raise IndexError(msg)

        s_begin = index
        if self.is_npy:
            # s_begin = index % n_timepoint  # select start timestamp
            s_begin = self.stride * s_begin

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if not self.is_npy:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        else:
            seq_x_mark = torch.zeros((seq_x.shape[0], self.num_time_features))
            seq_y_mark = torch.zeros((seq_x.shape[0], self.num_time_features))

        if self.diff:
            if self.features == "M":
                seq_x[0, self.width // 2 :] = 0
            elif self.features in {"MS", "S"}:
                seq_x[0, -1] = 0

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        if self.is_npy:
            return (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_pred_index(self):
        return self.index[self.seq_len + -1 : -self.pred_len]


class Dataset_MW(BaseForecastDataset):
    def __init__(
        self,
        args,
        root_path,
        data_path,
        flag="train",
        size=(144, 48, 48),
        features="MS",
        target="GWL",
        timeenc=0,
        freq="D",
    ) -> None:
        if size is None:
            size = (144, 48, 48)
        super().__init__(args, root_path, data_path, flag, size, features, target, timeenc, freq)

        self.train_end = "2007-12-31"
        self.val_end = "2012-12-31"

        if self.set_type == 0:
            self.dynamic_scaler = StandardScaler()
        else:
            self.dynamic_scaler = args.dynamic_scaler

        self.static_path = os.path.join(root_path, "static/static_features_MW_1toMW_3207.csv")

        self.diff = getattr(args, "diff", False)
        self.__read_data__()

        self.scaler = StandardScaler()
        # mean_/scale_ are typed Optional but are always set once the scalers are fitted.
        self.scaler.mean_ = np.concat([self.dynamic_scaler.mean_, self.scaler_static.mean_])  # pyright: ignore[reportArgumentType, reportCallIssue]
        self.scaler.scale_ = np.concat([self.dynamic_scaler.scale_, self.scaler_static.scale_])  # pyright: ignore[reportArgumentType,reportCallIssue]

    def __read_data__(self):
        import glob

        # Get all MW files
        mw_files = glob.glob(os.path.join(self.root_path, self.data_path, "MW_*.csv"))
        self.mw_ids = sorted([os.path.basename(f).replace(".csv", "") for f in mw_files])

        # print("ACHTUNG NUR 100 BRUNNGEN data_loader read_data")
        # self.mw_ids = self.mw_ids[:80]
        self.train_mws = self.mw_ids
        self.val_mws = self.mw_ids
        self.test_mws = self.mw_ids
        # Split wells: 70% train, 20% val, 10% test
        # np.random.seed(42)  # for reproducibility
        # np.random.shuffle(self.mw_ids)
        # n_total = len(self.mw_ids)
        # n_train = int(0.7 * n_total)
        # n_val = int(0.2 * n_total)
        # self.train_mws = self.mw_ids[:n_train]
        # self.val_mws = self.mw_ids[n_train:n_train + n_val]
        # self.test_mws = self.mw_ids[n_train + n_val:]

        if self.set_type == 0:
            relevant_mws = self.train_mws
        elif self.set_type == 1:
            relevant_mws = self.val_mws
        else:
            relevant_mws = self.test_mws

        self.well_data = {}
        self.mw_list = []
        self.cutoffs = [0]

        static_df = pd.read_csv(self.static_path, index_col=0)
        static_df.set_index("MW_ID", inplace=True)

        static_df.drop(
            [
                "Proj_ID",
                "Operator",
                "Depth",
                "UpFilter",
                "LoFilter",
                "ScrLength",
                "PreState",  #'Pumping',
                "Easting (EPSG:3035)",
                "Northing (EPSG:3035)",
            ],
            axis=1,
            inplace=True,
        )

        # Encode categorical columns
        categorical_cols = [
            "AquiferMed",
            "HUEK250_HU",
            "HUEK250_RT",
            "HUEK250_CT",
            "HUEK250_DC",
            "HUEK250_GC",
            "HUMUS1000_OC",
        ]
        self.encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            static_df[col] = le.fit_transform(static_df[col])  # pyright: ignore[reportArgumentType, reportCallIssue]
            self.encoders[col] = le

        static_train = static_df.loc[self.train_mws]
        self.scaler_static = StandardScaler().fit(static_train.values)
        static_df = pd.DataFrame(
            self.scaler_static.transform(static_df), columns=static_df.columns, index=static_df.index
        )

        for mw_id in relevant_mws:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, f"{mw_id}.csv"))

            df_raw.rename(columns={"Unnamed: 0": "date"}, inplace=True)
            df_raw["date"] = pd.to_datetime(df_raw["date"])

            if self.set_type == 0:
                df_raw = df_raw[df_raw["date"] <= self.train_end]
            elif self.set_type == 1:
                df_raw = df_raw[(df_raw["date"] > self.train_end) & (df_raw["date"] <= self.val_end)]
            elif self.set_type == 2:
                df_raw = df_raw[df_raw["date"] > self.val_end]

            # Load static
            if mw_id not in static_df.index:
                print(f"Skipping {mw_id}, no static")
                continue
            static_row = static_df.loc[mw_id]
            static_features = self.scaler_static.transform(static_row.values.reshape(1, -1)).flatten()
            static_features = static_row.values

            if self.diff:
                cols = [self.target] if self.features in ["S", "MS"] else list(df_raw.columns[1:])
                df_diff = df_raw[cols].diff()
                df_diff.iloc[0] = 0
                df_diff.columns = df_diff.columns + "_d1"
                df_raw = pd.concat([df_raw, df_diff], axis=1)

            cols = list(df_raw.columns)
            cols.remove("date")
            df_raw = df_raw[["date", *cols]]

            if self.features in {"M", "MS"}:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == "S" and self.diff:
                df_data = df_raw[[self.target, f"{self.target}_d1"]]
            else:
                df_data = df_raw[[self.target]]
            if self.set_type == 0:
                self.dynamic_scaler.partial_fit(df_data.values)

            df_stamp = df_raw[["date"]]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            data_stamp = get_data_stamp(df_stamp, self.timeenc, self.freq)

            # data_x = torch.tensor(data, dtype=torch.float32)
            data_stamp = torch.tensor(data_stamp, dtype=torch.float32)

            # Add windows
            num_windows = len(df_data.values) - self.seq_len - self.pred_len + 1
            if num_windows > 0:
                self.well_data[mw_id] = {
                    "data_x": df_data.values,
                    "data_stamp": data_stamp,
                    "width_dynamic": df_data.values.shape[1],
                    "static": static_features,
                    "data_x_raw": df_raw.set_index("date"),  # df_data  # without static
                }

                self.mw_list.append(mw_id)
                self.cutoffs.append(self.cutoffs[-1] + num_windows)

        for mw_id in self.mw_list:
            self.well_data[mw_id]["data_x"] = torch.tensor(
                self.dynamic_scaler.transform(self.well_data[mw_id]["data_x"]), dtype=torch.float32
            )

    @property
    def data_x_raw(self):
        first_well = next(iter(self.well_data.keys()))
        return self.well_data[first_well]["data_x_raw"]

    def __getitem__(self, index):
        # Find the MW and the offset
        mw_id, i = None, None
        for j in range(len(self.cutoffs) - 1):
            if index < self.cutoffs[j + 1]:
                mw_id = self.mw_list[j]
                i = index - self.cutoffs[j]
                break

        data = self.well_data[mw_id]

        s_begin = i
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data["data_x"][s_begin:s_end]
        seq_y = data["data_x"][r_begin:r_end]
        seq_x_mark = data["data_stamp"][s_begin:s_end]
        seq_y_mark = data["data_stamp"][r_begin:r_end]

        if self.diff:
            if self.features == "M":
                seq_x[0, data["width_dynamic"] // 2 :] = 0
            elif self.features in {"MS", "S"}:
                seq_x[0, -1] = 0

        # Add static features
        static_repeated = np.tile(data["static"], (self.seq_len, 1))
        seq_x = torch.cat([seq_x, torch.tensor(static_repeated, dtype=torch.float32)], dim=1)

        # For seq_y, also add static
        static_repeated_y = np.tile(data["static"], (self.label_len + self.pred_len, 1))
        seq_y = torch.cat([seq_y, torch.tensor(static_repeated_y, dtype=torch.float32)], dim=1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return self.cutoffs[-1]

    def inverse_transform(self, data):
        # Since multiple scalers, this is not straightforward; perhaps implement per sample if needed
        return data

    def get_pred_index(self):
        # Per well, but for simplicity
        return []
