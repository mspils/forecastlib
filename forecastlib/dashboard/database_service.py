"""Database service for forecast monitoring dashboard.

This module handles all database operations and queries for the forecast monitoring system.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml
from sqlalchemy import and_, between, desc, distinct, func, select
from sqlalchemy.orm import Session

from forecastlib.dashboard.constants import (
    BUFFER_TIME,
    LOOKBACK_EXTERNAL_FORECAST_EXISTENCE,
    NUM_RECENT_FORECASTS,
    NUM_RECENT_LOGS,
)
from forecastlib.database.orm_classes import (
    InputForecastsLong,
    Log,
    Metric,
    Model,
    ModelSensor,
    PegelForecastsLong,
    SensorData,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for database operations related to forecast monitoring."""

    def __init__(self, engine) -> None:
        """Initialize the database service with a SQLAlchemy engine.

        Args:
            engine: SQLAlchemy database engine

        """
        self.engine = engine

    def get_active_models_status(self, debug_timestamp: pd.Timestamp | None = None) -> dict | None:
        """Get status of all active models and their recent forecasts.

        Args:
            debug_timestamp: Optional timestamp for debugging/static testing

        Returns:
            Dictionary containing model status info or None on error

        """
        try:
            now = debug_timestamp or datetime.now(timezone.utc)
            last_required_hour = now - BUFFER_TIME
            last_required_hour = last_required_hour.replace(minute=0, second=0, microsecond=0)
            while last_required_hour.hour % 3 != 0:
                last_required_hour -= timedelta(hours=1)

            with Session(self.engine) as session:
                active_models = session.query(Model).filter(Model.is_active == 1).all()

                model_status = []
                for model in active_models:
                    hparams = yaml.load(model.blob.yaml, Loader=yaml.FullLoader)
                    target_sensor = model.target_sensor_name

                    current_forecast = (
                        session.query(PegelForecastsLong)
                        .filter(
                            and_(
                                PegelForecastsLong.model_id == model.id,
                                PegelForecastsLong.tstamp >= last_required_hour,
                            )
                        )
                        .first()
                    )

                    last_valid_forecast = (
                        session.query(PegelForecastsLong)
                        .filter(
                            and_(
                                PegelForecastsLong.model_id == model.id,
                                PegelForecastsLong.value.isnot(None),
                            )
                        )
                        .order_by(PegelForecastsLong.tstamp.desc())
                        .first()
                    )

                    model_status.append(
                        {
                            "model_name": model.model_name,
                            "model_id": model.id,
                            "freq": hparams["freq"],
                            "sensor_name": target_sensor,
                            "has_current_forecast": current_forecast is not None,
                            "last_forecast_time": (last_valid_forecast.tstamp if last_valid_forecast else None),
                            "forecast_created": (last_valid_forecast.created if last_valid_forecast else None),
                        }
                    )

            logger.info("Model status check completed: %d active models", len(model_status))
            return {
                "model_status": model_status,
                "last_check_time": now,
                "required_timestamp": last_required_hour,
            }

        except Exception as e:
            logger.exception("Error getting model status: %s", str(e))
            return None

    def get_recent_forecasts(
        self, model_id: int, freq="1h", debug_timestamp: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Get recent forecasts for a specific model.

        Args:
            model_id: ID of the model
            debug_timestamp: Optional timestamp for debugging/static testing

        Returns:
            DataFrame with recent forecasts

        """
        try:
            now = debug_timestamp or datetime.now(timezone.utc)
            time_threshold = now - NUM_RECENT_FORECASTS * pd.Timedelta(hours=3)

            with Session(self.engine) as session:
                recent_forecasts = (
                    session.query(PegelForecastsLong)
                    .filter(
                        PegelForecastsLong.model_id == model_id,
                        PegelForecastsLong.tstamp >= time_threshold,
                    )
                    .all()
                )
                if not recent_forecasts:
                    logger.debug(
                        "No forecasts found after %s for model_id %d, getting most recent",
                        time_threshold,
                        model_id,
                    )
                    max_tstamp = (
                        session.query(func.max(PegelForecastsLong.tstamp))
                        .filter(PegelForecastsLong.model_id == model_id)
                        .scalar()
                    )

                    if max_tstamp:
                        recent_forecasts = (
                            session.query(PegelForecastsLong)
                            .filter(
                                PegelForecastsLong.model_id == model_id,
                                PegelForecastsLong.tstamp == max_tstamp,
                            )
                            .all()
                        )
                        logger.debug("Using most recent forecast timestamp: %s", max_tstamp)

            if recent_forecasts:
                df = pd.DataFrame(
                    [
                        {
                            "tstamp": fcst.tstamp,
                            "member": fcst.member,
                            "horizon_step": fcst.horizon_step,
                            "value": fcst.value,
                            "created": fcst.created,
                        }
                        for fcst in recent_forecasts
                    ]
                )
                df.dropna(subset=["value"], inplace=True)

                try:
                    time_delta = pd.to_timedelta(freq)
                except ValueError as e:
                    if freq == "h":
                        time_delta = pd.Timedelta(hours=1)
                    elif freq in {"d", "D"}:
                        time_delta = pd.Timedelta(days=1)
                    elif freq in {"w", "W"}:
                        time_delta = pd.Timedelta(weeks=1)
                    else:
                        raise ValueError(e)
                df["target_time"] = df["tstamp"] + df["horizon_step"] * time_delta

                logger.debug("Retrieved %d recent forecasts for model_id %d", len(df), model_id)
                return df
            logger.debug("No forecasts found for model_id %d", model_id)
            return pd.DataFrame()

        except Exception as e:
            logger.exception("Error getting recent forecasts for model_id %d: %s", model_id, str(e))
            return pd.DataFrame()

    def get_input_forecasts(
        self, sensor_names: list[str], debug_timestamp: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Get input forecasts for given sensor names.

        Args:
            sensor_names: List of sensor names
            debug_timestamp: Optional timestamp for debugging/static testing

        Returns:
            Tuple of (DataFrame with input forecasts, list of sensors with external forecasts)

        """
        try:
            now = debug_timestamp or datetime.now(timezone.utc)

            stmt = select(InputForecastsLong).where(
                InputForecastsLong.sensor_name.in_(sensor_names),
                between(
                    InputForecastsLong.tstamp,
                    now - (NUM_RECENT_FORECASTS + 1) * pd.Timedelta(hours=3),
                    now + pd.Timedelta(hours=24),
                ),
            )

            df = pd.read_sql(sql=stmt, con=self.engine)

            with Session(self.engine) as session:
                existing = (
                    session.query(distinct(InputForecastsLong.sensor_name))
                    .filter(InputForecastsLong.sensor_name.in_(sensor_names))
                    .filter(InputForecastsLong.tstamp >= now - timedelta(days=LOOKBACK_EXTERNAL_FORECAST_EXISTENCE))
                    .all()
                )

            ext_forecast_names = [x[0] for x in existing]
            logger.debug("Retrieved %d input forecasts for %d sensors", len(df), len(sensor_names))
            return df, ext_forecast_names

        except Exception as e:
            logger.exception("Error getting input forecasts: %s", str(e))
            msg = f"Error getting input forecasts: {e!s}"
            raise RuntimeError(msg) from e

    def get_recent_logs(self, sensor_name: str | None) -> list[dict]:
        """Get recent logs for a specific sensor.

        Args:
            sensor_name: Name of the sensor

        Returns:
            List of log entries as dictionaries

        """
        if sensor_name is None:
            return []

        try:
            with Session(self.engine) as session:
                logs = (
                    session.query(Log)
                    .filter(Log.gauge == sensor_name)
                    .order_by(desc(Log.created))
                    .limit(NUM_RECENT_LOGS)
                    .all()
                )

                return [
                    {
                        "timestamp": (log.created.strftime("%Y-%m-%d %H:%M:%S") if log.created else "N/A"),
                        "level": log.loglevelname or "N/A",
                        "sensor": log.gauge or "N/A",
                        "message": log.message or "N/A",
                        "module": log.module or "N/A",
                        "function": log.funcname or "N/A",
                        "line": log.lineno or 0,
                        "exception": log.exception or "",
                    }
                    for log in logs
                ]
        except Exception as e:
            logger.exception("Error getting logs for sensor %s: %s", sensor_name, str(e))
            return []

    def get_historical_data(self, model_id: int, debug_timestamp: pd.Timestamp | None = None) -> pd.DataFrame:
        """Get last 144 hours of sensor data for all sensors associated with the model.

        Args:
            model_id: ID of the model
            debug_timestamp: Optional timestamp for debugging/static testing

        Returns:
            DataFrame with historical sensor data

        """
        try:
            with Session(self.engine) as session:
                model_sensors = session.query(ModelSensor).filter(ModelSensor.model_id == model_id).all()

                sensor_names = [ms.sensor_name for ms in model_sensors]

                now = debug_timestamp or datetime.now(timezone.utc)
                time_threshold = now - timedelta(hours=144) - BUFFER_TIME

                stmt = select(SensorData).where(
                    SensorData.tstamp >= time_threshold,
                    SensorData.sensor_name.in_(sensor_names),
                )

                df = pd.read_sql(sql=stmt, con=self.engine, index_col="tstamp")
                df = df.pivot(columns="sensor_name", values="sensor_value")  # [sensor_names]

                logger.debug(
                    "Retrieved historical data for model_id %d: %d/%d sensors, %d rows",
                    model_id,
                    len(df.columns),
                    len(sensor_names),
                    len(df),
                )
                return df

        except Exception as e:
            logger.exception("Error getting historical data for model_id %d: %s", model_id, str(e))
            return pd.DataFrame()

    def get_model_metrics(self, model_id: int, metric_name: str = "mae") -> tuple[pd.DataFrame | None, list | None]:
        """Get metrics for a specific model.

        Args:
            model_id: ID of the model
            metric_name: Name of the metric to retrieve

        Returns:
            Tuple of (DataFrame with metrics, time boundaries)

        """
        try:
            metric_names = [
                f"train_{metric_name}",
                f"test_{metric_name}",
                f"val_{metric_name}",
                f"train_{metric_name}_flood",
                f"test_{metric_name}_flood",
                f"val_{metric_name}_flood",
            ]

            stmt = select(Metric).where(Metric.model_id == model_id, Metric.metric_name.in_(metric_names))

            df = pd.read_sql(sql=stmt, con=self.engine)

            try:
                df = df.pivot(index="horizon_step", columns="metric_name", values="value")
            except ValueError as e:
                logger.debug("Error pivoting metrics DataFrame: %s, trying again with tags", str(e))
                df = df.pivot(index="horizon_step", columns=["metric_name", "tag"], values="value")
                df.columns = [f"{x}_{y}" for (x, y) in df.columns.to_flat_index()]
                df.columns = df.columns.str.replace("_normal", "")

            logger.debug("Retrieved metrics for model_id %d: %d metric_names", model_id, len(metric_names))
            return df, None

        except Exception as e:
            logger.exception("Error getting model metrics for model_id %d: %s", model_id, str(e))
            return None, None

    def get_confidence(self, model_id: int, flood: bool = False, subset: str = "test") -> pd.DataFrame:
        """Get confidence interval metrics for a specific model.

        Args:
            model_id: ID of the model
            flood: Whether to get flood confidence metrics
            subset: Data subset ('train', 'test', 'val')

        Returns:
            DataFrame with confidence metrics

        """
        try:
            if flood:
                metric_names = [
                    f"{subset}_conf05_flood",
                    f"{subset}_conf10_flood",
                    f"{subset}_conf50_flood",
                ]
            else:
                metric_names = [
                    f"{subset}_conf05",
                    f"{subset}_conf10",
                    f"{subset}_conf50",
                ]

            stmt = select(Metric).where(Metric.model_id == model_id, Metric.metric_name.in_(metric_names))
            df = pd.read_sql(sql=stmt, con=self.engine)

            try:
                df = df.pivot(index="horizon_step", columns="metric_name", values="value")
            except ValueError as e:
                logger.debug("Error pivoting confidence DataFrame: %s, trying again with tags", str(e))
                df = df.pivot(index="horizon_step", columns=["metric_name", "tag"], values="value")
                df.columns = [f"{x}_{y}" for (x, y) in df.columns.to_flat_index()]
                df.columns = df.columns.str.replace("_normal", "")

            logger.debug("Retrieved confidence metrics for model_id %d", model_id)
            return df

        except Exception as e:
            logger.exception("Error getting confidence interval for model_id %d: %s", model_id, str(e))
            return pd.DataFrame()

    def get_all_logs(self, limit: int = 1000) -> list[dict]:
        """Get all logs from the database.

        Args:
            limit: Maximum number of logs to retrieve

        Returns:
            List of log entries as dictionaries

        """
        try:
            with Session(self.engine) as session:
                logs = session.query(Log).order_by(desc(Log.created)).limit(limit).all()

                return [
                    {
                        "timestamp": log.created.strftime("%Y-%m-%d %H:%M:%S") if log.created else "N/A",
                        "level": log.loglevelname,
                        "sensor": log.gauge,
                        "message": log.message,
                        "module": log.module,
                        "function": log.funcname,
                        "line": log.lineno,
                        "exception": log.exception or "",
                    }
                    for log in logs
                ]

        except Exception as e:
            logger.exception("Error getting system logs: %s", str(e))
            return []
