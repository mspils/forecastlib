"""ORM classes adapted to new database DDL.
This file was updated to match the new schema in the provided DLL (long forecast
tables and modified model/metric fields).
"""

import datetime

from sqlalchemy import (
    INTEGER,
    TIMESTAMP,
    VARCHAR,
    Double,
    ForeignKeyConstraint,
    Identity,
    Index,
    LargeBinary,
    PrimaryKeyConstraint,
    Sequence,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# pylint: disable=unsubscriptable-object


class Base(DeclarativeBase):
    pass


class Log(Base):
    __tablename__ = "log"
    __table_args__ = (PrimaryKeyConstraint("id", name="sys_c008734"),)

    id: Mapped[int] = mapped_column(
        INTEGER,
        Identity(
            on_null=False,
            start=1,
            increment=1,
            minvalue=1,
            maxvalue=9999999999999999999999999999,
            cycle=False,
            cache=20,
            order=False,
        ),
        primary_key=True,
    )
    created: Mapped[datetime.datetime | None] = mapped_column(TIMESTAMP)
    loglevelname: Mapped[str | None] = mapped_column(VARCHAR(32))
    message: Mapped[str | None] = mapped_column(VARCHAR(256))
    module: Mapped[str | None] = mapped_column(VARCHAR(64))
    funcname: Mapped[str | None] = mapped_column(VARCHAR(64))
    lineno: Mapped[int | None] = mapped_column(INTEGER)
    exception: Mapped[str | None] = mapped_column(VARCHAR(256))
    gauge: Mapped[str | None] = mapped_column(VARCHAR(256))


class InputForecastsMeta(Base):
    __tablename__ = "input_forecasts_meta"
    __table_args__ = (PrimaryKeyConstraint("sensor_name", "vhs_gebiet", name="UNIQUE_EXT_FORECASTS"),)

    sensor_name: Mapped[str] = mapped_column(VARCHAR(256), primary_key=True)
    vhs_gebiet: Mapped[str] = mapped_column(VARCHAR(256), primary_key=True)
    ensemble_members: Mapped[int] = mapped_column(INTEGER)
    forecast_length: Mapped[int] = mapped_column(INTEGER)

    def __repr__(self) -> str:
        return (
            f"InputForecastsMeta(sensor_name='{self.sensor_name}', "
            f"vhs_gebiet='{self.vhs_gebiet}', "
            f"ensemble_members={self.ensemble_members}, "
            f"forecast_length={self.forecast_length})"
        )


class Model(Base):
    __tablename__ = "model"
    __table_args__ = (PrimaryKeyConstraint("id", name="SYS_C009031"), Index("id_modellname", "model_name", unique=True))

    id: Mapped[int] = mapped_column(
        INTEGER,
        # Identity(
        #     start=1,
        #     increment=1,
        #     minvalue=1,
        #     maxvalue=9999999999,
        #     cycle=False,
        #     cache=20,
        #     order=False
        # ),
        Sequence("MODELL_SEQ"),
        primary_key=True,
    )

    model_name: Mapped[str] = mapped_column(VARCHAR(256), nullable=False)
    description: Mapped[str | None] = mapped_column(VARCHAR(1024))
    target_sensor_name: Mapped[str | None] = mapped_column(VARCHAR(256))
    in_size: Mapped[int | None] = mapped_column(INTEGER)
    kommentar: Mapped[str | None] = mapped_column(VARCHAR(1024))
    is_active: Mapped[int | None] = mapped_column(INTEGER)
    is_ensemble: Mapped[int | None] = mapped_column(INTEGER)
    is_variance: Mapped[int | None] = mapped_column(INTEGER)
    kind: Mapped[str | None] = mapped_column(VARCHAR(100))

    metric: Mapped[list["Metric"]] = relationship("Metric", back_populates="model")
    model_sensor: Mapped[list["ModelSensor"]] = relationship("ModelSensor", back_populates="model")
    # Each Modell must have exactly one ModellBlob; model-side relationship is required
    blob: Mapped["ModelBlob"] = relationship(
        "ModelBlob", back_populates="model", uselist=False, cascade="all, delete-orphan"
    )
    # Relationship to ensemble members where this Model is the ensemble (many children)
    ensemble_members: Mapped[list["EnsembleMember"]] = relationship(
        "EnsembleMember",
        back_populates="ensemble_model",
        foreign_keys="EnsembleMember.ensemble_model_id",
        cascade="all, delete-orphan",
    )
    # Relationship to ensemble entries where this Model is the submodel (i.e. part of other ensembles)
    submodel_members: Mapped[list["EnsembleMember"]] = relationship(
        "EnsembleMember",
        back_populates="submodel_model",
        foreign_keys="EnsembleMember.submodel_model_id",
    )

    def __repr__(self) -> str:
        return f"Model(id={self.id}, modelname='{self.model_name}', aktiv={self.is_active})"


class Sensor(Base):
    __tablename__ = "sensor"
    __table_args__ = (
        PrimaryKeyConstraint("sensor_name", name="sensor_name"),
        Index("sensor_mstnr", "mstnr"),
        Index("sensor_mstnr_sensor_name", "mstnr", "sensor_name"),
    )

    sensor_name: Mapped[str] = mapped_column(VARCHAR(256), primary_key=True)
    mstnr: Mapped[str | None] = mapped_column(VARCHAR(30))
    parameter_kurzname: Mapped[str | None] = mapped_column(VARCHAR(30))
    ts_shortname: Mapped[str | None] = mapped_column(VARCHAR(50))
    beschreibung: Mapped[str | None] = mapped_column(VARCHAR(500))
    has_ext_forecast: Mapped[int | None] = mapped_column(INTEGER)

    # input_forecasts: Mapped[List['InputForecasts']] = relationship('InputForecasts', back_populates='sensor')
    input_forecasts_long: Mapped[list["InputForecastsLong"]] = relationship(
        "InputForecastsLong", back_populates="sensor"
    )
    model_sensor: Mapped[list["ModelSensor"]] = relationship("ModelSensor", back_populates="sensor")
    # pegel_forecasts: Mapped[List['PegelForecasts']] = relationship('PegelForecasts', back_populates='sensor')
    pegel_forecasts_long: Mapped[list["PegelForecastsLong"]] = relationship(
        "PegelForecastsLong", back_populates="sensor"
    )
    sensor_data: Mapped[list["SensorData"]] = relationship("SensorData", back_populates="sensor")

    def __repr__(self) -> str:
        return (
            f"Sensor(sensor_name='{self.sensor_name}', "
            f"mstnr='{self.mstnr}', "
            f"parameter_kurzname='{self.parameter_kurzname}', "
            f"ts_shortname='{self.ts_shortname}')"
        )


class TmpSensorData(Base):
    __tablename__ = "tmp_sensor_data"
    __table_args__ = (PrimaryKeyConstraint("tstamp", "sensor_name", name="tmp_sensordata_id"),)

    tstamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, primary_key=True)
    sensor_name: Mapped[str] = mapped_column(VARCHAR(128), primary_key=True)
    sensor_value: Mapped[float | None] = mapped_column(Double)


class InputForecastsLong(Base):
    __tablename__ = "input_forecasts_long"
    __table_args__ = (
        ForeignKeyConstraint(["sensor_name"], ["sensor.sensor_name"], name="FK_INPUTFORECAST_SENSOR"),
        PrimaryKeyConstraint("tstamp", "sensor_name", "member", "horizon_step", name="SYS_C009003"),
    )

    sensor_name: Mapped[str] = mapped_column(VARCHAR(256), primary_key=True)
    member: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    tstamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, primary_key=True)
    horizon_step: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    value: Mapped[float | None] = mapped_column(Double)
    created: Mapped[datetime.datetime | None] = mapped_column(TIMESTAMP)

    sensor: Mapped["Sensor"] = relationship("Sensor", back_populates="input_forecasts_long")


class Metric(Base):
    __tablename__ = "metric"
    __table_args__ = (
        ForeignKeyConstraint(["model_id"], ["model.id"], name="FK_METRIC_MODEL", ondelete="CASCADE"),
        PrimaryKeyConstraint("model_id", "metric_name", "horizon_step", name="SYS_C009008"),
    )

    model_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    metric_name: Mapped[str] = mapped_column(VARCHAR(100), primary_key=True)
    horizon_step: Mapped[int | None] = mapped_column(INTEGER, primary_key=True)
    value: Mapped[float] = mapped_column(Double, nullable=False)
    tag: Mapped[str | None] = mapped_column(VARCHAR(100))

    model: Mapped["Model"] = relationship("Model", back_populates="metric")


class ModelSensor(Base):
    __tablename__ = "model_sensor"
    __table_args__ = (
        ForeignKeyConstraint(["model_id"], ["model.id"], ondelete="CASCADE", name="modell_sensor_modell"),
        ForeignKeyConstraint(["sensor_name"], ["sensor.sensor_name"], name="modell_sensor_name"),
        PrimaryKeyConstraint("model_id", "sensor_name", name="id_model_sensor"),
    )

    model_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    sensor_name: Mapped[str] = mapped_column(VARCHAR(256), primary_key=True)
    vhs_gebiet: Mapped[str | None] = mapped_column(VARCHAR(256))
    ix: Mapped[int | None] = mapped_column(INTEGER)

    model: Mapped["Model"] = relationship("Model", back_populates="model_sensor")
    sensor: Mapped["Sensor"] = relationship("Sensor", back_populates="model_sensor")


class PegelForecastsLong(Base):
    __tablename__ = "pegel_forecasts_long"
    __table_args__ = (
        ForeignKeyConstraint(["sensor_name"], ["sensor.sensor_name"], name="FK_PEGELFORECAST_SENSOR"),
        ForeignKeyConstraint(["model_id"], ["model.id"], name="FK_PEGELFORECAST_MODELL"),
        PrimaryKeyConstraint("sensor_name", "model_id", "member", "tstamp", "horizon_step", name="SYS_C009006"),
    )

    # id: Mapped[float] = mapped_column(
    #    NUMBER(38, 0, False),
    #    Identity(start=1, increment=1, minvalue=1, maxvalue=9999999999999999999999999999, cycle=False, cache=20, order=False),
    #    primary_key=True
    # )
    sensor_name: Mapped[str] = mapped_column(VARCHAR(256))
    model_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    tstamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP)
    member: Mapped[int] = mapped_column(INTEGER, nullable=False)
    horizon_step: Mapped[int] = mapped_column(INTEGER, nullable=False)
    value: Mapped[float] = mapped_column(Double)
    variance: Mapped[float | None] = mapped_column(Double)
    created: Mapped[datetime.datetime | None] = mapped_column(TIMESTAMP, server_default=func.now())

    sensor: Mapped["Sensor"] = relationship("Sensor", back_populates="pegel_forecasts_long")


class ModelBlob(Base):
    __tablename__ = "model_blob"
    __table_args__ = (
        ForeignKeyConstraint(["model_id"], ["model.id"], name="FK_MODEL_BLOB_MODEL", ondelete="CASCADE"),
        PrimaryKeyConstraint("model_id", name="SYS_C009012"),
        # UniqueConstraint('model_id', name='uq_modell_blob_model_id')
    )

    model_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    artifact: Mapped[bytes | None] = mapped_column(LargeBinary)
    yaml: Mapped[str | None] = mapped_column(Text)

    model: Mapped["Model"] = relationship("Model", back_populates="blob")


class SensorData(Base):
    __tablename__ = "sensor_data"
    __table_args__ = (
        ForeignKeyConstraint(["sensor_name"], ["sensor.sensor_name"], name="sensordata_name"),
        PrimaryKeyConstraint("tstamp", "sensor_name", name="sensordata_id"),
        Index("sensordata_name", "sensor_name"),
        Index("sensordata_ts", "tstamp"),
    )

    tstamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, primary_key=True)
    sensor_name: Mapped[str] = mapped_column(VARCHAR(256), primary_key=True)
    sensor_value: Mapped[float | None] = mapped_column(Double)

    sensor: Mapped["Sensor"] = relationship("Sensor", back_populates="sensor_data")


class EnsembleMember(Base):
    __tablename__ = "ensemble_member"
    __table_args__ = (
        ForeignKeyConstraint(["ensemble_model_id"], ["model.id"], name="FK_ENSEMBLE_MODEL", ondelete="CASCADE"),
        ForeignKeyConstraint(["submodel_model_id"], ["model.id"], name="FK_SUBMODEL_MODEL", ondelete="CASCADE"),
        PrimaryKeyConstraint("ensemble_model_id", "member_index", name="SYS_C009007"),
    )
    ensemble_model_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    submodel_model_id: Mapped[int] = mapped_column(INTEGER, nullable=False)
    member_index: Mapped[int] = mapped_column(INTEGER, nullable=False)
    weight: Mapped[float] = mapped_column(Double, nullable=True)

    ensemble_model: Mapped["Model"] = relationship(
        "Model",
        back_populates="ensemble_members",
        foreign_keys=[ensemble_model_id],
    )

    submodel_model: Mapped["Model"] = relationship(
        "Model",
        back_populates="submodel_members",
        foreign_keys=[submodel_model_id],
    )
