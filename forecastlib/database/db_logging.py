import logging
import sys

import pandas as pd
from sqlalchemy.orm import Session

from forecastlib.database.orm_classes import Log


class DatabaseHandler(logging.Handler):
    """adapted from https://gist.github.com/ykessler/2662203."""

    def __init__(self, engine) -> None:
        logging.Handler.__init__(self)
        self.engine = engine
        self.session = Session(bind=self.engine)

        # Create table if needed:
        # Log.__table__.create(self.engine, checkfirst=True)

    def emit(self, record) -> None:
        if record.exc_info:
            record.exc_text = logging._defaultFormatter.formatException(record.exc_info)
        else:
            record.exc_text = ""

        log = Log(
            created=pd.to_datetime(record.created, unit="s"),
            loglevelname=record.levelname,
            message=record.message,
            module=record.module,
            funcname=record.funcName,
            lineno=record.lineno,
            exception=record.exc_text,
            gauge=record.gauge_id,
        )
        self.session.add(log)
        self.session.commit()


def prepare_logging_1() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Executing %s with parameters %s ", sys.argv[0], sys.argv[1:])


def prepare_logging_2(engine) -> None:

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.gauge_id = kwargs.pop("gauge_id", None)
        return record

    logging.setLogRecordFactory(record_factory)

    log_formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    db_handler = DatabaseHandler(engine)
    db_handler.setFormatter(log_formatter)

    db_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(db_handler)
    logging.info("Executing %s with parameters %s ", sys.argv[0], sys.argv[1:])
