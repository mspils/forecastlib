import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from forecastlib.data_provider.data_module import CustomDataModule

SEQ_LEN, LABEL_LEN, PRED_LEN = 24, 12, 12
BATCH_SIZE = 8
N_ROWS = 500


@pytest.fixture(scope="session")
def csv_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    n = 500
    pd.DataFrame(
        {
            "date": pd.date_range("2020", periods=n, freq="h"),
            "a": rng.random(n).cumsum(),
            "OT": rng.random(n).cumsum(),
            "b": rng.random(n).cumsum(),
        }
    ).to_csv(d / "x.csv", index=False)
    return d


@pytest.fixture
def data_args(csv_dir):
    """Minimal CustomDataModule args for the CSV in csv_dir, target OT."""
    return {
        "data": "custom",
        "embed": "timeF",
        "num_workers": 0,
        "batch_size": BATCH_SIZE,
        "root_path": str(csv_dir),
        "data_path": "x.csv",
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "features": "MS",
        "target": "OT",
        "freq": "h",
        "augmentation_ratio": 0,
        "diff": False,
    }


@pytest.fixture(scope="session")
def covariate_df():
    """Data with covariates besides the target: precipitation and temp, in that file order."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020", periods=N_ROWS, freq="h"),
            "a": rng.random(N_ROWS).cumsum(),
            "precipitation": rng.gamma(0.5, 2.0, N_ROWS),
            "OT": rng.random(N_ROWS).cumsum(),
            "temp": 10 + 5 * rng.standard_normal(N_ROWS),
            "b": rng.random(N_ROWS).cumsum(),
        }
    )


@pytest.fixture
def covariate_args(data_args, covariate_df, tmp_path):
    covariate_df.to_csv(tmp_path / "covariates.csv", index=False)
    return data_args | {"root_path": str(tmp_path), "data_path": "covariates.csv"}


MODEL_ARGS = {
    "model_id": "test",
    "task_name": "long_term_forecast",
    "moving_avg": 5,
    "loss": "MSE",
    "lradj": "type1",
    "learning_rate": 1e-3,
    "train_epochs": 1,
    "patience": 1,
}


def make_model_args(data_args, model):
    dm = CustomDataModule(data_args)
    args = dm.infer_args()
    for k, v in (MODEL_ARGS | {"model": model}).items():
        setattr(args, k, v)
    return dm, args


def fit(module, dm, log_dir, callbacks=()):
    """Train briefly on CPU, writing hparams.yaml and checkpoints/ to log_dir like the training scripts do."""
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        logger=CSVLogger(log_dir, name="", version=""),
        callbacks=[ModelCheckpoint(dirpath=log_dir / "checkpoints"), *callbacks],
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    trainer.fit(module, datamodule=dm)
