import numpy as np
import pandas as pd
import pytest

SEQ_LEN, LABEL_LEN, PRED_LEN = 24, 12, 12
BATCH_SIZE = 8


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
