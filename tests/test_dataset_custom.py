import numpy as np
import pandas as pd
import pytest

from forecastlib.data_provider.data_module import CustomDataModule

SEQ_LEN, LABEL_LEN, PRED_LEN = 24, 12, 12
BATCH_SIZE = 8


@pytest.fixture(scope="module")
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


def make_datamodule(csv_dir, features, diff):
    args = {
        "data": "custom",
        "embed": "timeF",
        "num_workers": 0,
        "batch_size": BATCH_SIZE,
        "root_path": str(csv_dir),
        "data_path": "x.csv",
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "features": features,
        "target": "OT",
        "freq": "h",
        "augmentation_ratio": 0,
        "diff": diff,
    }
    return CustomDataModule(args)


@pytest.mark.parametrize(
    ("features", "expected_names", "expected_target_idx"),
    [
        ("M", ["a", "b", "OT"], [0, 1, 2]),
        ("MS", ["a", "b", "OT"], [2]),
        ("S", ["OT"], [0]),
    ],
)
def test_infer_args_without_diff(csv_dir, features, expected_names, expected_target_idx):
    dm = make_datamodule(csv_dir, features, diff=False)
    args = dm.infer_args()

    assert dm.feature_names == expected_names
    assert args.enc_in == len(expected_names)
    assert args.target_idx == expected_target_idx


@pytest.mark.parametrize(
    ("features", "expected_names", "expected_base_idx", "expected_target_idx"),
    [
        ("M", ["a", "b", "OT", "a_d1", "b_d1", "OT_d1"], [0, 1, 2], [3, 4, 5]),
        ("MS", ["a", "b", "OT", "OT_d1"], [2], [3]),
        ("S", ["OT", "OT_d1"], [0], [1]),
    ],
)
def test_infer_args_with_diff(csv_dir, features, expected_names, expected_base_idx, expected_target_idx):
    dm = make_datamodule(csv_dir, features, diff=True)
    args = dm.infer_args()

    assert dm.feature_names == expected_names
    assert args.enc_in == len(expected_names)
    assert args.base_idx == expected_base_idx
    assert args.target_idx == expected_target_idx

    raw = dm.train_set.data_x_raw
    assert np.allclose(raw["OT"].diff().iloc[1:], raw["OT_d1"].iloc[1:])


@pytest.mark.parametrize("features", ["M", "MS", "S"])
def test_diff_window_zeroes_first_step_without_mutating_data(csv_dir, features):
    ds = make_datamodule(csv_dir, features, diff=True).train_set
    d1 = slice(ds.width // 2, None) if features == "M" else slice(-1, None)
    idx = 5
    stored = ds.data_x.copy()

    for _ in range(2):  # a second access must see the same, unmodified data
        seq_x, *_ = ds[idx]
        assert np.all(seq_x[0, d1] == 0)
        assert np.allclose(seq_x[0, : d1.start], stored[idx, : d1.start])
        assert np.allclose(seq_x[1:], stored[idx + 1 : idx + SEQ_LEN])

    assert np.array_equal(ds.data_x, stored)


@pytest.mark.parametrize("diff", [False, True])
@pytest.mark.parametrize("features", ["M", "MS", "S"])
def test_dataloader_shapes(csv_dir, features, diff):
    dm = make_datamodule(csv_dir, features, diff)
    args = dm.infer_args()

    seq_x, seq_y, seq_x_mark, seq_y_mark = next(iter(dm.train_dataloader()))

    assert seq_x.shape == (BATCH_SIZE, SEQ_LEN, args.enc_in)
    assert seq_y.shape == (BATCH_SIZE, LABEL_LEN + PRED_LEN, args.enc_in)
    assert seq_x_mark.shape[:2] == (BATCH_SIZE, SEQ_LEN)
    assert seq_y_mark.shape[:2] == (BATCH_SIZE, LABEL_LEN + PRED_LEN)
