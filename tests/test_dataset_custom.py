import numpy as np
import pandas as pd
import pytest
from conftest import BATCH_SIZE, LABEL_LEN, PRED_LEN, SEQ_LEN

from forecastlib.data_provider.data_module import CustomDataModule


def make_datamodule(data_args, features, diff):
    return CustomDataModule(data_args | {"features": features, "diff": diff})


@pytest.mark.parametrize(
    ("features", "expected_names", "expected_target_idx"),
    [
        ("M", ["a", "b", "OT"], [0, 1, 2]),
        ("MS", ["a", "b", "OT"], [2]),
        ("S", ["OT"], [0]),
    ],
)
def test_infer_args_without_diff(data_args, features, expected_names, expected_target_idx):
    dm = make_datamodule(data_args, features, diff=False)
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
def test_infer_args_with_diff(data_args, features, expected_names, expected_base_idx, expected_target_idx):
    dm = make_datamodule(data_args, features, diff=True)
    args = dm.infer_args()

    assert dm.feature_names == expected_names
    assert args.enc_in == len(expected_names)
    assert args.base_idx == expected_base_idx
    assert args.target_idx == expected_target_idx

    raw = dm.train_set.data_x_raw
    assert np.allclose(raw["OT"].diff().iloc[1:], raw["OT_d1"].iloc[1:])


@pytest.mark.parametrize("features", ["M", "MS", "S"])
def test_diff_window_zeroes_first_step_without_mutating_data(data_args, features):
    ds = make_datamodule(data_args, features, diff=True).train_set
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
def test_dataloader_shapes(data_args, features, diff):
    dm = make_datamodule(data_args, features, diff)
    args = dm.infer_args()

    seq_x, seq_y, seq_x_mark, seq_y_mark = next(iter(dm.train_dataloader()))

    assert seq_x.shape == (BATCH_SIZE, SEQ_LEN, args.enc_in)
    assert seq_y.shape == (BATCH_SIZE, LABEL_LEN + PRED_LEN, args.enc_in)
    assert seq_x_mark.shape[:2] == (BATCH_SIZE, SEQ_LEN)
    assert seq_y_mark.shape[:2] == (BATCH_SIZE, LABEL_LEN + PRED_LEN)


def test_date_col_is_renamed(data_args, csv_dir, tmp_path):
    pd.read_csv(csv_dir / "x.csv").rename(columns={"date": "tstamp"}).to_csv(tmp_path / "x.csv", index=False)
    args = data_args | {"root_path": str(tmp_path), "date_col": "tstamp"}

    dm = CustomDataModule(args)
    reference = make_datamodule(data_args, "MS", diff=False)

    assert dm.feature_names == ["a", "b", "OT"]
    assert np.array_equal(dm.train_set.data_stamp, reference.train_set.data_stamp)
    assert dm.hparams["date_col"] == "tstamp"  # needed to rebuild the datamodule in from_disk
