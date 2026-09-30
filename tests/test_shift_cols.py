import numpy as np
import pytest
from conftest import N_ROWS, PRED_LEN, SEQ_LEN, fit, make_model_args

from forecastlib.data_provider.data_module import CustomDataModule
from forecastlib.models.LightningWrapper import CustomLightningModule

SHIFT = 6


@pytest.fixture
def shift_args(covariate_args):
    return covariate_args | {"shift_cols": {"precipitation": SHIFT}}


def test_last_rows_of_seq_x_hold_the_future(shift_args, covariate_df):
    ds = CustomDataModule(shift_args).train_set  # train split starts at row 0, so dataset index == file row
    raw = ds.data_x_raw.reset_index(drop=True)
    precipitation = covariate_df["precipitation"].to_numpy()
    col = list(raw).index("precipitation")

    assert np.allclose(raw["precipitation"], precipitation[SHIFT : SHIFT + len(raw)])
    assert np.allclose(raw["temp"], covariate_df["temp"].iloc[: len(raw)])  # other columns untouched

    for idx in (0, 7, len(ds) - 1):
        seq_x, *_ = ds[idx]
        s_end = idx + SEQ_LEN
        # The last SHIFT rows of the window are the SHIFT steps after it, i.e. the start of the forecast period.
        future = ds.scaler.transform(raw.iloc[s_end - SHIFT : s_end])[:, col]
        assert np.allclose(seq_x[-SHIFT:, col], future)
        assert np.allclose(raw["precipitation"].iloc[s_end - SHIFT : s_end], precipitation[s_end : s_end + SHIFT])


def test_rows_without_future_are_dropped(shift_args, covariate_args):
    # With the defaults the test split ends at the last row, so its length shows the dropped rows.
    shifted = CustomDataModule(shift_args)
    plain = CustomDataModule(covariate_args)

    assert shifted.test_set.data_x_raw.index[-1] == N_ROWS - 1 - SHIFT
    assert not shifted.test_set.data_x_raw.isna().any().any()
    assert len(plain.train_set) + len(plain.val_set) + len(plain.test_set) > len(shifted.train_set) + len(
        shifted.val_set
    ) + len(shifted.test_set)


def test_shifted_column_stays_a_data_column(shift_args, covariate_args):
    shifted = CustomDataModule(shift_args)
    plain = CustomDataModule(covariate_args)

    assert shifted.feature_names == plain.feature_names
    assert shifted.infer_args().enc_in == plain.infer_args().enc_in
    assert shifted.hparams["shift_cols"] == {"precipitation": SHIFT}


def test_combines_with_known_cols(shift_args):
    dm = CustomDataModule(shift_args | {"known_cols": ["temp"]})

    assert dm.feature_names == ["a", "precipitation", "b", "OT"]
    assert dm.train_set.data_stamp.shape[1] == 4 + 1


@pytest.mark.parametrize(
    ("shift_cols", "error", "match"),
    [
        ({"OT": 1}, ValueError, "target"),
        ({"precipitation": 0}, ValueError, "must be an int"),
        ({"precipitation": PRED_LEN + 1}, ValueError, "must be an int"),
        ({"precipitation": 1.5}, ValueError, "must be an int"),
        ({"rain": 1}, KeyError, "rain"),
    ],
)
def test_invalid_shift_cols_raise(covariate_args, shift_cols, error, match):
    with pytest.raises(error, match=match):
        CustomDataModule(covariate_args | {"shift_cols": shift_cols})


def test_shift_col_in_known_cols_raises(shift_args):
    with pytest.raises(ValueError, match="known_cols"):
        CustomDataModule(shift_args | {"known_cols": ["precipitation"]})


def test_model_without_known_covariate_support_trains(shift_args, tmp_path):
    """The point of shift_cols: any model can use the future values, here DLinear."""
    dm, args = make_model_args(shift_args, "DLinear")
    fit(CustomLightningModule(args), dm, tmp_path)
    CustomLightningModule.from_disk(tmp_path, device="cpu")
