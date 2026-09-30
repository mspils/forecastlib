import numpy as np
import pytest
import torch
from conftest import LABEL_LEN, N_ROWS, PRED_LEN, SEQ_LEN, fit, make_model_args

from forecastlib.data_provider.data_module import CustomDataModule
from forecastlib.models.LightningWrapper import CustomLightningModule
from forecastlib.utils.tools import load_model_settings

TIME_LEN = 4  # time features for embed=timeF, freq=h
# Given in the opposite order to the file, the marks must follow known_cols.
KNOWN_COLS = ["temp", "precipitation"]
TFT_ARGS = {"d_model": 16, "n_heads": 2, "dropout": 0.1}


@pytest.fixture
def known_args(covariate_args):
    return covariate_args | {"known_cols": KNOWN_COLS}


def scaled_known(df):
    """The known columns scaled with train-split statistics, computed independently of the dataset."""
    train = df[KNOWN_COLS].iloc[: int(N_ROWS * 0.7)]
    return ((df[KNOWN_COLS] - train.mean()) / train.std(ddof=0)).to_numpy()


def test_future_known_values_are_in_seq_y_mark(known_args, covariate_df):
    ds = CustomDataModule(known_args).train_set  # train split starts at row 0, so dataset index == file row
    expected = scaled_known(covariate_df)

    for idx in (0, 7, len(ds) - 1):
        seq_x, _, seq_x_mark, seq_y_mark = ds[idx]
        future = slice(idx + SEQ_LEN, idx + SEQ_LEN + PRED_LEN)

        assert np.allclose(seq_y_mark[-PRED_LEN:, TIME_LEN:], expected[future])
        assert np.allclose(seq_x_mark[:, TIME_LEN:], expected[idx : idx + SEQ_LEN])
        assert seq_x.shape[-1] == 3  # a, b, OT only

    assert "precipitation" not in list(ds.data_x_raw)
    assert "temp" not in list(ds.data_x_raw)


@pytest.mark.parametrize("diff", [False, True])
@pytest.mark.parametrize(
    ("features", "names", "diff_names"),
    [
        ("M", ["a", "b", "OT"], ["a_d1", "b_d1", "OT_d1"]),
        ("MS", ["a", "b", "OT"], ["OT_d1"]),
        ("S", ["OT"], ["OT_d1"]),
    ],
)
def test_shapes_and_inferred_args(known_args, features, names, diff_names, diff):
    dm = CustomDataModule(known_args | {"features": features, "diff": diff})
    args = dm.infer_args()
    expected_names = names + diff_names if diff else names
    n_data = len(expected_names)

    assert dm.feature_names == expected_names  # known columns are never differenced
    assert args.enc_in == args.dec_in == args.c_out == n_data
    assert dm.scaler.mean_.shape == (n_data,)
    assert dm.known_scaler.mean_.shape == (len(KNOWN_COLS),)
    assert args.known_len_extra == len(KNOWN_COLS)
    assert args.observed_pos == list(range(n_data))
    assert args.static_pos == []

    seq_x, seq_y, seq_x_mark, seq_y_mark = next(iter(dm.train_dataloader()))
    assert seq_x.shape[-1] == seq_y.shape[-1] == n_data
    assert seq_x_mark.shape[-1] == seq_y_mark.shape[-1] == TIME_LEN + len(KNOWN_COLS)
    assert seq_y_mark.shape[1] == LABEL_LEN + PRED_LEN


def test_target_in_known_cols_raises(known_args):
    with pytest.raises(ValueError, match="target"):
        CustomDataModule(known_args | {"known_cols": ["OT"]})


def test_missing_known_col_raises(known_args):
    with pytest.raises(KeyError, match="rain"):
        CustomDataModule(known_args | {"known_cols": ["rain"]})


def test_empty_known_cols_changes_nothing(known_args):
    without = {k: v for k, v in known_args.items() if k != "known_cols"}
    dm_default = CustomDataModule(without)
    dm_empty = CustomDataModule(known_args | {"known_cols": []})
    args_default, args_empty = dm_default.infer_args(), dm_empty.infer_args()

    for split in ("train_set", "val_set", "test_set"):
        ds_default, ds_empty = getattr(dm_default, split), getattr(dm_empty, split)
        assert np.array_equal(ds_default.data_x, ds_empty.data_x)
        assert np.array_equal(ds_default.data_stamp, ds_empty.data_stamp)
    # Without known_cols the covariate columns stay ordinary data columns, as before.
    assert dm_empty.feature_names == ["a", "precipitation", "temp", "b", "OT"]
    assert dm_empty.known_scaler is None
    for key in ("known_scaler", "known_len_extra"):
        assert key not in dm_empty.hparams
        assert key not in dm_default.hparams
    assert not hasattr(args_empty, "known_len_extra")
    assert args_empty.enc_in == args_default.enc_in


def test_tft_trains_and_reloads(known_args, tmp_path):
    dm, args = make_model_args(known_args, "TemporalFusionTransformer")
    for k, v in TFT_ARGS.items():
        setattr(args, k, v)
    module = CustomLightningModule(args)
    assert len(module.model.embedding.known_embedding.extra_embedding) == len(KNOWN_COLS)

    log_dir = tmp_path / "run"
    fit(module, dm, log_dir)

    loaded = CustomLightningModule.from_disk(log_dir, device="cpu")
    for p_trained, p_loaded in zip(module.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(p_trained.cpu(), p_loaded.cpu())

    # Inference code needs the known scaler to scale forecast values.
    settings = load_model_settings(log_dir)
    assert np.allclose(settings.known_scaler.mean_, dm.known_scaler.mean_)
    assert settings.known_cols == KNOWN_COLS

    batch = next(iter(dm.val_dataloader()))
    module.eval()
    loaded.eval()
    with torch.no_grad():
        out_trained = module(*(b.float() for b in batch))
        out_loaded = loaded(*(b.float() for b in batch))
    assert torch.allclose(out_trained, out_loaded)


def test_model_without_known_covariate_support_raises(known_args):
    _, args = make_model_args(known_args, "DLinear")
    with pytest.raises(ValueError, match="DLinear doesn't support known covariates"):
        CustomLightningModule(args)
