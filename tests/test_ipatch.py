from types import SimpleNamespace

import pytest
import torch
from conftest import PRED_LEN, SEQ_LEN, fit, make_model_args

from forecastlib.models import iPatch
from forecastlib.models.LightningWrapper import CustomLightningModule

IPATCH_ARGS = {
    "main_cycle": 12,
    "d_model": 16,
    "n_heads": 2,
    "e_layers": 1,
    "d_ff": 32,
    "dropout": 0.1,
    "factor": 1,
    "activation": "gelu",
}


def configs(**overrides):
    base = {
        "seq_len": 96,
        "pred_len": 12,
        "task_name": "long_term_forecast",
        "embed": "timeF",
        "freq": "h",
        **IPATCH_ARGS,
        "main_cycle": 24,
    }
    return SimpleNamespace(**(base | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"model_trend": True},  # even cycle: the trend's moving average must keep the length
        {"x_mark_size": 0},  # no time marks as tokens
        {"x_mark_size": 2},  # only the first mark columns
        {"use_norm": False},
    ],
)
def test_forward_shapes(overrides):
    model = iPatch.Model(configs(**overrides)).eval()
    seq_len = configs(**overrides).seq_len
    for n_vars in (1, 3):
        out = model(torch.randn(2, seq_len, n_vars), torch.randn(2, seq_len, 4), None, None)
        assert out.shape == (2, 12, n_vars)


def test_needs_main_cycle():
    with pytest.raises(ValueError, match="main_cycle"):
        iPatch.Model(configs(main_cycle=None))


def test_seq_len_must_be_whole_cycles():
    with pytest.raises(ValueError, match=r"seq_len=100 and main_cycle=24; e.g. seq_len=96 or 120"):
        iPatch.Model(configs(seq_len=100))


def test_too_few_mark_columns():
    model = iPatch.Model(configs(x_mark_size=6))
    with pytest.raises(ValueError, match="x_mark_size"):
        model(torch.randn(2, 96, 3), torch.randn(2, 96, 4), None, None)


@pytest.mark.parametrize(("features", "model_trend"), [("MS", False), ("MS", True), ("M", False), ("S", True)])
def test_trains_and_reloads(data_args, tmp_path, features, model_trend):
    assert SEQ_LEN % IPATCH_ARGS["main_cycle"] == 0
    dm, args = make_model_args(data_args | {"features": features}, "iPatch")
    for k, v in (IPATCH_ARGS | {"model_trend": model_trend}).items():
        setattr(args, k, v)
    module = CustomLightningModule(args)
    fit(module, dm, tmp_path)

    loaded = CustomLightningModule.from_disk(tmp_path, device="cpu")
    assert isinstance(loaded.model, iPatch.Model)
    assert loaded.model.model_trend == model_trend
    module.eval()
    loaded.eval()
    batch = [b.float() for b in next(iter(dm.val_dataloader()))]
    with torch.no_grad():
        out = loaded(*batch)
        assert torch.allclose(module(*batch), out)
    assert out.shape[1:] == (PRED_LEN, args.enc_in)
