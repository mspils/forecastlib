import pytest
import torch
from conftest import PRED_LEN, fit, make_model_args
from torch import nn

from forecastlib.models import DLinear
from forecastlib.models.LightningWrapper import CustomLightningModule
from forecastlib.utils.callbacks import StepWiseMetricsCallback, StepWiseMetricsCallbackWaterlevel


class TinyLinear(nn.Module):
    """A model that isn't in model_dict: one linear layer from seq_len to pred_len per channel."""

    def __init__(self, configs):
        super().__init__()
        self.proj = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):  # noqa: ARG002 - the interface all models share
        return self.proj(x_enc.transpose(1, 2)).transpose(1, 2)


@pytest.mark.parametrize(
    ("model", "expected_class", "expected_name"),
    [
        ("DLinear", DLinear.Model, "DLinear"),
        (DLinear.Model, DLinear.Model, "DLinear"),  # a registered class is stored under its registry name
        (TinyLinear, TinyLinear, "TinyLinear"),
    ],
)
def test_model_from_name_or_class(data_args, model, expected_class, expected_name):
    _, args = make_model_args(data_args, model)
    module = CustomLightningModule(args)

    assert isinstance(module.model, expected_class)
    assert module.hparams["model"] == expected_name
    assert module(*module.example_input_array).shape[1] == PRED_LEN


def test_unknown_model_name_raises(data_args):
    _, args = make_model_args(data_args, "NoSuchModel")
    with pytest.raises(KeyError, match="NoSuchModel"):
        CustomLightningModule(args)


def test_custom_class_round_trip_from_disk(data_args, tmp_path):
    dm, args = make_model_args(data_args, TinyLinear)
    module = CustomLightningModule(args)

    fit(module, dm, tmp_path)

    # The name alone can't be resolved, the class has to be passed back in.
    with pytest.raises(KeyError, match="TinyLinear"):
        CustomLightningModule.from_disk(tmp_path, device="cpu")

    loaded = CustomLightningModule.from_disk(tmp_path, device="cpu", model_class=TinyLinear)
    assert isinstance(loaded.model, TinyLinear)
    for p_trained, p_loaded in zip(module.model.parameters(), loaded.model.parameters(), strict=True):
        assert torch.equal(p_trained.cpu(), p_loaded.cpu())


@pytest.mark.parametrize(
    ("callback_class", "features"),
    [(StepWiseMetricsCallbackWaterlevel, "MS"), (StepWiseMetricsCallback, "M")],  # the latter only supports M
)
@pytest.mark.parametrize("model", ["DLinear", TinyLinear])
def test_metric_callbacks_reload_model(data_args, tmp_path, callback_class, features, model):
    """The callbacks reload the checkpoint in on_fit_end, which must also work for a custom model class."""
    dm, args = make_model_args(data_args | {"features": features}, model)
    fit(CustomLightningModule(args), dm, tmp_path, callbacks=[callback_class()])


def test_filter_without_samples_gives_nan(data_args, tmp_path):
    """A custom filter that selects nothing must give NaN metrics, not crash or log garbage."""
    filters = {
        "": lambda x, true, pred: torch.ones(x.shape[0], dtype=torch.bool),  # noqa: ARG005
        "_none": lambda x, true, pred: torch.zeros(x.shape[0], dtype=torch.bool),  # noqa: ARG005
        "_some": lambda x, true, pred: torch.arange(x.shape[0]) % 2 == 0,  # noqa: ARG005
    }
    callback = StepWiseMetricsCallbackWaterlevel(filter_dict=filters)
    dm, args = make_model_args(data_args, "DLinear")
    fit(CustomLightningModule(args), dm, tmp_path, callbacks=[callback])

    for split in ("train", "val", "test"):
        for metric in ("mse", "kge", "conf50"):  # conf50 uses torch.quantile, which fails on empty input
            assert torch.isnan(callback.metric_dict[f"{split}_{metric}_none"]).all()
            assert callback.metric_dict[f"{split}_{metric}_none"].shape == (PRED_LEN,)
            assert not torch.isnan(callback.metric_dict[f"{split}_{metric}_some"]).any()
            assert not torch.isnan(callback.metric_dict[f"{split}_{metric}"]).any()
