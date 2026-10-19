import pickle

import pytest
import torch
import yaml
from conftest import PRED_LEN, fit, make_model_args
from torch import nn

from forecastlib.models import DLinear
from forecastlib.models.LightningWrapper import CustomLightningModule, EnsembleModule
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


def test_target_index_buffer(data_args):
    """target_idx as a tensor for indexing on the GPU without waits; not in checkpoints, moves with the module."""
    _, args = make_model_args(data_args | {"features": "M"}, "DLinear")
    module = CustomLightningModule(args)

    assert module.target_idx_t.tolist() == list(module.target_idx) == [0, 1, 2]
    assert module.target_idx_t.dtype == torch.long
    assert "target_idx_t" not in module.state_dict()  # older checkpoints don't have it and must still load
    assert module.to("meta").target_idx_t.device.type == "meta"


def test_from_db_ignores_training_device(data_args, tmp_path):
    """A model stored with the device it was trained on (here a GPU this machine doesn't have) loads onto the
    requested device and predicts; the training device must not leak into inference."""
    dm, args = make_model_args(data_args, "DLinear")
    args.device = torch.device("cpu")
    fit(CustomLightningModule(args), dm, tmp_path)

    hparams = yaml.load((tmp_path / "hparams.yaml").read_text(), Loader=yaml.FullLoader)
    assert "scaler" in hparams
    hparams["device"] = pickle.dumps(torch.device("cuda:1"))  # as stored by a run on another machine's GPU
    yaml_clob = yaml.dump(hparams)
    blob = next((tmp_path / "checkpoints").iterdir()).read_bytes()

    model = CustomLightningModule.from_db(yaml_clob, blob)
    assert {p.device for p in model.parameters()} == {torch.device("cpu")}
    assert model.args.device == torch.device("cpu")
    model.eval()
    batch = next(iter(dm.test_dataloader()))
    with torch.no_grad():
        _x, pred, true = model.predict_step(batch, batch_idx=0)
    assert pred.shape == true.shape
    assert torch.isfinite(pred).all()


def test_from_disk_ignores_training_device(data_args, tmp_path):
    """Same as for the database: a run folder from another machine's GPU loads onto the requested device."""
    dm, args = make_model_args(data_args, "DLinear")
    args.device = torch.device("cpu")
    fit(CustomLightningModule(args), dm, tmp_path)
    hparams_path = tmp_path / "hparams.yaml"
    hparams = yaml.load(hparams_path.read_text(), Loader=yaml.FullLoader)
    hparams["device"] = pickle.dumps(torch.device("cuda:1"))
    hparams_path.write_text(yaml.dump(hparams))

    model = CustomLightningModule.from_disk(tmp_path)
    assert {p.device for p in model.parameters()} == {torch.device("cpu")}


def test_ensemble_submodels_follow_the_ensemble(data_args, tmp_path):
    """Submodels are not registered children, but .to()/.double() on the ensemble must reach them, and an ensemble
    run folder must load again."""
    diff_args = data_args | {"diff": True}
    member_dirs = []
    for i in range(2):
        dm, args = make_model_args(diff_args, "DLinear")
        args.device = torch.device("cpu")
        fit(CustomLightningModule(args), dm, tmp_path / f"member{i}")
        member_dirs.append(tmp_path / f"member{i}")

    dm, args = make_model_args(diff_args, "DLinear")
    for key, value in {
        "model_list": [CustomLightningModule.from_disk(d) for d in member_dirs],
        "norm_func": "softmax",
        "hidden_size": 8,
        "dropout": 0.0,
        "num_layers": 2,
    }.items():
        setattr(args, key, value)
    ensemble = EnsembleModule(args)

    ensemble.double()
    assert all(p.dtype == torch.float64 for m in ensemble.model_list for p in m.parameters())
    ensemble.float()
    assert all(p.dtype == torch.float32 for m in ensemble.model_list for p in m.parameters())
    # The submodels' weights stay out of the ensemble's checkpoint.
    assert all(key.startswith("weighting.") for key in ensemble.state_dict())

    fit(ensemble, dm, tmp_path / "ensemble")
    loaded = EnsembleModule.from_disk(tmp_path / "ensemble", member_dirs)
    assert {p.device for m in loaded.model_list for p in m.parameters()} == {torch.device("cpu")}
