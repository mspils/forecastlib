import pickle

import numpy as np
import pytest
import yaml
from conftest import fit, make_model_args

from forecastlib.models.LightningWrapper import CustomLightningModule
from forecastlib.utils.tools import load_model_settings, load_yaml


def test_load_yaml_reads_what_lightning_writes():
    data = {
        "pair": (1, 2),
        "scaler": pickle.dumps({"mean": 1.0}),
        "gauge": np.str_("DE110000"),
        "ratio": np.float64(0.5),
        "dtype": np.dtype("<f8"),
        "plain": [1, "a", None],
    }
    loaded = load_yaml(yaml.dump(data))  # yaml.dump writes python/... tags like Lightning does
    assert loaded["pair"] == (1, 2)
    assert pickle.loads(loaded["scaler"]) == {"mean": 1.0}
    assert loaded["gauge"] == "DE110000"
    assert type(loaded["gauge"]) is str
    assert loaded["ratio"] == 0.5
    assert loaded["dtype"] == np.dtype("<f8")
    assert loaded["plain"] == [1, "a", None]


@pytest.mark.parametrize(
    "document",
    [
        "x: !!python/object/apply:os.system ['echo hacked']",
        "x: !!python/name:os.system",
        "x: !!python/object/new:collections.OrderedDict [[[a, 1]]]",
        "x: !!python/object/apply:numpy.dtype {args: [O8]}",  # object dtype could hold Python objects
    ],
)
def test_load_yaml_refuses_python_objects(document):
    with pytest.raises(yaml.constructor.ConstructorError):
        load_yaml(document)


def test_trained_run_loads_with_the_safe_loader(data_args, tmp_path):
    dm, args = make_model_args(data_args, "DLinear")
    fit(CustomLightningModule(args), dm, tmp_path)
    settings = load_model_settings(tmp_path)
    assert settings["seq_len"] == args.seq_len
    assert hasattr(settings["scaler"], "transform")  # unpickled from !!binary by load_model_settings
    CustomLightningModule.from_disk(tmp_path)
