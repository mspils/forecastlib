import copy
import pickle
from argparse import Namespace

import pytest

from forecastlib.utils.tools import ConfigTracker, setup_gpu_args


def test_one_store_for_attribute_and_item_access():
    cfg = ConfigTracker({"a": 1})
    cfg.b = 2
    cfg["c"] = 3
    assert (cfg["b"], cfg.c) == (2, 3)
    assert "b" in cfg
    assert dict(cfg) == {"a": 1, "b": 2, "c": 3}
    assert str(cfg) == str({"a": 1, "b": 2, "c": 3})


def test_tracking_rules():
    cfg = ConfigTracker({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    _ = cfg.a
    _ = getattr(cfg, "b", None)
    assert hasattr(cfg, "c")
    assert cfg.get("d") == 4
    assert cfg.get("missing", 7) == 7
    # untracked reads
    _ = cfg["e"]
    _ = "e" in cfg
    _ = list(cfg), dict(cfg), cfg.to_dict(), str(cfg), repr(cfg)
    cfg.f = 6  # writing doesn't count as reading
    assert list(cfg.accessed_dict()) == ["a", "b", "c", "d"]
    assert cfg.accessed_attrs == {"a", "b", "c", "d"}


def test_written_then_read_settings_are_tracked():
    """Values a module computes and then uses belong into its hparams."""
    cfg = ConfigTracker({})
    cfg.enc_in = 3
    assert cfg.enc_in == 3
    assert cfg.accessed_dict() == {"enc_in": 3}


def test_missing_settings_raise_like_object_and_dict():
    cfg = ConfigTracker({"a": 1})
    with pytest.raises(AttributeError):
        _ = cfg.missing
    with pytest.raises(KeyError):
        _ = cfg["missing"]
    assert getattr(cfg, "missing", None) is None
    assert cfg.accessed_dict() == {}


def test_wrapping_copies_and_does_not_leak():
    """A module wrapping another module's tracker reads its values but neither its access record nor its writes
    reach the original."""
    outer = ConfigTracker({"seq_len": 24, "batch_size": 8})
    _ = outer.batch_size  # outer's own read
    outer.enc_in = 3  # e.g. written by CustomDataModule.infer_args
    inner = ConfigTracker(outer)
    assert inner.enc_in == 3
    assert inner.seq_len == 24
    inner.model_count = 2
    assert outer.accessed_dict() == {"batch_size": 8}  # inner's reads don't show up here
    assert "model_count" not in outer  # nor its writes
    assert inner.accessed_dict() == {"enc_in": 3, "seq_len": 24}  # outer's reads don't show up there


def test_sources_and_overrides():
    assert ConfigTracker(Namespace(a=1)).to_dict() == {"a": 1}
    assert ConfigTracker({"a": 1}, a=2, b=3).to_dict() == {"a": 2, "b": 3}
    assert ConfigTracker().to_dict() == {}
    source = {"a": 1}
    cfg = ConfigTracker(source)
    cfg.a = 5
    assert source == {"a": 1}  # the source dict is not modified
    with pytest.raises(TypeError):
        ConfigTracker(42)


def test_pickle_and_copy_keep_values_and_access_record():
    cfg = ConfigTracker({"a": 1, "nested": {"x": [1, 2]}})
    _ = cfg.a
    for clone in (pickle.loads(pickle.dumps(cfg)), copy.deepcopy(cfg), copy.copy(cfg)):
        assert clone.to_dict() == cfg.to_dict()
        assert clone.accessed_dict() == {"a": 1}
    deep = copy.deepcopy(cfg)
    deep["nested"]["x"].append(3)
    assert cfg["nested"]["x"] == [1, 2]


def test_setup_gpu_args_on_tracker():
    cfg = ConfigTracker({})
    setup_gpu_args(cfg)
    assert {"devices", "accelerator", "device", "use_amp"} <= set(cfg)


def test_equality_and_underscore_settings():
    assert ConfigTracker({"a": 1}) == {"a": 1}
    cfg = ConfigTracker({"_cache": {}})  # datasets keep caches on the config
    assert cfg._cache == {}
    with pytest.raises(AttributeError):
        _ = cfg.__missing_dunder__
