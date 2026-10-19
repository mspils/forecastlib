"""All kind of handy little tools, mostly for handling configs."""

import contextlib
import inspect
import pickle
from argparse import Namespace
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml


class ConfigTracker(Mapping):
    """A configuration that remembers which settings were read.

    Modules wrap their configuration in one and save the settings they actually read as hyperparameters, so a run
    folder or database entry holds exactly what is needed to rebuild the model.

    Rules:
        * One store: ``cfg.name = v`` and ``cfg["name"] = v`` write to the same dict that ``cfg.name``, ``cfg["name"]``,
          ``"name" in cfg``, iteration, ``dict(cfg)`` and ``str(cfg)`` read from.
        * Tracked reads: ``cfg.name``, ``getattr(cfg, "name", default)``, ``hasattr`` and ``cfg.get("name", default)``
          mark a setting as accessed when it exists.
        * Untracked reads: ``cfg["name"]``, ``in``, iteration, ``to_dict()`` and printing.
          Writing never marks a setting.
        * Wrapping copies: ``ConfigTracker(other)`` starts from a shallow copy of ``other``'s settings with an empty
          access record, so reads by one module never show up in another module's tracker and writes don't leak back.
        * Missing settings raise ``AttributeError`` for attribute access and ``KeyError`` for ``[]``, like an object
          and a dict do.

    Settings named like the methods (``get``, ``keys``, ``items``, ``values``, ``to_dict``, ``accessed_dict``,
    ``accessed_attrs``) can only be read with ``[]``.
    """

    __slots__ = ("_accessed", "_data")

    def __init__(self, config: "Mapping | Namespace | ConfigTracker | None" = None, /, **overrides: Any) -> None:
        """Create a tracker from a dict-like object, an ``argparse.Namespace`` or another tracker (copied)."""
        if config is None:
            data = {}
        elif isinstance(config, ConfigTracker):
            data = dict(config._data)  # noqa: SLF001
        elif isinstance(config, Mapping):
            data = dict(config)
        elif hasattr(config, "__dict__"):
            data = dict(vars(config))
        else:
            msg = f"Cannot build a ConfigTracker from {type(config).__name__}"
            raise TypeError(msg)
        data.update(overrides)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_accessed", {})  # dict as an ordered set

    # --- attribute access (tracked) -------------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Read a setting and mark it as accessed. Only called when normal attribute lookup fails."""
        if name in ConfigTracker.__slots__ or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)  # not a setting; also keeps copy/pickle away from the settings
        try:
            value = self._data[name]
        except KeyError:
            msg = f"No setting {name!r}"
            raise AttributeError(msg) from None
        self._accessed[name] = None
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Write a setting."""
        if name in ConfigTracker.__slots__:
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name: str) -> None:
        """Remove a setting."""
        try:
            del self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name: str, default: Any = None) -> Any:
        """Read a setting like ``getattr(cfg, name, default)``: tracked if it exists."""
        if name in self._data:
            self._accessed[name] = None
            return self._data[name]
        return default

    # --- dict access (untracked) ----------------------------------------------------------------------------------

    def __getitem__(self, name: str) -> Any:
        """Read a setting without marking it as accessed."""
        return self._data[name]

    def __setitem__(self, name: str, value: Any) -> None:
        """Write a setting."""
        self._data[name] = value

    def __delitem__(self, name: str) -> None:
        """Remove a setting."""
        del self._data[name]

    def __iter__(self) -> Any:
        """Iterate over setting names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of settings."""
        return len(self._data)

    def __contains__(self, name: object) -> bool:
        """Whether a setting exists (does not mark it as accessed)."""
        return name in self._data

    # --- tracking -------------------------------------------------------------------------------------------------

    @property
    def accessed_attrs(self) -> set[str]:
        """Names of the settings read through attribute access or ``get``."""
        return set(self._accessed)

    def accessed_dict(self) -> dict[str, Any]:
        """The accessed settings and their current values, in the order they were first read."""
        return {k: self._data[k] for k in self._accessed if k in self._data}

    def to_dict(self) -> dict[str, Any]:
        """A shallow copy of all settings."""
        return dict(self._data)

    # --- misc -----------------------------------------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Pickle and copy support."""
        return {"data": self._data, "accessed": list(self._accessed)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Pickle and copy support."""
        object.__setattr__(self, "_data", state["data"])
        object.__setattr__(self, "_accessed", dict.fromkeys(state["accessed"]))

    def __dir__(self) -> list[str]:
        """Include setting names for autocompletion."""
        return sorted({*super().__dir__(), *(k for k in self._data if isinstance(k, str))})

    def __str__(self) -> str:
        """The settings as a dict."""
        return str(self._data)

    def __repr__(self) -> str:
        """``ConfigTracker({...})``."""
        return f"ConfigTracker({self._data!r})"


def parse_yaml_combinations(yaml_path: Path) -> list[dict]:
    """Parse a YAML file and generate all parameter combinations.

    Expected YAML structure (preferred):
    default:
      <default keys/values>
    combinations:
      - model: [val1, val2]
        data_settings:
          - data: ETTh2
            root_path: dataset/ETT-small
            data_path: ETTh2.csv
          - data: ETTh1
            root_path: dataset/ETT-small
            data_path: ETTh1.csv

    Grouped parameters are lists of dicts. Each dict in the list represents
    a group, and its key/value pairs are unpacked into the result config.
    Simple parameters are lists of scalar values.

    Returns:
        A list of configuration dictionaries representing all parameter combinations.

    """
    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # Determine defaults and combination blocks
    defaults = data.get("defaults")

    defaults = {}
    combos_source = data.get("combinations")
    if not isinstance(combos_source, list):
        combos_source = [combos_source]

    combinations = []
    # Ensure combos_source is iterable list of dicts
    for block in combos_source:
        param_values = []
        param_names = []
        param_is_group = {}  # Track which params are dict-based groups

        for key, value in block.items():
            param_names.append(key)

            # Guard: if a single non-list value provided, wrap it
            values = value if isinstance(value, list) else [value]

            # Check if this is a dict-based group (list of dicts)
            if values and isinstance(values[0], dict):
                # Dict-based group: treat each dict as a single option
                param_values.append(values)
                param_is_group[key] = True
            else:
                # Simple parameter: wrap each scalar value
                param_values.append([[v] for v in values])
                param_is_group[key] = False

        # Generate Cartesian product
        for combo in product(*param_values):
            # Start from defaults then override with combo-specific values
            result = dict(defaults) if isinstance(defaults, dict) else {}
            for i, name in enumerate(param_names):
                item = combo[i]

                if param_is_group[name]:
                    # This is a dict-based group: unpack it
                    result.update(item)
                else:
                    # This is a simple scalar value
                    result[name] = item[0]
            combinations.append(result)

    return combinations


def group_configs_by_data_and_model(configs: list[dict]) -> dict[str, dict]:
    """Group configurations by (data_path, model) to create separate studies.

    Args:
        configs: list of configuration dicts

    Returns:
        dict mapping (data_path, model) -> list of configs for that combination

    """
    groups = {}
    for config in configs:
        data_path = config.get("data_path", "unknown")
        model = config.get("model", "unknown")
        key = (data_path, model)

        if key not in groups:
            groups[key] = []
        groups[key].append(config)

    return groups


def setup_gpu_args(args: Namespace, prefered_gpu: int | None = None) -> None:
    """Configure GPU arguments for PyTorch and Lightning training.

    Sets up CUDA device configuration, accelerator type, and mixed precision settings.
    If CUDA is available, configures the specified GPU device with high precision matmul.
    Falls back to CPU if CUDA is unavailable.

    Args:
        args: Namespace object or dictionary containing training arguments to be updated.
        prefered_gpu: Preferred GPU device index. If None, defaults to device 3.
                     Ignored if CUDA is not available.

    Returns:
        None. Updates args in-place with 'devices', 'accelerator', 'device', and 'use_amp' keys.

    """
    if torch.cuda.is_available():  # and args.use_gpu:
        torch.set_float32_matmul_precision("high")
        devices = [3] if prefered_gpu is None else [prefered_gpu]  # TODO 'auto'
        accelerator = "cuda"
        device = torch.device(f"cuda:{devices[0]}")  # Necessary for WPMixer
    else:
        devices = "auto"
        accelerator = "auto"
        device = torch.device("cpu")  # Necessary for WPMixer, will probably break stuff in this case

    if isinstance(args, dict):
        args["devices"] = devices
        args["accelerator"] = accelerator
        args["device"] = device
        args["use_amp"] = False  # TODO do we want to and can we use automatic mixed precision? (WPMixer)

    else:
        args.devices = devices
        args.accelerator = accelerator
        args.device = device
        args.use_amp = False  # TODO do we want to and can we use automatic mixed precision? (WPMixer)


def load_model_settings(model_dir: Path, device: str | torch.device = None) -> ConfigTracker:
    """Load model settings from a YAML configuration file and return a ConfigTracker instance.

    This function reads hyperparameters from a 'hparams.yaml' file located in the specified
    model directory. It handles deserialization of pickled objects (scaler and device) and
    allows overriding the device setting.

    Args:
        model_dir (Path): Path to the model directory containing the 'hparams.yaml' file.
        device (str | torch.Device, optional): Device to use for the model (e.g., 'cpu', 'cuda').
            If provided, this overrides any device setting in the YAML file. Defaults to None.

    Returns:
        ConfigTracker: A ConfigTracker instance initialized with the loaded configuration data.

    Raises:
        FileNotFoundError: If the 'hparams.yaml' file does not exist in the model directory.
        yaml.YAMLError: If the YAML file cannot be parsed.

    Notes:
        - Pickled scaler, known_scaler and device objects are automatically deserialized if present in the YAML.
        - If device deserialization fails, it is silently suppressed and the YAML value is retained.

    """
    with (model_dir / "hparams.yaml").open(encoding="utf-8") as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        for key in ("scaler", "known_scaler"):
            if key in yaml_data:
                yaml_data[key] = pickle.loads(yaml_data[key])

        if device is not None:
            yaml_data["device"] = device
        elif "device" in yaml_data:
            with contextlib.suppress(TypeError):
                yaml_data["device"] = pickle.loads(yaml_data["device"])
    return ConfigTracker(yaml_data)


def resolve_class(name_or_class: str | type, registry: dict[str, type], arg_name: str) -> type:
    """Return the class for a registry name, or the class itself if one was passed."""
    if isinstance(name_or_class, str):
        if name_or_class not in registry:
            msg = f"Unknown {arg_name} {name_or_class!r}, choose one of {list(registry)} or pass a class"
            raise KeyError(msg)
        return registry[name_or_class]
    if inspect.isclass(name_or_class):
        return name_or_class
    msg = f"{arg_name} must be either a registry name string or a class"
    raise TypeError(msg)


def class_name(name_or_class: str | type, registry: dict[str, type]) -> str:
    """Return the registry name for a class, falling back to its __name__. Strings are returned as-is.

    Used to store a class argument in hparams, which only hold serialisable values.
    """
    if isinstance(name_or_class, str):
        return name_or_class
    for name, cls in registry.items():
        if cls is name_or_class:
            return name
    return name_or_class.__name__
