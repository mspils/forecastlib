"""All kind of handy little tools, mostly for handling configs."""

import contextlib
import pickle
from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml


class ConfigTracker:
    """Basically a dotdict that keeps track of what has been accessed via self.var."""

    # pylint: disable=no-member
    def __init__(self, config: dict | Namespace | Any) -> None:
        """Create a ConfigTracker from a dict or similar."""
        self._config = config
        self.accessed_attrs = set()

    def __getattr__(self, name: str) -> Any:
        """Return items from self._config.

        self._config may be a dict or a ConfigTracker and adds the key to self.accessed_attrs.
        """
        try:
            v = self._config[name] if isinstance(self._config, dict) else getattr(self._config, name)
        except KeyError:
            raise AttributeError(name)
        self.accessed_attrs.add(name)
        return v

    def __getitem__(self, name: str) -> Any:
        """Return items from self._config which in itself may be a dict or a ConfigTracker."""
        return self._config[name] if isinstance(self._config, dict) else getattr(self._config, name)

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for membership testing."""
        if isinstance(self._config, dict):
            return name in self._config
        return hasattr(self._config, name)

    def __str__(self) -> str:
        """Return (str(self._config))."""
        return str(self._config)

    def __repr__(self) -> str:
        """Return string representation os self._config."""
        return self._config.__repr__()


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
        - Pickled scaler and device objects are automatically deserialized if present in the YAML.
        - If device deserialization fails, it is silently suppressed and the YAML value is retained.

    """
    with (model_dir / "hparams.yaml").open(encoding="utf-8") as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        if "scaler" in yaml_data:
            yaml_data["scaler"] = pickle.loads(yaml_data["scaler"])

        if device is not None:
            yaml_data["device"] = device
        elif "device" in yaml_data:
            with contextlib.suppress(TypeError):
                yaml_data["device"] = pickle.loads(yaml_data["device"])
    return ConfigTracker(yaml_data)
