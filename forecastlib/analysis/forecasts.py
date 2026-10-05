"""Run trained models on their data split, giving the forecasts as a DataFrame next to the observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from forecastlib.analysis.runs import _is_missing, _label

FORECAST_COLUMNS = ["run", "label", "issue_time", "step", "valid_time", "prediction", "observed"]


def predict(
    runs: pd.DataFrame,
    *,
    split: str = "test",
    start: Any = None,
    end: Any = None,
    label: str | list[str] | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    **load_options: Any,
) -> pd.DataFrame:
    """Load each run's checkpoint and data, and predict the samples of ``split``.

    Every run in ``runs`` is loaded and predicted, so filter first. To compare runs in one plot they should share
    their data and seq_len/pred_len. Predicting a whole test split takes a while on CPU; ``start``/``end`` limit it
    to the forecasts issued in that period, e.g. the days around a flood.

    Args:
        runs: A load_runs table (or a filtered part of it), it needs the run and path columns.
        split: "train", "val" or "test".
        start: Only predict forecasts issued at or after start (anything pd.Timestamp accepts).
        end: Only predict forecasts issued at or before end.
        label: Column(s) of ``runs`` to name the runs by in plots, e.g. "model". Default the run id.
        device: Device to predict on, e.g. "cuda".
        batch_size: Prediction batch size.
        **load_options: root_path / data_path to override where the data file is (relative paths in hparams.yaml
            are relative to the current working directory); dataset_class / model_class for runs trained with
            classes outside the registries.

    Returns:
        Long DataFrame with run, label, issue_time (time of the last observation the forecast saw), step (1 ..
        pred_len), valid_time, prediction and observed (the observation at valid_time), in the target's original
        units. Quantile models add one column per quantile level, e.g. q0.1, q0.5, q0.9.

    """
    if split not in {"train", "val", "test"}:
        msg = f"split must be 'train', 'val' or 'test', got {split!r}"
        raise ValueError(msg)
    per_run = runs.drop_duplicates("run")
    labels = _labels(per_run, label)

    frames = []
    for run, run_dir in zip(per_run["run"].astype(str), per_run["path"], strict=True):
        data = _predict_run(
            run_dir, split=split, start=start, end=end, device=device, batch_size=batch_size, **load_options
        )
        frames.append(data.assign(run=run, label=labels[run]))
    frames = [f for f in frames if not f.empty]
    if not frames:
        msg = f"No forecasts issued between {start} and {end} in the {split} split"
        raise ValueError(msg)

    data = pd.concat(frames, ignore_index=True)
    # Millions of rows for a long test split: store the repeated strings once.
    data["run"] = pd.Categorical(data["run"], categories=list(labels))
    data["label"] = pd.Categorical(data["label"], categories=list(dict.fromkeys(labels.values())))
    return data[FORECAST_COLUMNS + [c for c in data.columns if c not in FORECAST_COLUMNS]]


def load_data(
    runs: pd.DataFrame,
    *,
    unshift: dict[str, int] | None = None,
    root_path: str | None = None,
    data_path: str | None = None,
) -> pd.DataFrame:
    """Read the data file the runs were trained on, indexed by time, e.g. for the ``secondary`` axis of the plots.

    Columns shifted by the dataset (shift_cols) are shifted while loading, the file itself holds them at their own
    time. Columns that are already shifted in the file can be moved back with ``unshift``.

    Args:
        runs: A load_runs table (or a filtered part of it); all its runs must use the same data file.
        unshift: {column: n} for columns whose row t holds the value of t + n (an artificial forecast). They are
            moved n rows later, back to their own time, like ``data[column].shift(n)``.
        root_path: Override the folder of the data file, like for predict.
        data_path: Override the data file name.

    """
    per_run = runs.drop_duplicates("run")
    files = set()
    for _, run in per_run.iterrows():
        root = root_path or run.get("root_path")
        path = data_path or run.get("data_path")
        if _is_missing(root) or _is_missing(path):
            msg = f"Run {run['run']} has no root_path/data_path hyperparameter, pass root_path and data_path"
            raise KeyError(msg)
        date_col = run.get("date_col")
        files.add((str(Path(root) / path), "date" if _is_missing(date_col) else date_col))
    if len(files) != 1:
        msg = f"The runs use different data files or date columns, filter them first: {sorted(files)}"
        raise ValueError(msg)

    ((file, date_col),) = files
    data = pd.read_csv(file, parse_dates=[date_col]).set_index(date_col).rename_axis("time")
    for column, n in (unshift or {}).items():
        if column not in data.columns:
            msg = f"unshift: {column!r} is not a column of {file}"
            raise KeyError(msg)
        data[column] = data[column].shift(n)
    return data


def quantile_columns(forecasts: pd.DataFrame) -> list[str]:
    """Return the quantile columns (q0.1, q0.5, ...) of a predict table, lowest level first."""
    return [c for c in forecasts.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]


def _predict_run(
    run_dir: Path,
    *,
    split: str,
    start: Any,
    end: Any,
    device: str,
    batch_size: int,
    root_path: str | None = None,
    data_path: str | None = None,
    dataset_class: type | None = None,
    model_class: type | None = None,
) -> pd.DataFrame:
    # Imported here: they pull in every model, which only forecasting needs.
    from forecastlib.data_provider.data_module import CustomDataModule  # noqa: PLC0415
    from forecastlib.models.LightningWrapper import CustomLightningModule  # noqa: PLC0415

    module = CustomLightningModule.from_disk(run_dir, device=device, model_class=model_class)
    datamodule = CustomDataModule.from_disk(
        run_dir, root_path=root_path, data_path=data_path, dataset_class=dataset_class, device=device
    )
    dataset = getattr(datamodule, f"{split}_set")
    if not hasattr(dataset, "dates"):
        msg = f"{type(dataset).__name__} has no dates attribute, forecasts need the time of each row"
        raise TypeError(msg)
    target = datamodule.hparams["target"]
    seq_len = dataset.seq_len
    observed = dataset.data_x_raw[target].to_numpy()

    # Sample i sees rows i .. i+seq_len-1 and forecasts the pred_len rows after them.
    issue_times = dataset.dates[np.arange(len(dataset)) + seq_len - 1]
    in_period = np.ones(len(issue_times), dtype=bool)
    if start is not None:
        in_period &= issue_times >= pd.Timestamp(start)
    if end is not None:
        in_period &= issue_times <= pd.Timestamp(end)
    samples = np.flatnonzero(in_period)
    if len(samples) == 0:
        return pd.DataFrame()

    module.eval()
    if module.quantiles is not None:
        module.set_return_quantiles(True)
    preds, quantile_preds = [], []
    loader = DataLoader(Subset(dataset, samples), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            out = module.predict_step([b.to(module.device) for b in batch], i)
            preds.append(out[1])
            if module.quantiles is not None:
                quantile_preds.append(out[3])

    # pred: [N, pred_len, n_targets]. With several targets (features M) pick the configured target column.
    channel = 0 if len(module.target_idx) == 1 else datamodule.feature_names.index(target)
    pred = torch.cat(preds)[:, :, channel].numpy()
    pred_len = pred.shape[1]

    rows = (samples[:, None] + seq_len + np.arange(pred_len)[None, :]).ravel()  # dataset row of each value
    data = pd.DataFrame(
        {
            "issue_time": np.repeat(issue_times[samples], pred_len),
            "step": np.tile(np.arange(1, pred_len + 1), len(samples)),
            "valid_time": dataset.dates[rows],
            "prediction": pred.ravel(),
            "observed": observed[rows],
        }
    )
    if quantile_preds:
        quantiles = torch.cat(quantile_preds)[:, :, channel].numpy()  # [N, pred_len, n_q]
        for j, level in enumerate(module.quantiles):
            data[f"q{level:g}"] = quantiles[:, :, j].ravel()
    return data


def _labels(per_run: pd.DataFrame, label: str | list[str] | None) -> dict[str, str]:
    """Legend label per run: its id, or the values of the given columns (made unique if needed)."""
    runs = per_run["run"].astype(str)
    if label is None:
        return dict(zip(runs, runs, strict=True))
    columns = [label] if isinstance(label, str) else list(label)
    names = per_run[columns].apply(lambda row: " / ".join(_label(v) for v in row), axis=1)
    names.index = runs
    counts = names.groupby(names).cumcount()
    duplicated = names.duplicated(keep=False)
    names[duplicated] = names[duplicated] + " #" + (counts[duplicated] + 1).astype(str)
    return names.to_dict()
