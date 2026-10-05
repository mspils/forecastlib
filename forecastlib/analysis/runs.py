"""Load logged runs into pandas DataFrames: per-horizon metrics or training curves, with the hyperparameters joined.

A run is any folder containing an ``hparams.yaml``, as written by Lightning's TensorBoardLogger and CSVLogger
(e.g. ``<experiment>/<model>/lightning_logs/version_0``). Its scalars are read from the TensorBoard event files, or
from ``metrics.csv`` if there are none. They are split into:

- per-horizon metrics, logged by the metric callbacks at steps 1..pred_len, e.g. ``test_nse``: ``load_runs``. The
  prefix becomes ``split`` (train/val/test), the rest ``metric`` (nse), the step is the forecast step.
- everything logged during training (train_loss_step, val_loss, epoch, hp_metric, ...): ``load_history``.

Both return one long DataFrame with one row per value and the run's hyperparameters as extra columns, so runs are
filtered and grouped with plain pandas, e.g. ``runs[runs.model == "LSTM"]``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

SPLITS = ("train", "val", "test")
BANDS = (None, "std", "minmax")
METRIC_COLUMNS = ["run", "path", "split", "metric", "step", "value"]
HISTORY_COLUMNS = ["run", "path", "tag", "step", "value", "wall_time"]


def load_runs(paths: str | Path | Iterable[str | Path], source: str = "auto") -> pd.DataFrame:
    """Per-horizon metrics of every run below one or more folders.

    Args:
        paths: Folder(s) to search recursively for runs (folders with an hparams.yaml).
        source: Where to read the scalars from: "tensorboard" (event files), "csv" (metrics.csv) or "auto" (event
            files if present, otherwise metrics.csv).

    Returns:
        Long DataFrame with run (the run folder relative to the loaded folder), path (the run folder, e.g. for
        CustomLightningModule.from_disk), split, metric, step (forecast step), value, then one column per
        hyperparameter. Pickled hyperparameters (scaler, device) are left out, lists become tuples.

    """
    return _load(paths, source=source, history=False)


def load_history(paths: str | Path | Iterable[str | Path], source: str = "auto") -> pd.DataFrame:
    """Training curves of every run below one or more folders: run, path, tag, step, value, wall_time, hparams.

    step is the global training step, wall_time is NaN for metrics.csv. See load_runs for the arguments.
    """
    return _load(paths, source=source, history=True)


def hparam_columns(df: pd.DataFrame) -> list[str]:
    """Return the hyperparameter columns of a load_runs / load_history table."""
    return [c for c in df.columns if c not in {*METRIC_COLUMNS, *HISTORY_COLUMNS}]


def varying_hparams(df: pd.DataFrame) -> list[str]:
    """Hyperparameters that differ between the runs in df (a missing value counts as a value)."""
    per_run = df.drop_duplicates("run")
    return [c for c in hparam_columns(per_run) if per_run[c].nunique(dropna=False) > 1]


def summary(
    df: pd.DataFrame,
    metrics: str | list[str] | None = None,
    *,
    split: str = "test",
    step: int | Iterable[int] | None = None,
    hparams: list[str] | None = None,
) -> pd.DataFrame:
    """One row per run: hyperparameters and each metric, averaged over the forecast steps.

    Args:
        df: A load_runs table (or a filtered part of it).
        metrics: Metric name(s), default all.
        split: Which split's metrics to use.
        step: Use only this forecast step (or these steps) instead of averaging over all of them.
        hparams: Hyperparameter columns to include, default the varying ones.

    """
    metrics = sorted(df["metric"].unique()) if metrics is None else _as_list(metrics)
    data = _select(df, metrics, [split])
    if step is not None:
        data = data[data["step"].isin(_as_list(step))]
    values = data.pivot_table(index="run", columns="metric", values="value", aggfunc="mean", observed=True)
    values = values.reindex(columns=metrics)
    values.columns.name = None
    hparams = varying_hparams(df) if hparams is None else hparams
    per_run = df.drop_duplicates("run").set_index("run")[hparams]
    per_run.index = per_run.index.astype(str)
    values.index = values.index.astype(str)
    return per_run.join(values)


def aggregate(
    df: pd.DataFrame,
    metrics: str | list[str],
    *,
    split: str | list[str] = "test",
    by: str | list[str] | None = None,
    agg: str | Callable = "mean",
    band: str | None = "std",
) -> pd.DataFrame:
    """Aggregate metrics per forecast step over the runs in each group of ``by``.

    Args:
        df: A load_runs table (or a filtered part of it).
        metrics: Metric name(s).
        split: Split name(s).
        by: Column(s) to group the runs by, usually hyperparameters. None keeps every run separate.
        agg: Aggregation per step, anything DataFrameGroupBy.agg takes ("mean", "median", "min", "max", ...).
        band: Spread around the value: "std" (value ± standard deviation), "minmax" or None.

    Returns:
        Long table with the ``by`` columns (or ``run``), ``group`` (the legend label), split, metric, step, value,
        lower, upper and n_runs.

    """
    if band not in BANDS:
        msg = f"band must be one of {BANDS}, got {band!r}"
        raise ValueError(msg)
    data = _select(df, _as_list(metrics), _as_list(split))
    if by is None:
        data = data.assign(group=data["run"].astype(str), lower=np.nan, upper=np.nan, n_runs=1)
        columns = ["run", "group", "split", "metric", "step", "value", "lower", "upper", "n_runs"]
        return data[columns].reset_index(drop=True)

    by = _as_list(by)
    unknown = [b for b in by if b not in df.columns]
    if unknown:
        msg = f"Unknown column(s) {unknown}"
        raise KeyError(msg)
    grouped = data.groupby([*by, "split", "metric", "step"], dropna=False, sort=False, observed=True)["value"]
    result = grouped.agg(value=agg, std="std", min="min", max="max", n_runs="count").reset_index()

    if band == "std":
        result["lower"] = result["value"] - result["std"]
        result["upper"] = result["value"] + result["std"]
    elif band == "minmax":
        result["lower"], result["upper"] = result["min"], result["max"]
    else:
        result["lower"] = result["upper"] = np.nan
    result["group"] = result[by].apply(lambda row: " / ".join(_label(v) for v in row), axis=1)
    return result[[*by, "group", "split", "metric", "step", "value", "lower", "upper", "n_runs"]]


def _load(paths: str | Path | Iterable[str | Path], *, source: str, history: bool) -> pd.DataFrame:
    if source not in {"auto", "tensorboard", "csv"}:
        msg = f"source must be 'auto', 'tensorboard' or 'csv', got {source!r}"
        raise ValueError(msg)
    roots = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]

    frames, hparam_rows = [], {}
    for root in roots:
        if not root.exists():
            msg = f"{root} does not exist"
            raise FileNotFoundError(msg)
        run_dirs = sorted({p.parent for p in root.rglob("hparams.yaml")})
        if not run_dirs:
            warnings.warn(f"No runs (folders with an hparams.yaml) found below {root}", stacklevel=3)

        for run_dir in run_dirs:
            run = _run_id(run_dir, root, prefix_root=len(roots) > 1)
            hparams = _read_hparams(run_dir / "hparams.yaml")
            scalars = _read_scalars(run_dir, source)
            if scalars.empty:
                warnings.warn(f"No logged scalars found for run {run}", stacklevel=3)
            metrics, curves = _split_scalars(scalars, hparams.get("pred_len"))
            frames.append((curves if history else metrics).assign(run=run, path=run_dir))
            hparam_rows[run] = hparams

    columns = HISTORY_COLUMNS if history else METRIC_COLUMNS
    frames = [f for f in frames if not f.empty]
    data = pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)

    hparams = pd.DataFrame.from_dict(hparam_rows, orient="index")
    clashes = [c for c in hparams.columns if c in {*METRIC_COLUMNS, *HISTORY_COLUMNS}]
    if clashes:
        warnings.warn(f"Hyperparameters {clashes} renamed to hparam_<name>, they clash with columns", stacklevel=3)
        hparams = hparams.rename(columns={c: f"hparam_{c}" for c in clashes})
    data = data.join(hparams, on="run")
    # Many rows per run: store the run ids once.
    data["run"] = pd.Categorical(data["run"], categories=list(hparam_rows))
    return data


def _run_id(run_dir: Path, root: Path, *, prefix_root: bool) -> str:
    rel = run_dir.relative_to(root).as_posix()
    run = run_dir.name if rel == "." else rel
    return f"{root.name}/{run}" if prefix_root else run


def _read_hparams(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader) or {}  # FullLoader like load_model_settings: !!binary, tuples
    # Pickled objects (scaler, device) can't be compared or grouped, lists can't be hashed.
    return {k: _hashable(v) for k, v in raw.items() if not isinstance(v, bytes)}


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    return value


def _read_scalars(run_dir: Path, source: str) -> pd.DataFrame:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    csv_file = run_dir / "metrics.csv"
    if source == "tensorboard" or (source == "auto" and event_files):
        scalars = _read_tensorboard(run_dir) if event_files else _empty_scalars()
    elif csv_file.exists():
        scalars = _read_csv(csv_file)
    else:
        scalars = _empty_scalars()
    # A step can be logged twice (hp_metric, resumed runs), keep the latest value.
    return scalars.drop_duplicates(["tag", "step"], keep="last")


def _read_tensorboard(run_dir: Path) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415 - slow import

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})  # 0: keep all values
    accumulator.Reload()
    rows = [
        (tag, event.step, event.value, event.wall_time)
        for tag in accumulator.Tags()["scalars"]
        for event in accumulator.Scalars(tag)
    ]
    return pd.DataFrame(rows, columns=["tag", "step", "value", "wall_time"])


def _read_csv(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path)
    scalars = wide.melt(id_vars="step", var_name="tag", value_name="value").dropna(subset=["value"])
    return scalars.assign(wall_time=np.nan)[["tag", "step", "value", "wall_time"]]


def _empty_scalars() -> pd.DataFrame:
    return pd.DataFrame(columns=["tag", "step", "value", "wall_time"])


def _split_scalars(scalars: pd.DataFrame, pred_len: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate per-horizon metrics (steps exactly 1..pred_len) from the training history."""
    horizon_tags = []
    for tag, group in scalars.groupby("tag"):
        steps = sorted(group["step"].astype(int))
        n = len(steps)
        expected = n if pred_len is None else pred_len
        if tag != "epoch" and steps == list(range(1, n + 1)) and n == expected and (n > 1 or pred_len == 1):
            horizon_tags.append(tag)

    is_horizon = scalars["tag"].isin(horizon_tags)
    metrics = scalars[is_horizon].copy()
    parts = metrics["tag"].str.split("_", n=1, expand=True).reindex(columns=[0, 1])
    has_split = parts[0].isin(SPLITS) & parts[1].notna()
    metrics["split"] = parts[0].where(has_split, "")
    metrics["metric"] = parts[1].where(has_split, metrics["tag"])
    metrics["step"] = metrics["step"].astype(int)
    return metrics[["split", "metric", "step", "value"]], scalars[~is_horizon]


def _select(df: pd.DataFrame, metrics: list[str], splits: list[str]) -> pd.DataFrame:
    for name, wanted in (("metric", metrics), ("split", splits)):
        available = set(df[name].unique())
        unknown = [w for w in wanted if w not in available]
        if unknown:
            msg = f"Unknown {name}(s) {unknown}, available: {sorted(available)}"
            raise KeyError(msg)
    return df[df["metric"].isin(metrics) & df["split"].isin(splits)]


def _as_list(value: Any) -> list:
    if isinstance(value, (list, tuple, set, range, pd.Index, np.ndarray)):
        return list(value)
    return [value]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _label(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # an int hyperparameter missing in some runs is stored as float
    return str(value)
