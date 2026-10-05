"""Building blocks for forecast tables: data loading, running a model over samples, the table layout.

The forecast table (see ``Analysis.predict``) is long, one row per forecast value::

    run, label, [member], issue_time, step, valid_time, prediction, observed, [q0.1, q0.5, ...]

``member`` only exists for ensemble forecasts, the quantile columns only for quantile models.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from forecastlib.analysis.runs import _as_list, _is_missing, _label

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from forecastlib.models.LightningWrapper import CustomLightningModule

FORECAST_COLUMNS = ["run", "label", "member", "issue_time", "step", "valid_time", "prediction", "observed"]


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
        root_path: Override the folder of the data file.
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
    """Return the quantile columns (q0.1, q0.5, ...) of a forecast table, lowest level first."""
    return [c for c in forecasts.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]


def select_times(
    times: Iterable[pd.Timestamp],
    *,
    start: Any = None,
    end: Any = None,
    every: int | None = None,
    hours: int | Iterable[int] | None = None,
) -> pd.DatetimeIndex:
    """Keep the times in [start, end], at the given hours of the day, then every n-th of those."""
    times = pd.DatetimeIndex(times).unique().sort_values()
    if start is not None:
        times = times[times >= pd.Timestamp(start)]
    if end is not None:
        times = times[times <= pd.Timestamp(end)]
    if hours is not None:
        times = times[times.hour.isin(_as_list(hours))]
    if every is not None:
        times = times[::every]
    return times


def run_model(
    module: CustomLightningModule, samples: Sequence, batch_size: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict samples ((seq_x, seq_y, seq_x_mark, seq_y_mark) tuples) with a CustomLightningModule.

    Returns:
        The predictions [N, pred_len, n_targets] and, for quantile models, the quantiles [N, pred_len, n_targets,
        n_quantiles], both in the target's original units.

    """
    module.eval()
    quantile_model = module.quantiles is not None
    if quantile_model:
        module.set_return_quantiles(True)
    preds, quantiles = [], []
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(samples, batch_size=batch_size, shuffle=False)):
            out = module.predict_step([torch.as_tensor(b).to(module.device) for b in batch], i)
            preds.append(out[1])
            if quantile_model:
                quantiles.append(out[3])
    pred = torch.cat(preds).numpy()
    return pred, torch.cat(quantiles).numpy() if quantile_model else None


def forecast_rows(
    issue_times: Sequence[pd.Timestamp],
    valid_times: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray,
    quantiles: np.ndarray | None = None,
    quantile_levels: Sequence[float] = (),
    members: Sequence | None = None,
) -> pd.DataFrame:
    """Lay out N forecasts of pred_len steps as forecast table rows (without run and label).

    valid_times, prediction and observed are [N, pred_len], quantiles [N, pred_len, n_quantiles].
    """
    n, pred_len = prediction.shape
    data = {
        "issue_time": np.repeat(pd.DatetimeIndex(issue_times), pred_len),
        "step": np.tile(np.arange(1, pred_len + 1), n),
        "valid_time": pd.DatetimeIndex(np.asarray(valid_times).ravel()),
        "prediction": prediction.ravel(),
        "observed": np.asarray(observed, dtype=float).ravel(),
    }
    if members is not None:
        data = {"member": np.repeat(np.asarray(members, dtype=object), pred_len), **data}
    rows = pd.DataFrame(data)
    for j, level in enumerate(quantile_levels):
        rows[f"q{level:g}"] = quantiles[:, :, j].ravel()
    return rows


def run_labels(runs: pd.DataFrame, label: str | list[str] | None) -> dict[str, str]:
    """Legend label per run: its id, or the values of the given columns (made unique if needed)."""
    per_run = runs.drop_duplicates("run")
    ids = per_run["run"].astype(str)
    if label is None:
        return dict(zip(ids, ids, strict=True))
    columns = [label] if isinstance(label, str) else list(label)
    names = per_run[columns].apply(lambda row: " / ".join(_label(v) for v in row), axis=1)
    names.index = ids
    counts = names.groupby(names).cumcount()
    duplicated = names.duplicated(keep=False)
    names[duplicated] = names[duplicated] + " #" + (counts[duplicated] + 1).astype(str)
    return names.to_dict()
