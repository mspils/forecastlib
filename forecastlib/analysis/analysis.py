"""Object based analysis of an experiment: its runs, their metrics, and their forecasts on the data.

``Analysis`` loads everything through a few hooks, so other data sources are a subclass away::

    class ParquetAnalysis(Analysis):
        def load_observations(self):
            return pd.read_parquet("data/bille.parquet")  # indexed by time, the columns of the training data

    class EnsembleAnalysis(ParquetAnalysis):
        def load_members(self, issue_time):
            # ensemble forecasts issued at issue_time, long: member, valid_time, <forecast columns>
            return read_members_from_grib(issue_time)

    analysis = EnsembleAnalysis("logs/experiment_1").filter("model == 'LSTM'")
    forecasts = analysis.predict(start="2022-02-18", end="2022-02-20", hours=12, label="num_layers")
    analysis.plot_forecasts(forecasts, secondary="yw_reinbek")

Runs trained with model or dataset classes outside forecastlib's registries need those classes to be loaded::

    analysis = Analysis("logs/experiment_1", model_classes=[MyModel], dataset_classes=[MyDataset])

Hooks: load_runs (the metric logs), load_observations (the data), load_members (ensemble forecasts per issue time),
load_model (a run's checkpoint), make_dataset (training preprocessing on a frame), make_windows (the model inputs of
one issue time).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from forecastlib.analysis import plots
from forecastlib.analysis.forecasts import (
    FORECAST_COLUMNS,
    forecast_rows,
    load_data,
    run_labels,
    run_model,
    select_times,
)
from forecastlib.analysis.runs import aggregate, load_runs, summary, varying_hparams

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from forecastlib.data_provider.data_loader import Dataset_Custom
    from forecastlib.models.LightningWrapper import CustomLightningModule
    from forecastlib.utils.tools import ConfigTracker

Sample = tuple  # (seq_x, seq_y, seq_x_mark, seq_y_mark), as the datasets return them


class Analysis:
    """The runs of an experiment, their metrics and forecasts.

    Args:
        logs: Folder(s) with the run folders, passed to load_runs. Alternatively ``runs``.
        runs: An already loaded (and maybe filtered) load_runs table.
        device: Device for the models, e.g. "cuda".
        batch_size: Prediction batch size.
        unshift: {column: n} for columns of the data file whose row t holds the value of t + n (an artificial
            forecast shifted in the file itself). For plotting they are moved n rows later, back to their own time.
            Only the ``secondary`` series of the plots are affected, the models still get the columns as in the file.
            Columns shifted by the dataset (shift_cols) are at their own time in the file already.
        model_classes: Model classes outside forecastlib's registry that runs were trained with, e.g.
            ``[TinyLinear]``. hparams.yaml only stores a custom class's name, so loading needs the class; it is
            matched by ``__name__``, or pass a dict {name: class}. Built-in models need nothing.
        dataset_classes: The same for dataset classes outside the registry (subclasses of Dataset_Custom).

    Attributes:
        runs: The load_runs table: per-horizon metrics with the hyperparameters. Plain pandas, filter it directly
            or with ``filter``.

    """

    def __init__(
        self,
        logs: str | Path | Iterable[str | Path] | None = None,
        *,
        runs: pd.DataFrame | None = None,
        device: str = "cpu",
        batch_size: int = 256,
        unshift: dict[str, int] | None = None,
        model_classes: Iterable[type] | dict[str, type] | None = None,
        dataset_classes: Iterable[type] | dict[str, type] | None = None,
    ) -> None:
        """Load the runs from logs, or take an already loaded runs table."""
        if (logs is None) == (runs is None):
            msg = "Pass either logs or runs"
            raise ValueError(msg)
        self.runs = self.load_runs(logs) if runs is None else runs
        self.device = device
        self.batch_size = batch_size
        self.unshift = dict(unshift or {})
        self.model_classes = _by_name(model_classes)
        self.dataset_classes = _by_name(dataset_classes)
        self._observations = None
        self._plot_observations = None
        # Per run, shared with filtered copies.
        self._models: dict[str, Any] = {}
        self._settings: dict[str, ConfigTracker] = {}

    def __repr__(self) -> str:
        """Show the number of runs and the hyperparameters that differ between them."""
        n_runs = self.runs["run"].nunique()
        return f"{type(self).__name__}({n_runs} runs, varying hparams={self.varying_hparams()})"

    # Hooks: override these for other data sources.

    def load_runs(self, logs: str | Path | Iterable[str | Path]) -> pd.DataFrame:
        """Load the runs' metrics and hyperparameters, see load_runs."""
        return load_runs(logs)

    def load_observations(self) -> pd.DataFrame:
        """Return the data the models run on: indexed by time, with the columns of the training data file.

        Default: the data file of the runs' hparams (root_path, data_path, date_col).
        """
        return load_data(self.runs)

    def load_members(self, issue_time: pd.Timestamp) -> pd.DataFrame | None:  # noqa: ARG002 - a hook
        """Return the ensemble forecasts issued at issue_time, or None for a deterministic forecast.

        Long DataFrame with member, valid_time and the forecast columns (e.g. precipitation). For each member the
        values replace the observations at their valid times after issue_time, then the inputs are built like in
        training: through shift_cols they reach the end of seq_x, through known_cols the future marks. Overriding
        this switches predict to building the inputs per issue time.
        """
        return None

    def load_model(self, path: Path) -> Any:
        """Load a run's model, with its class from model_classes if it isn't a built-in model."""
        from forecastlib.models.LightningWrapper import CustomLightningModule, model_dict  # noqa: PLC0415
        from forecastlib.utils.tools import load_model_settings  # noqa: PLC0415

        name = load_model_settings(path).model
        model_class = _custom_class(name, self.model_classes, model_dict, "model_classes")
        return CustomLightningModule.from_disk(path, device=self.device, model_class=model_class)

    def make_dataset(self, settings: ConfigTracker, frame: pd.DataFrame, flag: str) -> Dataset_Custom:
        """Build the run's dataset on a frame with the scalers saved with the model, as in training.

        Default: the run's dataset class (Dataset_Custom or a subclass, from dataset_classes if it isn't a built-in
        one), see Dataset_Custom for frame and flag.
        """
        from forecastlib.data_provider.data_loader import Dataset_Custom  # noqa: PLC0415
        from forecastlib.data_provider.data_module import data_dict  # noqa: PLC0415

        name = settings.data
        dataset_class = _custom_class(name, self.dataset_classes, data_dict, "dataset_classes") or data_dict[name]
        if not issubclass(dataset_class, Dataset_Custom):
            msg = f"{dataset_class.__name__} can't be built from a frame, override make_dataset"
            raise TypeError(msg)
        return dataset_class(
            settings,
            root_path=None,
            frame=frame,
            flag=flag,
            size=(settings.seq_len, settings.label_len, settings.pred_len),
            features=settings.features,
            target=settings.target,
            timeenc=0 if settings.embed != "timeF" else 1,
            freq=settings.freq,
            scaler=settings.scaler,
            known_scaler=getattr(settings, "known_scaler", None),
        )

    def make_windows(
        self,
        settings: ConfigTracker,
        observations: pd.DataFrame,
        issue_time: pd.Timestamp,
        members: pd.DataFrame | None,
    ) -> list[tuple[Any, Sample]]:
        """Return (member, sample) for every member of one issue time, (None, sample) without members.

        Default: cut the rows around issue_time, overwrite the member's forecast columns at their valid times and
        run that through make_dataset with flag "all". Override to build the inputs differently, e.g. with an
        operational dataset that takes the members directly.
        """
        window = _window(settings, observations, issue_time)
        variants = [(None, window)] if members is None else list(_member_windows(window, members, issue_time))
        return [(member, self.make_dataset(settings, frame, "all")[0]) for member, frame in variants]

    # Runs and metrics.

    def filter(self, condition: str | pd.Series | Callable[[pd.DataFrame], pd.Series]) -> Analysis:
        """Return an Analysis with only some runs: a query string, a boolean mask on runs, or a function of runs.

        E.g. ``analysis.filter("model == 'LSTM'")`` or ``analysis.filter(analysis.runs.dropout < 0.3)``. Loaded models
        are shared with the copy.
        """
        if isinstance(condition, str):
            runs = self.runs.query(condition)
        else:
            mask = condition(self.runs) if callable(condition) else condition
            runs = self.runs[mask]
        filtered = copy.copy(self)
        filtered.runs = runs
        # Our own copy; another subset may use another data file.
        filtered._observations = filtered._plot_observations = None  # noqa: SLF001
        return filtered

    def varying_hparams(self) -> list[str]:
        """See varying_hparams."""
        return varying_hparams(self.runs)

    def summary(self, metrics: str | list[str] | None = None, **options: Any) -> pd.DataFrame:
        """See summary."""
        return summary(self.runs, metrics, **options)

    def aggregate(self, metrics: str | list[str], **options: Any) -> pd.DataFrame:
        """See aggregate."""
        return aggregate(self.runs, metrics, **options)

    def plot_horizon(self, metrics: str | list[str], **options: Any) -> plots.FigureResult:
        """See plot_horizon."""
        return plots.plot_horizon(self.runs, metrics, **options)

    def plot_hparam(self, metric: str, hparam: str, **options: Any) -> plots.FigureResult:
        """See plot_hparam."""
        return plots.plot_hparam(self.runs, metric, hparam, **options)

    # Forecasts.

    @property
    def observations(self) -> pd.DataFrame:
        """load_observations, loaded once."""
        if self._observations is None:
            self._observations = self.load_observations().sort_index()
        return self._observations

    @property
    def plot_observations(self) -> pd.DataFrame:
        """The observations as shown in plots: with the ``unshift`` columns moved back to their own time."""
        if self._plot_observations is None:
            observations = self.observations.copy()
            for column, n in self.unshift.items():
                if column not in observations.columns:
                    msg = f"unshift: {column!r} is not a column of the observations"
                    raise KeyError(msg)
                observations[column] = observations[column].shift(n)
            self._plot_observations = observations
        return self._plot_observations

    def model(self, run: str) -> Any:
        """Return the loaded model of a run, loaded once."""
        if run not in self._models:
            self._models[run] = self.load_model(self._path(run))
        return self._models[run]

    def settings(self, run: str) -> ConfigTracker:
        """All hyperparameters of a run (model and data), with the scalers unpickled."""
        from forecastlib.utils.tools import load_model_settings  # noqa: PLC0415

        if run not in self._settings:
            self._settings[run] = load_model_settings(self._path(run))
        return self._settings[run]

    def predict(
        self,
        *,
        split: str = "test",
        start: Any = None,
        end: Any = None,
        every: int | None = None,
        hours: int | Iterable[int] | None = None,
        issue_times: Iterable[Any] | None = None,
        label: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Predict every run for the selected issue times.

        Without load_members the issue times come from the split and are predicted in batches. With load_members
        the inputs are built per issue time and member (make_windows), from the issue times in [start, end] of the
        observations, or ``issue_times``.

        Args:
            split: "train", "val" or "test"; only without load_members.
            start: Only issue times at or after start (anything pd.Timestamp accepts).
            end: Only issue times at or before end.
            every: Only every n-th of the remaining issue times.
            hours: Only issue times at these hours of the day, e.g. 12 for the midday runs.
            issue_times: Exactly these issue times instead (with load_members).
            label: Column(s) of runs to name the runs by in plots, e.g. "model". Default the run id.

        Returns:
            The forecast table, see forecastlib.analysis.forecasts. Ensembles have a member column.

        """
        labels = run_labels(self.runs, label)
        ensemble = type(self).load_members is not Analysis.load_members
        if not ensemble and split not in {"train", "val", "test"}:
            msg = f"split must be 'train', 'val' or 'test', got {split!r}"
            raise ValueError(msg)

        frames = []
        for run in labels:
            if ensemble:
                times = self._ensemble_issue_times(issue_times, start, end, every, hours)
                data = self._predict_windows(run, times)
            else:
                data = self._predict_split(run, split, start=start, end=end, every=every, hours=hours)
            frames.append(data.assign(run=run, label=labels[run]))
        frames = [f for f in frames if not f.empty]
        if not frames:
            msg = "No issue times selected"
            raise ValueError(msg)

        data = pd.concat(frames, ignore_index=True)
        # Millions of rows for a long test split: store the repeated strings once.
        data["run"] = pd.Categorical(data["run"], categories=list(labels))
        data["label"] = pd.Categorical(data["label"], categories=list(dict.fromkeys(labels.values())))
        columns = [c for c in FORECAST_COLUMNS if c in data.columns]
        return data[columns + [c for c in data.columns if c not in columns]]

    def plot_issue(self, forecasts: pd.DataFrame, issue_time: Any, **options: Any) -> plots.FigureResult:
        """See plot_issue. Uses the full observations, ``secondary`` may also be column name(s) of them."""
        return plots.plot_issue(forecasts, issue_time, **self._plot_options(forecasts, options))

    def plot_lead(self, forecasts: pd.DataFrame, steps: int | Iterable[int], **options: Any) -> plots.FigureResult:
        """See plot_lead. Uses the full observations, ``secondary`` may also be column name(s) of them."""
        return plots.plot_lead(forecasts, steps, **self._plot_options(forecasts, options))

    def plot_forecasts(self, forecasts: pd.DataFrame, **options: Any) -> plots.FigureResult:
        """See plot_forecasts. Uses the full observations, ``secondary`` may also be column name(s) of them."""
        return plots.plot_forecasts(forecasts, **self._plot_options(forecasts, options))

    # Internals.

    def _path(self, run: str) -> Path:
        paths = self.runs.loc[self.runs["run"] == run, "path"]
        if paths.empty:
            msg = f"Unknown run {run!r}"
            raise KeyError(msg)
        return paths.iloc[0]

    def _predict_split(self, run: str, split: str, **selection: Any) -> pd.DataFrame:
        settings, module = self.settings(run), self.model(run)
        dataset = self.make_dataset(settings, self.observations, split)
        seq_len = dataset.seq_len
        all_issue_times = dataset.dates[np.arange(len(dataset)) + seq_len - 1]
        samples = np.flatnonzero(all_issue_times.isin(select_times(all_issue_times, **selection)))
        if len(samples) == 0:
            return pd.DataFrame()

        pred, quantiles = run_model(module, Subset(dataset, samples), self.batch_size)
        rows = samples[:, None] + seq_len + np.arange(pred.shape[1])[None, :]  # dataset row of each value
        channel = _target_channel(module, dataset, settings.target)
        return forecast_rows(
            all_issue_times[samples],
            dataset.dates[rows.ravel()].to_numpy().reshape(rows.shape),
            pred[:, :, channel],
            dataset.data_x_raw[settings.target].to_numpy()[rows],
            None if quantiles is None else quantiles[:, :, channel],
            module.quantiles or (),
        )

    def _predict_windows(self, run: str, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        settings, module = self.settings(run), self.model(run)
        observations = self.observations
        keys, samples = [], []
        for issue_time in issue_times:
            for member, sample in self.make_windows(settings, observations, issue_time, self.load_members(issue_time)):
                keys.append((issue_time, member))
                samples.append(sample)
        if not samples:
            return pd.DataFrame()

        pred, quantiles = run_model(module, samples, self.batch_size)
        pred_len = pred.shape[1]
        issued = pd.DatetimeIndex([k[0] for k in keys])
        step = _time_step(observations.index)
        valid = np.stack([pd.date_range(t + step, periods=pred_len, freq=step) for t in issued])
        observed = observations[settings.target].reindex(valid.ravel()).to_numpy().reshape(valid.shape)
        channel = 0
        if len(module.target_idx) > 1:  # several targets (features M): find the target in the dataset's columns
            dataset = self.make_dataset(settings, _window(settings, observations, issued[0]), "all")
            channel = _target_channel(module, dataset, settings.target)
        return forecast_rows(
            issued,
            valid,
            pred[:, :, channel],
            observed,
            None if quantiles is None else quantiles[:, :, channel],
            module.quantiles or (),
            members=None if all(k[1] is None for k in keys) else [k[1] for k in keys],
        )

    def _ensemble_issue_times(
        self,
        issue_times: Iterable[Any] | None,
        start: Any,
        end: Any,
        every: int | None,
        hours: int | Iterable[int] | None,
    ) -> pd.DatetimeIndex:
        if issue_times is not None:
            return pd.DatetimeIndex([pd.Timestamp(t) for t in issue_times])
        if start is None or end is None:
            msg = "Ensemble forecasts are built per issue time, pass start and end (or issue_times)"
            raise ValueError(msg)
        return select_times(self.observations.index, start=start, end=end, every=every, hours=hours)

    def _plot_options(self, forecasts: pd.DataFrame, options: dict[str, Any]) -> dict[str, Any]:
        """Fill in the observed target and resolve secondary column names from the observations."""
        options = dict(options)
        secondary = options.get("secondary")
        if isinstance(secondary, str) or (isinstance(secondary, list) and all(isinstance(s, str) for s in secondary)):
            options["secondary"] = self.plot_observations[secondary]
        if options.get("observed") is None:
            options["observed"] = self._observed_target(forecasts)
        return options

    def _observed_target(self, forecasts: pd.DataFrame) -> pd.Series | None:
        """Return the observations of the forecasts' target, None if the runs aren't ours or targets differ."""
        try:
            targets = {self.settings(run).target for run in forecasts["run"].astype(str).unique()}
        except KeyError:
            return None
        if len(targets) != 1 or (target := targets.pop()) not in self.observations.columns:
            return None
        return self.observations[target]


def _by_name(classes: Iterable[type] | dict[str, type] | None) -> dict[str, type]:
    """Return {name: class}: a dict as given, classes under their __name__ (what hparams.yaml stores)."""
    if classes is None:
        return {}
    if isinstance(classes, dict):
        return dict(classes)
    return {cls.__name__: cls for cls in classes}


def _custom_class(name: str, custom: dict[str, type], registry: dict[str, type], option: str) -> type | None:
    """Return the custom class stored under name, None for a built-in one, or fail with a hint."""
    if name in custom:
        return custom[name]
    if name in registry:
        return None
    msg = (
        f"{name!r} is neither built into forecastlib nor in {option}; pass the class: Analysis(..., {option}=[{name}])"
    )
    raise KeyError(msg)


def _time_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    step = pd.Series(index[:100]).diff().mode()
    if step.empty:
        msg = "Can't infer the time step of the observations"
        raise ValueError(msg)
    return step.iloc[0]


def _target_channel(module: CustomLightningModule, dataset: Dataset_Custom, target: str) -> int:
    """Index of the target in the model's predictions: 0, or its column with several targets (features M)."""
    return 0 if len(module.target_idx) == 1 else list(dataset.data_x_raw.columns).index(target)


def _window(settings: ConfigTracker, observations: pd.DataFrame, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Cut the rows a sample issued at issue_time needs: seq_len up to it, pred_len after, the shift_cols margin.

    Rows after the end of the observations are NaN; they only reach the (unused) future part of seq_y.
    """
    seq_len, pred_len = settings.seq_len, settings.pred_len
    shift_cols = getattr(settings, "shift_cols", None)
    margin = max((shift_cols or {}).values(), default=0)  # Dataset_Custom drops these rows after shifting

    position = observations.index.get_indexer([issue_time])[0]
    if position < 0:
        msg = f"{issue_time} is not a time of the observations"
        raise KeyError(msg)
    if position + 1 < seq_len:
        msg = f"{issue_time} has only {position + 1} observations up to it, the model needs seq_len={seq_len}"
        raise ValueError(msg)
    step = _time_step(observations.index)
    past = observations.index[position - seq_len + 1 : position + 1]
    future = pd.date_range(issue_time + step, periods=pred_len + margin, freq=step)
    return observations.reindex(past.append(future))


def _member_windows(
    window: pd.DataFrame, members: pd.DataFrame, issue_time: pd.Timestamp
) -> Iterator[tuple[Any, pd.DataFrame]]:
    """Yield (member, window with the member's forecast in place of the observations after issue_time)."""
    columns = [c for c in members.columns if c not in {"member", "valid_time"}]
    missing = [c for c in columns if c not in window.columns]
    if missing:
        msg = f"Member forecast columns {missing} are not columns of the observations"
        raise KeyError(msg)
    for member, forecast in members.groupby("member", sort=False):
        values = forecast.set_index("valid_time")[columns]
        values = values[(values.index > issue_time) & values.index.isin(window.index)]
        frame = window.copy()
        frame.loc[values.index, columns] = values.to_numpy()
        yield member, frame
