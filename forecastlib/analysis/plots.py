"""Plotly figures for load_runs and predict tables.

Every function takes the DataFrame (filter it with pandas first) and returns the figure, or (figure, data) with
``return_data=True``, where data holds exactly the values that were plotted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, qualitative
from plotly.subplots import make_subplots

from forecastlib.analysis.forecasts import quantile_columns
from forecastlib.analysis.runs import _as_list, aggregate, summary

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

BAND_OPACITY = 0.2
OBSERVED_COLOR = "#222222"
SECONDARY_COLORS = qualitative.Dark2
TEMPLATE = "plotly_white"

FigureResult = go.Figure | tuple[go.Figure, pd.DataFrame]  # the figure, or (figure, data) with return_data=True


def plot_horizon(
    runs: pd.DataFrame,
    metrics: str | list[str],
    *,
    split: str | list[str] = "test",
    by: str | list[str] | None = None,
    agg: str | Callable = "mean",
    band: str | None = "std",
    title: str | None = None,
    return_data: bool = False,
) -> FigureResult:
    """Plot metrics over the forecast steps, one line per group of ``by`` (or per run).

    Several metrics become subplot columns, several splits subplot rows. With ``by`` the line is the ``agg`` over the
    group's runs and ``band`` ("std", "minmax" or None) is drawn around it. A group keeps its color in every subplot
    and one legend entry toggles all of its lines. Data: the aggregate table.
    """
    data = aggregate(runs, metrics, split=split, by=by, agg=agg, band=band)
    # Subplots in the order asked for, leaving out combinations that were never logged.
    metric_names = [m for m in _as_list(metrics) if m in set(data["metric"])]
    splits = [s for s in _as_list(split) if s in set(data["split"])]
    groups = list(dict.fromkeys(data["group"]))
    colors = _colors(groups)

    fig = make_subplots(
        rows=len(splits),
        cols=len(metric_names),
        shared_xaxes=True,
        subplot_titles=[f"{s} {m}".strip() for s in splits for m in metric_names],
        vertical_spacing=0.12 if len(splits) > 1 else 0.0,
    )
    cells = dict(iter(data.groupby(["group", "split", "metric"], sort=False)))
    for group in groups:
        show_legend = True
        for row, split_name in enumerate(splits, start=1):
            for col, metric in enumerate(metric_names, start=1):
                cell = cells.get((group, split_name, metric))
                if cell is None:
                    continue
                cell = cell.sort_values("step")
                position = {"row": row, "col": col}
                if cell["lower"].notna().any():
                    _add_band(fig, cell["step"], cell["lower"], cell["upper"], group, colors[group], position)
                fig.add_trace(
                    go.Scatter(
                        x=cell["step"],
                        y=cell["value"],
                        mode="lines",
                        name=group,
                        legendgroup=group,
                        showlegend=show_legend,
                        line={"color": colors[group]},
                        customdata=cell[["n_runs"]],
                        hovertemplate=f"{group}<br>step %{{x}}: %{{y:.4g}} (%{{customdata[0]}} runs)<extra></extra>",
                    ),
                    **position,
                )
                show_legend = False

    legend_title = "run" if by is None else ", ".join(_as_list(by))
    fig.update_xaxes(title_text="forecast step", row=len(splits))
    fig.update_layout(title=title, legend_title_text=legend_title, template=TEMPLATE)
    return (fig, data) if return_data else fig


def plot_hparam(
    runs: pd.DataFrame,
    metric: str,
    hparam: str,
    *,
    split: str = "test",
    step: int | Iterable[int] | None = None,
    color: str | None = None,
    log_x: bool = False,
    title: str | None = None,
    return_data: bool = False,
) -> FigureResult:
    """Scatter one metric per run (averaged over the forecast steps, or at ``step``) against a hyperparameter.

    Data: the summary table.
    """
    hparams = [hparam] if color is None or color == hparam else [hparam, color]
    data = summary(runs, metric, split=split, step=step, hparams=hparams)
    plotted = data.reset_index()
    if color is not None and not pd.api.types.is_float_dtype(plotted[color]):
        plotted[color] = plotted[color].astype(str)  # discrete colors for models, bools, ints
    fig = px.scatter(plotted, x=hparam, y=metric, color=color, hover_name="run", log_x=log_x, title=title)
    fig.update_layout(template=TEMPLATE)
    return (fig, data) if return_data else fig


def plot_issue(
    forecasts: pd.DataFrame,
    issue_time: Any,
    *,
    history: int | None = None,
    secondary: pd.Series | pd.DataFrame | None = None,
    title: str | None = None,
    return_data: bool = False,
) -> FigureResult:
    """Plot the forecasts issued at ``issue_time`` (or the latest issue time before it) with the observations.

    Args:
        forecasts: A predict table (or a filtered part of it).
        issue_time: Anything pd.Timestamp accepts.
        history: Number of observed steps shown before the issue time, default 2 * pred_len. Observations are
            taken from the table, so the history only reaches back to the first predicted issue time.
        secondary: Series or DataFrame indexed by time, e.g. columns of load_data, drawn dotted on a second y-axis.
        title: Figure title, default the issue time.
        return_data: Also return the plotted data: label, run, time, value, axis and the quantile columns.

    """
    issue = _resolve_issue_time(forecasts, issue_time)
    issued = forecasts[forecasts["issue_time"] == issue]
    history = 2 * int(forecasts["step"].max()) if history is None else history

    observed = _observed(forecasts)
    before = observed.loc[:issue].iloc[-history:]
    observed = observed.loc[before.index[0] : issued["valid_time"].max()]

    quantiles = quantile_columns(forecasts)
    data = pd.concat(
        [
            _observed_rows(observed),
            issued.rename(columns={"valid_time": "time", "prediction": "value"})[
                ["label", "run", "time", "value", *quantiles]
            ],
        ],
        ignore_index=True,
    ).assign(axis="primary")
    secondary = _secondary_frame(secondary)
    fig = _new_figure(secondary)
    _add_series(fig, data, _labels(forecasts), quantiles)
    data = pd.concat([data, _add_secondary(fig, secondary, observed.index[0], observed.index[-1])], ignore_index=True)
    fig.add_vline(x=issue, line={"dash": "dot", "color": "grey"})
    fig.update_layout(title=title or f"Forecast issued {issue}", template=TEMPLATE, hovermode="x unified")
    return (fig, data) if return_data else fig


def plot_lead(
    forecasts: pd.DataFrame,
    steps: int | Iterable[int],
    *,
    start: Any = None,
    end: Any = None,
    secondary: pd.Series | pd.DataFrame | None = None,
    title: str | None = None,
    return_data: bool = False,
) -> FigureResult:
    """Plot the forecasts for fixed lead times against the observations, one subplot row per step.

    Args:
        forecasts: A predict table (or a filtered part of it).
        steps: Forecast step(s), e.g. [1, 24, 48].
        start: Only show valid times at or after start.
        end: Only show valid times at or before end.
        secondary: Series or DataFrame indexed by time, e.g. columns of load_data, drawn dotted on a second y-axis.
        title: Figure title.
        return_data: Also return the plotted data: step, label, run, time, value, axis and the quantile columns.

    """
    steps = _as_list(steps)
    unknown = sorted(set(steps) - set(forecasts["step"]))
    if unknown:
        msg = f"Unknown step(s) {unknown}, forecasts cover 1..{forecasts['step'].max()}"
        raise KeyError(msg)
    observed = _observed(forecasts).loc[start:end]
    in_period = forecasts["valid_time"].between(observed.index[0], observed.index[-1])

    quantiles = quantile_columns(forecasts)
    frames = []
    for step in steps:
        at_step = forecasts[in_period & (forecasts["step"] == step)]
        frames.append(_observed_rows(observed).assign(step=step))
        frames.append(
            at_step.rename(columns={"valid_time": "time", "prediction": "value"})[
                ["step", "label", "run", "time", "value", *quantiles]
            ]
        )
    data = pd.concat(frames, ignore_index=True)[["step", "label", "run", "time", "value", *quantiles]]
    data["axis"] = "primary"

    labels = _labels(forecasts)
    secondary = _secondary_frame(secondary)
    fig = _new_figure(
        secondary,
        rows=len(steps),
        shared_xaxes=True,
        subplot_titles=[f"step {s}" for s in steps],
        vertical_spacing=0.06,
    )
    secondary_rows = []
    for row, step in enumerate(steps, start=1):
        _add_series(
            fig, data[data["step"] == step], labels, quantiles, position={"row": row, "col": 1}, legend=row == 1
        )
        rows = _add_secondary(fig, secondary, observed.index[0], observed.index[-1], row=row, legend=row == 1)
        secondary_rows.append(rows.assign(step=step))
    data = pd.concat([data, *secondary_rows], ignore_index=True)
    fig.update_layout(title=title, template=TEMPLATE, hovermode="x unified", height=max(450, 250 * len(steps)))
    return (fig, data) if return_data else fig


def plot_forecasts(
    forecasts: pd.DataFrame,
    *,
    start: Any = None,
    end: Any = None,
    every: int | None = None,
    hours: int | Iterable[int] | None = None,
    secondary: pd.Series | pd.DataFrame | None = None,
    title: str | None = None,
    return_data: bool = False,
) -> FigureResult:
    """Plot whole forecasts of several issue times against the observations, one line per forecast.

    Lines are colored by label with one legend entry per label. The issue times are picked with select_forecasts,
    the observations come from the whole table, so they stay continuous however few forecasts are shown.

    Args:
        forecasts: A predict table (or a filtered part of it).
        start: Only forecasts issued at or after start.
        end: Only forecasts issued at or before end.
        every: Only every n-th of the remaining issue times.
        hours: Only forecasts issued at these hours of the day, e.g. 12 for the midday runs.
        secondary: Series or DataFrame indexed by time, e.g. columns of load_data, drawn dotted on a second y-axis.
        title: Figure title.
        return_data: Also return the plotted data: label, run, issue_time, step, time, value, axis and the
            quantile columns. Observations have the label "observed", secondary series their column name.

    """
    selected = select_forecasts(forecasts, start=start, end=end, every=every, hours=hours)
    if selected.empty:
        msg = "No forecasts left after selecting by start, end, every and hours"
        raise ValueError(msg)
    observed = _observed(forecasts).loc[selected["issue_time"].min() : selected["valid_time"].max()]

    columns = ["label", "run", "issue_time", "step", "time", "value", *quantile_columns(forecasts)]
    data = pd.concat(
        [_observed_rows(observed), selected.rename(columns={"valid_time": "time", "prediction": "value"})],
        ignore_index=True,
    )[columns].assign(axis="primary")

    secondary = _secondary_frame(secondary)
    fig = _new_figure(secondary)
    fig.add_trace(
        go.Scatter(x=observed.index, y=observed, mode="lines", name="observed", line={"color": OBSERVED_COLOR})
    )
    labels = _labels(selected)
    colors = _colors(labels)
    for label in labels:
        x, y, issued = _with_gaps(selected[selected["label"] == label])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                line={"color": colors[label], "width": 1},
                customdata=issued,
                hovertemplate=f"{label}<br>issued %{{customdata}}<br>%{{x}}: %{{y:.4g}}<extra></extra>",
            )
        )
    data = pd.concat([data, _add_secondary(fig, secondary, observed.index[0], observed.index[-1])], ignore_index=True)
    fig.update_layout(title=title, template=TEMPLATE)
    return (fig, data) if return_data else fig


def select_forecasts(
    forecasts: pd.DataFrame,
    *,
    start: Any = None,
    end: Any = None,
    every: int | None = None,
    hours: int | Iterable[int] | None = None,
) -> pd.DataFrame:
    """Keep the forecasts of some issue times: in [start, end], at the given hours, then every n-th of those.

    E.g. ``hours=12`` keeps the midday runs, ``every=6`` every 6th issue time, ``hours=[0, 12], every=2`` the
    midday and midnight runs of every other day.
    """
    times = forecasts["issue_time"].drop_duplicates().sort_values()
    if start is not None:
        times = times[times >= pd.Timestamp(start)]
    if end is not None:
        times = times[times <= pd.Timestamp(end)]
    if hours is not None:
        times = times[times.dt.hour.isin(_as_list(hours))]
    if every is not None:
        times = times.iloc[::every]
    return forecasts[forecasts["issue_time"].isin(times)]


def _secondary_frame(secondary: pd.Series | pd.DataFrame | None) -> pd.DataFrame | None:
    if secondary is None:
        return None
    frame = secondary.to_frame() if isinstance(secondary, pd.Series) else secondary
    if not isinstance(frame.index, pd.DatetimeIndex):
        msg = "secondary needs a DatetimeIndex, e.g. columns of fa.load_data(runs)"
        raise TypeError(msg)
    return frame.sort_index()


def _new_figure(secondary: pd.DataFrame | None, rows: int = 1, **subplot_options: Any) -> go.Figure:
    """Create a figure with one column of subplots, each with a second y-axis if there are secondary series."""
    if secondary is None and rows == 1:
        return go.Figure()
    specs = [[{"secondary_y": secondary is not None}] for _ in range(rows)]
    return make_subplots(rows=rows, cols=1, specs=specs, **subplot_options)


def _add_secondary(
    fig: go.Figure, secondary: pd.DataFrame | None, start: Any, end: Any, *, row: int = 1, legend: bool = True
) -> pd.DataFrame:
    """Draw the secondary series between start and end dotted on the second y-axis, return them as plot data."""
    if secondary is None:
        return pd.DataFrame(columns=["label", "time", "value", "axis"])
    window = secondary.loc[start:end]
    for i, column in enumerate(window.columns):
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window[column],
                mode="lines",
                name=str(column),
                legendgroup=f"secondary {column}",
                showlegend=legend,
                line={"color": SECONDARY_COLORS[i % len(SECONDARY_COLORS)], "dash": "dot"},
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
    fig.update_yaxes(title_text=", ".join(map(str, window.columns)), secondary_y=True)
    long = window.rename_axis("time").reset_index().melt(id_vars="time", var_name="label", value_name="value")
    return long.assign(axis="secondary")


def _with_gaps(forecasts: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Concatenate the forecasts of one label into a single line with a gap after each issue time.

    One trace per label instead of one per forecast keeps plots with hundreds of forecasts fast.
    """
    gaps = forecasts.groupby("issue_time", as_index=False).agg(valid_time=("valid_time", "max"))
    gaps = gaps.assign(step=np.inf, prediction=np.nan)  # NaN breaks the line, inf sorts it last
    joined = pd.concat([forecasts[gaps.columns], gaps], ignore_index=True).sort_values(["issue_time", "step"])
    return joined["valid_time"], joined["prediction"], joined["issue_time"].dt.strftime("%Y-%m-%d %H:%M")


def _add_series(
    fig: go.Figure,
    data: pd.DataFrame,
    labels: list[str],
    quantiles: list[str],
    *,
    position: dict | None = None,
    legend: bool = True,
) -> None:
    """Observed line and one line per forecast label, with the outermost quantiles as a band."""
    position = position or {}
    observed = data[data["label"] == "observed"]
    fig.add_trace(
        go.Scatter(
            x=observed["time"],
            y=observed["value"],
            mode="lines",
            name="observed",
            legendgroup="observed",
            showlegend=legend,
            line={"color": OBSERVED_COLOR},
        ),
        **position,
    )
    colors = _colors(labels)
    for label in labels:
        series = data[data["label"] == label]
        if series.empty:
            continue
        if quantiles and series[quantiles].notna().any().any():
            lower, upper = series[quantiles[0]], series[quantiles[-1]]
            _add_band(fig, series["time"], lower, upper, label, colors[label], position)
        fig.add_trace(
            go.Scatter(
                x=series["time"],
                y=series["value"],
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=legend,
                line={"color": colors[label]},
            ),
            **position,
        )


def _add_band(
    fig: go.Figure, x: pd.Series, lower: pd.Series, upper: pd.Series, group: str, color: str, position: dict
) -> None:
    """Shaded area between lower and upper, toggled with the group's legend entry."""
    r, g, b = hex_to_rgb(color)
    common = {"x": x, "mode": "lines", "line": {"width": 0}, "legendgroup": group, "showlegend": False}
    fig.add_trace(go.Scatter(y=upper, hoverinfo="skip", **common), **position)
    fig.add_trace(
        go.Scatter(y=lower, fill="tonexty", fillcolor=f"rgba({r},{g},{b},{BAND_OPACITY})", hoverinfo="skip", **common),
        **position,
    )


def _colors(groups: list[str]) -> dict[str, str]:
    palette = qualitative.Plotly if len(groups) <= len(qualitative.Plotly) else qualitative.Alphabet
    return {group: palette[i % len(palette)] for i, group in enumerate(groups)}


def _labels(forecasts: pd.DataFrame) -> list[str]:
    """Labels present in the table, in predict's order (filtering keeps unused categories)."""
    present = set(forecasts["label"].unique())
    if isinstance(forecasts["label"].dtype, pd.CategoricalDtype):
        return [c for c in forecasts["label"].cat.categories if c in present]
    return list(dict.fromkeys(forecasts["label"]))


def _observed(forecasts: pd.DataFrame) -> pd.Series:
    """Return the observations in a predict table, indexed by time."""
    observed = forecasts.drop_duplicates("valid_time").set_index("valid_time")["observed"].sort_index()
    observed.index.name = "time"
    return observed


def _observed_rows(observed: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"label": "observed", "run": None, "time": observed.index, "value": observed.to_numpy()})


def _resolve_issue_time(forecasts: pd.DataFrame, issue_time: Any) -> pd.Timestamp:
    times = pd.DatetimeIndex(forecasts["issue_time"].drop_duplicates().sort_values())
    position = times.searchsorted(pd.Timestamp(issue_time), side="right") - 1
    if position < 0:
        msg = f"No forecast issued at or before {issue_time}, the first issue time is {times[0]}"
        raise KeyError(msg)
    return times[position]
