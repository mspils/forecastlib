"""Analysis of trained runs with pandas and plotly.

``Analysis`` holds the runs of an experiment and loads models and data through hooks (see
forecastlib.analysis.analysis for subclassing, e.g. other file formats or ensemble forecasts)::

    from forecastlib.analysis import Analysis

    analysis = Analysis("logs/experiment_1")
    analysis.runs                                  # metrics + hyperparameters, a plain long DataFrame
    analysis.summary(["nse", "kge"]).sort_values("nse")
    analysis.plot_horizon(["nse", "mse"], split=["val", "test"], by="model", band="std")

    lstm = analysis.filter("model == 'LSTM'")
    forecasts = lstm.predict(start="2022-02-15", end="2022-03-01", label="num_layers")
    lstm.plot_issue(forecasts, "2022-02-21 10:00")
    lstm.plot_lead(forecasts, [1, 24, 48])
    lstm.plot_forecasts(forecasts, hours=12, secondary="yw_reinbek")  # every midday forecast, a 2nd y-axis

The functions behind the methods work on the DataFrames directly, e.g. ``plot_horizon(runs, ...)``.
"""

from forecastlib.analysis.analysis import Analysis
from forecastlib.analysis.forecasts import load_data, quantile_columns, select_times
from forecastlib.analysis.plots import (
    plot_forecasts,
    plot_horizon,
    plot_hparam,
    plot_issue,
    plot_lead,
    select_forecasts,
)
from forecastlib.analysis.runs import (
    aggregate,
    hparam_columns,
    load_history,
    load_runs,
    summary,
    varying_hparams,
)

__all__ = [
    "Analysis",
    "aggregate",
    "hparam_columns",
    "load_data",
    "load_history",
    "load_runs",
    "plot_forecasts",
    "plot_horizon",
    "plot_hparam",
    "plot_issue",
    "plot_lead",
    "quantile_columns",
    "select_forecasts",
    "select_times",
    "summary",
    "varying_hparams",
]
