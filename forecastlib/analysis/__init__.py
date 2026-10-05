"""Analysis of trained runs with pandas and plotly.

Load metrics or forecasts into long DataFrames, filter them with pandas, then aggregate and plot::

    from forecastlib import analysis as fa

    runs = fa.load_runs("logs/experiment_1")
    lstm = runs[runs.model == "LSTM"]
    fa.summary(runs, ["nse", "kge"]).sort_values("nse")
    fa.plot_horizon(runs, ["nse", "mse"], split=["val", "test"], by="model", band="std")

    forecasts = fa.predict(lstm, start="2022-02-15", end="2022-03-01", label="num_layers")
    fa.plot_issue(forecasts, "2022-02-21 10:00")
    fa.plot_lead(forecasts, [1, 24, 48])
    fa.plot_forecasts(forecasts, hours=12)  # every midday forecast as its own line

    data = fa.load_data(lstm)  # the data file, indexed by time, for a second y-axis
    fa.plot_forecasts(forecasts, hours=12, secondary=data["yw_reinbek"])
"""

from forecastlib.analysis.forecasts import load_data, predict, quantile_columns
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
    "predict",
    "quantile_columns",
    "select_forecasts",
    "summary",
    "varying_hparams",
]
