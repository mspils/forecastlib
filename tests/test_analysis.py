import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import PRED_LEN, SEQ_LEN, fit, make_model_args
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import default_collate

from forecastlib import analysis as fa
from forecastlib.data_provider.data_module import CustomDataModule
from forecastlib.models.LightningWrapper import CustomLightningModule

STEPS = 4
BASE = {"A": 1.0, "B": 2.0}  # metric offset per model
SLOPE = {1e-3: 0.1, 1e-4: 0.3}  # metric increase per forecast step per learning rate
RUN_IDS = [f"{m}/lightning_logs/version_{v}" for m in BASE for v in (0, 1)]


def expected_mse(model, lr, step):
    return BASE[model] + SLOPE[lr] * step


def write_runs(root, logger_class):
    """Four runs like the training scripts produce them: <model>/lightning_logs/version_<i>."""
    for model in BASE:
        for version, lr in enumerate(SLOPE):
            logger = logger_class(root / model, name="lightning_logs", version=version)
            hparams = {
                "model": model,
                "learning_rate": lr,
                "pred_len": STEPS,
                "target_idx": [7],
                "scaler": pickle.dumps(StandardScaler()),
            }
            if model == "A":
                hparams["num_layers"] = 2  # a hyperparameter only one model has
            logger.log_hyperparams(hparams)
            for epoch, step in enumerate((10, 20, 30)):  # training curves on global steps
                logger.log_metrics({"train_loss": 1.0 / step, "epoch": epoch}, step=step)
                logger.log_metrics({"val_loss": 2.0 / step}, step=step)
            for step in range(1, STEPS + 1):  # per-horizon metrics, like the metric callbacks
                mse = expected_mse(model, lr, step)
                logger.log_metrics({"val_mse": mse / 2, "test_mse": mse, "test_nse": -mse}, step=step)
            logger.finalize("success")


@pytest.fixture(scope="module", params=[TensorBoardLogger, CSVLogger], ids=["tensorboard", "csv"])
def log_dir(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("logs")
    write_runs(root, request.param)
    return root


@pytest.fixture(scope="module")
def runs(log_dir):
    return fa.load_runs(log_dir)


def test_load_runs(runs, log_dir):
    assert list(runs["run"].cat.categories) == RUN_IDS
    assert set(runs["path"]) == {log_dir / run for run in RUN_IDS}
    assert "scaler" not in runs.columns  # pickled
    assert set(runs["target_idx"]) == {(7,)}
    assert sorted(fa.varying_hparams(runs)) == ["learning_rate", "model", "num_layers"]
    assert set(fa.hparam_columns(runs)) == {"model", "learning_rate", "pred_len", "target_idx", "num_layers"}

    assert set(runs["split"]) == {"val", "test"}
    assert set(runs["metric"]) == {"mse", "nse"}
    assert sorted(runs["step"].unique()) == list(range(1, STEPS + 1))
    a0 = runs[(runs["run"] == RUN_IDS[0]) & (runs["split"] == "test") & (runs["metric"] == "mse")]
    assert np.allclose(a0.sort_values("step")["value"], [expected_mse("A", 1e-3, s) for s in range(1, STEPS + 1)])


def test_load_history(log_dir):
    history = fa.load_history(log_dir)
    assert {"train_loss", "val_loss", "epoch"} <= set(history["tag"])
    assert not history["tag"].str.contains("mse").any()
    val_loss = history[(history["run"] == RUN_IDS[0]) & (history["tag"] == "val_loss")]
    assert list(val_loss["step"]) == [10, 20, 30]
    assert set(history["model"]) == {"A", "B"}


def test_tensorboard_and_csv_agree(tmp_path):
    loaded = {}
    for logger_class in (TensorBoardLogger, CSVLogger):
        root = tmp_path / logger_class.__name__
        write_runs(root, logger_class)
        runs = fa.load_runs(root).drop(columns="path")
        runs["run"] = runs["run"].astype(str)
        loaded[logger_class] = runs.sort_values(["run", "split", "metric", "step"]).reset_index(drop=True)
    tb, csv = loaded[TensorBoardLogger], loaded[CSVLogger]
    pd.testing.assert_frame_equal(tb, csv[tb.columns], check_dtype=False)


def test_filtering_is_plain_pandas(runs):
    lstm_like = runs[runs["model"] == "A"]
    assert set(lstm_like["run"]) == set(RUN_IDS[:2])
    assert fa.varying_hparams(lstm_like) == ["learning_rate"]
    assert set(runs.query("learning_rate < 5e-4")["learning_rate"]) == {1e-4}


def test_summary(runs):
    table = fa.summary(runs, ["mse", "nse"])
    run = RUN_IDS[3]  # B, lr 1e-4
    steps = range(1, STEPS + 1)
    assert table.loc[run, "mse"] == pytest.approx(np.mean([expected_mse("B", 1e-4, s) for s in steps]))
    assert table.loc[run, "nse"] == pytest.approx(-table.loc[run, "mse"])
    assert list(table.columns[-2:]) == ["mse", "nse"]
    assert len(table) == 4

    at_step = fa.summary(runs[runs["model"] == "B"], "mse", split="val", step=STEPS, hparams=["model"])
    assert list(at_step.index) == RUN_IDS[2:]
    assert at_step.loc[run, "mse"] == pytest.approx(expected_mse("B", 1e-4, STEPS) / 2)


def test_aggregate_by_hparam(runs):
    data = fa.aggregate(runs, "mse", by="model", band="std")
    row = data[(data["model"] == "A") & (data["step"] == 2)].iloc[0]
    values = [expected_mse("A", lr, 2) for lr in SLOPE]

    assert row["value"] == pytest.approx(np.mean(values))
    assert row["upper"] - row["value"] == pytest.approx(np.std(values, ddof=1))
    assert row["n_runs"] == 2
    assert row["group"] == "A"

    minmax = fa.aggregate(runs, "mse", by=["model", "learning_rate"], band="minmax", agg="median")
    assert set(minmax["n_runs"]) == {1}
    assert set(minmax["group"]) == {"A / 0.001", "A / 0.0001", "B / 0.001", "B / 0.0001"}

    by_missing = fa.aggregate(runs, "mse", by="num_layers")
    assert set(by_missing["group"]) == {"2", "n/a"}


def test_aggregate_without_by_keeps_runs(runs):
    data = fa.aggregate(runs, ["mse", "nse"], split=["val", "test"], by=None)
    assert set(data["group"]) == set(RUN_IDS)
    assert len(data) == len(runs)


def test_aggregate_errors(runs):
    with pytest.raises(KeyError, match="rmse"):
        fa.aggregate(runs, "rmse")
    with pytest.raises(KeyError, match="depth"):
        fa.aggregate(runs, "mse", by="depth")
    with pytest.raises(ValueError, match="band"):
        fa.aggregate(runs, "mse", band="iqr")


def test_plot_horizon(runs):
    fig, data = fa.plot_horizon(runs, ["mse", "nse"], split=["val", "test"], by="model", return_data=True)
    pd.testing.assert_frame_equal(data, fa.aggregate(runs, ["mse", "nse"], split=["val", "test"], by="model"))

    lines = [t for t in fig.data if t.line.width != 0]
    bands = [t for t in fig.data if t.fill == "tonexty"]
    # 2 groups x 3 subplots (2 splits x 2 metrics, but no val nse), one band per line, one legend entry per group
    assert len(lines) == len(bands) == 2 * 3
    assert sorted(t.name for t in fig.data if t.showlegend is not False) == ["A", "B"]
    assert {t.xaxis for t in fig.data} == {"x", "x3", "x4"}

    no_band = fa.plot_horizon(runs, "mse", by=None, band=None)
    assert len(no_band.data) == 4  # a line per run, no bands


def test_plot_hparam(runs):
    fig, data = fa.plot_hparam(runs, "mse", "learning_rate", color="model", log_x=True, return_data=True)
    assert len(data) == 4
    assert sum(len(t.x) for t in fig.data) == 4
    assert fig.layout.xaxis.type == "log"


@pytest.fixture(scope="module")
def trained_runs(tmp_path_factory):
    """Two small DLinear runs on the same data, trained like the training scripts do."""
    root = tmp_path_factory.mktemp("trained")
    rng = np.random.default_rng(2)
    n = 400
    pd.DataFrame(
        {
            "date": pd.date_range("2021", periods=n, freq="h"),
            "a": rng.random(n).cumsum(),
            "OT": np.sin(np.arange(n) / 10) + rng.random(n) * 0.1,
        }
    ).to_csv(root / "data.csv", index=False)
    data_args = {
        "data": "custom",
        "embed": "timeF",
        "num_workers": 0,
        "batch_size": 16,
        "root_path": str(root),
        "data_path": "data.csv",
        "seq_len": SEQ_LEN,
        "label_len": 12,
        "pred_len": PRED_LEN,
        "features": "MS",
        "target": "OT",
        "freq": "h",
        "augmentation_ratio": 0,
        "diff": False,
    }
    for moving_avg in (5, 9):
        dm, args = make_model_args(data_args, "DLinear")
        args.moving_avg = moving_avg
        fit(CustomLightningModule(args), dm, root / "logs" / f"dlinear_{moving_avg}")
    # The runs only log training curves, so their hyperparameters come from load_history.
    return fa.load_history(root / "logs")


@pytest.fixture(scope="module")
def analysis(trained_runs):
    return fa.Analysis(runs=trained_runs)


@pytest.fixture(scope="module")
def forecasts(analysis):
    return analysis.predict(label="moving_avg")


def test_predict_matches_predict_step(trained_runs, forecasts):
    assert list(forecasts.columns) == [c for c in fa.forecasts.FORECAST_COLUMNS if c != "member"]
    assert list(forecasts["label"].cat.categories) == ["5", "9"]

    run = "dlinear_9"
    path = trained_runs.loc[trained_runs["run"] == run, "path"].iloc[0]
    module = CustomLightningModule.from_disk(path, device="cpu")
    module.eval()
    dataset = CustomDataModule.from_disk(path).test_set
    for sample in (0, 17):
        issue = dataset.dates[sample + SEQ_LEN - 1]
        ours = forecasts[(forecasts["run"] == run) & (forecasts["issue_time"] == issue)]
        with torch.no_grad():
            _, pred, true = module.predict_step(default_collate([dataset[sample]]), 0)

        assert list(ours["step"]) == list(range(1, PRED_LEN + 1))
        assert list(ours["valid_time"]) == list(dataset.dates[sample + SEQ_LEN : sample + SEQ_LEN + PRED_LEN])
        assert np.allclose(ours["prediction"], pred[0, :, 0].numpy(), atol=1e-5)
        # The truth predict_step trains against is the observation at each valid time, so the times line up.
        assert np.allclose(ours["observed"], true[0, :, 0].numpy(), atol=1e-4)


def test_predict_period(analysis, forecasts):
    times = forecasts["issue_time"].drop_duplicates().sort_values().to_list()
    part = analysis.predict(start=times[5], end=times[9])

    assert part["issue_time"].drop_duplicates().to_list() == times[5:10]
    merged = part.merge(forecasts, on=["run", "issue_time", "step"], suffixes=("", "_full"))
    assert len(merged) == len(part)
    assert np.allclose(merged["prediction"], merged["prediction_full"], atol=1e-5)

    every = analysis.predict(start=times[5], end=times[20], every=5, label="moving_avg")
    assert every["issue_time"].drop_duplicates().to_list() == times[5:21:5]

    with pytest.raises(ValueError, match="No issue times"):
        analysis.predict(start="2100-01-01")


def test_analysis_filter_and_metrics(analysis):
    lstm_like = analysis.filter("moving_avg == 9")
    assert set(lstm_like.runs["run"]) == {"dlinear_9"}
    assert len(analysis.runs["run"].unique()) == 2  # the original is unchanged
    assert set(analysis.filter(analysis.runs["moving_avg"] == 5).runs["run"]) == {"dlinear_5"}
    assert set(analysis.filter(lambda runs: runs["moving_avg"] > 6).runs["run"]) == {"dlinear_9"}


def test_analysis_secondary_by_name(analysis, forecasts):
    fig, data = analysis.plot_forecasts(forecasts, every=10, secondary="a", return_data=True)
    assert "a" in {t.name for t in fig.data}
    assert set(data.loc[data["axis"] == "secondary", "label"]) == {"a"}


def test_plot_issue(forecasts):
    times = forecasts["issue_time"].drop_duplicates().sort_values().to_list()
    issue = times[30]
    fig, data = fa.plot_issue(forecasts, issue + pd.Timedelta(minutes=30), history=6, return_data=True)

    assert set(data["label"]) == {"observed", "5", "9"}
    observed = data[data["label"] == "observed"]
    assert observed["time"].min() == issue - pd.Timedelta(hours=5)
    assert observed["time"].max() == issue + pd.Timedelta(hours=PRED_LEN)
    assert (data[data["label"] == "5"]["time"] > issue).all()
    assert [t.name for t in fig.data] == ["observed", "5", "9"]

    one_model = fa.plot_issue(forecasts[forecasts["label"] == "9"], issue)
    assert [t.name for t in one_model.data] == ["observed", "9"]

    with pytest.raises(KeyError, match="No forecast"):
        fa.plot_issue(forecasts, times[0] - pd.Timedelta(hours=1))


def test_plot_lead(forecasts):
    times = forecasts["issue_time"].drop_duplicates().sort_values().to_list()
    start, end = times[20], times[40]
    fig, data = fa.plot_lead(forecasts, [1, PRED_LEN], start=start, end=end, return_data=True)

    assert set(data["step"]) == {1, PRED_LEN}
    assert data["time"].between(start, end).all()
    lead = data[(data["step"] == PRED_LEN) & (data["label"] == "5")]
    expected = forecasts[(forecasts["step"] == PRED_LEN) & (forecasts["label"] == "5")]
    expected = expected[expected["valid_time"].between(start, end)]
    assert np.allclose(lead["value"], expected["prediction"])
    assert len(fig.data) == 2 * 3  # observed + 2 models per subplot

    with pytest.raises(KeyError, match="step"):
        fa.plot_lead(forecasts, PRED_LEN + 1)


def test_select_forecasts(forecasts):
    times = forecasts["issue_time"].drop_duplicates().sort_values()

    def issue_times(**selection):
        return fa.select_forecasts(forecasts, **selection)["issue_time"].drop_duplicates().sort_values().to_list()

    assert issue_times(every=5) == times.iloc[::5].to_list()
    assert issue_times(hours=12) == [t for t in times if t.hour == 12]
    assert issue_times(hours=[0, 12], every=2) == [t for t in times if t.hour in {0, 12}][::2]
    assert issue_times(start=times.iloc[10], end=times.iloc[20], every=3) == times.iloc[10:21:3].to_list()
    # Whole forecasts are kept, all steps of every run.
    selected = fa.select_forecasts(forecasts, hours=12)
    assert (selected.groupby(["run", "issue_time"], observed=True).size() == PRED_LEN).all()


def test_plot_forecasts(forecasts):
    fig, data = fa.plot_forecasts(forecasts, hours=[0, 12], return_data=True)
    issued = [t for t in forecasts["issue_time"].drop_duplicates() if t.hour in {0, 12}]

    assert [t.name for t in fig.data] == ["observed", "5", "9"]  # one trace per model, not per forecast
    model_5 = fig.data[1]
    # One gap (NaN) after each forecast, so every forecast is its own line.
    assert np.isnan(np.asarray(model_5.y, dtype=float)).sum() == len(issued)
    assert len(model_5.x) == len(issued) * (PRED_LEN + 1)
    assert set(data.loc[data["label"] == "5", "issue_time"]) == set(issued)

    # The observations stay continuous from the first selected issue time to the last forecast step.
    observed = data[data["label"] == "observed"]["time"]
    assert observed.min() == min(issued)
    assert observed.max() == max(issued) + pd.Timedelta(hours=PRED_LEN)
    assert observed.diff().dropna().eq(pd.Timedelta(hours=1)).all()

    with pytest.raises(ValueError, match="No forecasts"):
        fa.plot_forecasts(forecasts, hours=25)


def test_load_data(trained_runs):
    data = fa.load_data(trained_runs)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert list(data.columns) == ["a", "OT"]

    unshifted = fa.load_data(trained_runs, unshift={"a": 3})
    assert np.allclose(unshifted["a"].iloc[3:], data["a"].iloc[:-3])
    assert unshifted["a"].iloc[:3].isna().all()
    assert unshifted["OT"].equals(data["OT"])

    with pytest.raises(KeyError, match="rain"):
        fa.load_data(trained_runs, unshift={"rain": 1})


@pytest.mark.parametrize("plot", ["issue", "lead", "forecasts"])
def test_secondary_axis(forecasts, trained_runs, plot):
    secondary = fa.load_data(trained_runs)["a"]
    times = forecasts["issue_time"].drop_duplicates().sort_values().to_list()
    start, end = times[20], times[40]
    if plot == "issue":
        fig, data = fa.plot_issue(forecasts, start, secondary=secondary, return_data=True)
    elif plot == "lead":
        fig, data = fa.plot_lead(forecasts, [1, 2], start=start, end=end, secondary=secondary, return_data=True)
    else:
        fig, data = fa.plot_forecasts(forecasts, start=start, end=end, every=5, secondary=secondary, return_data=True)

    on_secondary = [t for t in fig.data if t.name == "a"]
    assert on_secondary
    assert all(fig.layout[t.yaxis.replace("y", "yaxis")].overlaying for t in on_secondary)  # a second y-axis
    assert set(data["axis"]) == {"primary", "secondary"}

    rows = data[data["axis"] == "secondary"]
    primary_times = data.loc[data["axis"] == "primary", "time"]
    assert rows["time"].min() == primary_times.min()
    assert rows["time"].max() == primary_times.max()
    assert np.allclose(rows["value"], secondary.loc[rows["time"]])


def test_secondary_needs_time_index(forecasts):
    with pytest.raises(TypeError, match="DatetimeIndex"):
        fa.plot_forecasts(forecasts, secondary=pd.Series([1.0, 2.0]))
