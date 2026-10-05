import numpy as np
import pandas as pd
import pytest
from conftest import LABEL_LEN, PRED_LEN, SEQ_LEN, fit, make_model_args

from forecastlib import analysis as fa
from forecastlib.models.LightningWrapper import CustomLightningModule

SHIFT = 6
TIME_LEN = 4  # time features for embed=timeF, freq=h


@pytest.fixture(scope="module")
def trained(tmp_path_factory, covariate_df):
    """An LSTM with precipitation through shift_cols and a TFT with it through known_cols, on the same data.

    Both mix the channels, so the precipitation reaches the target's forecast (DLinear e.g. would not).
    """
    root = tmp_path_factory.mktemp("ensemble")
    covariate_df.to_csv(root / "covariates.csv", index=False)
    data_args = {
        "data": "custom",
        "embed": "timeF",
        "num_workers": 0,
        "batch_size": 16,
        "root_path": str(root),
        "data_path": "covariates.csv",
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "features": "MS",
        "target": "OT",
        "freq": "h",
        "augmentation_ratio": 0,
        "diff": False,
    }
    dm, args = make_model_args(data_args | {"shift_cols": {"precipitation": SHIFT}}, "LSTM")
    lstm = {
        "hidden_size": 16,
        "num_layers": 1,
        "bidirectional": False,
        "use_layernorm": False,
        "use_residual": False,
        "dropout": 0.0,
        "activation": "relu",
        "output_activation": None,
    }
    for k, v in lstm.items():
        setattr(args, k, v)
    fit(CustomLightningModule(args), dm, root / "logs" / "shift")

    dm, args = make_model_args(data_args | {"known_cols": ["precipitation"]}, "TemporalFusionTransformer")
    for k, v in {"d_model": 16, "n_heads": 2, "dropout": 0.1}.items():
        setattr(args, k, v)
    fit(CustomLightningModule(args), dm, root / "logs" / "known")
    return fa.load_history(root / "logs")


class PerIssueTime(fa.Analysis):
    """Builds the inputs per issue time, but without members: must give the deterministic forecasts."""

    def load_members(self, issue_time):  # noqa: ARG002
        return None


class Ensemble(fa.Analysis):
    """Two members: the observed precipitation (a perfect forecast) and no rain at all."""

    def load_members(self, issue_time):
        valid = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h")
        observed = self.observations["precipitation"].reindex(valid).to_numpy()
        return pd.DataFrame(
            {
                "member": ["observed"] * PRED_LEN + ["dry"] * PRED_LEN,
                "valid_time": list(valid) * 2,
                "precipitation": np.concatenate([observed, np.zeros(PRED_LEN)]),
            }
        )


@pytest.fixture(scope="module")
def issue_times(trained):
    """Issue times in the test split of both runs (shift_cols drops rows, so the splits differ)."""
    forecasts = fa.Analysis(runs=trained).predict()
    per_run = [set(f["issue_time"]) for _, f in forecasts.groupby("run", observed=True)]
    return pd.DatetimeIndex(sorted(set.intersection(*per_run)))[[0, 7, 30]]


def test_windows_reproduce_the_split_forecasts(trained, issue_times):
    """Building the inputs per issue time from the observations gives exactly the training preprocessing."""
    split = fa.Analysis(runs=trained).predict(start=issue_times[0], end=issue_times[-1])
    split = split[split["issue_time"].isin(issue_times)]
    windows = PerIssueTime(runs=trained).predict(issue_times=issue_times)

    assert "member" not in windows.columns  # no members, a deterministic table
    merged = windows.merge(split, on=["run", "issue_time", "step"], suffixes=("", "_split"))
    assert len(merged) == len(split) == 2 * len(issue_times) * PRED_LEN
    assert (merged["valid_time"] == merged["valid_time_split"]).all()
    assert np.allclose(merged["prediction"], merged["prediction_split"], atol=1e-5)
    assert np.allclose(merged["observed"], merged["observed_split"])


def test_ensemble_table(trained, issue_times):
    forecasts = Ensemble(runs=trained).predict(issue_times=issue_times, label="run")
    deterministic = PerIssueTime(runs=trained).predict(issue_times=issue_times, label="run")

    assert list(forecasts.columns[:3]) == ["run", "label", "member"]
    assert len(forecasts) == 2 * len(issue_times) * 2 * PRED_LEN  # runs x issue times x members x steps
    per_member = forecasts.pivot_table(
        index=["run", "issue_time", "step"], columns="member", values=["prediction", "observed"], observed=True
    )
    # The observations don't depend on the member.
    assert np.allclose(per_member[("observed", "observed")], per_member[("observed", "dry")])

    # The perfect-forecast member is the deterministic forecast, no rain changes the forecast of both models.
    reference = deterministic.set_index(["run", "issue_time", "step"])["prediction"]
    assert np.allclose(per_member[("prediction", "observed")], reference.loc[per_member.index], atol=1e-5)
    for run in ("shift", "known"):
        observed_member = per_member.loc[run, ("prediction", "observed")]
        dry_member = per_member.loc[run, ("prediction", "dry")]
        assert not np.allclose(observed_member, dry_member)


def test_member_inputs(trained, issue_times):
    analysis = Ensemble(runs=trained)
    issue = issue_times[1]
    members = analysis.load_members(issue)

    # shift_cols: the last SHIFT rows of seq_x hold the member's precipitation, the history is unchanged.
    settings = analysis.settings("shift")
    windows = dict(analysis.make_windows(settings, analysis.observations, issue, members))
    column = list(analysis.make_dataset(settings, analysis.observations, "test").data_x_raw.columns).index(
        "precipitation"
    )
    dry_scaled = (0 - settings.scaler.mean_[column]) / settings.scaler.scale_[column]
    assert np.allclose(windows["dry"][0][-SHIFT:, column], dry_scaled)
    assert np.allclose(windows["dry"][0][:-SHIFT], windows["observed"][0][:-SHIFT])

    # known_cols: the future marks hold the member's precipitation, the history marks are unchanged.
    settings = analysis.settings("known")
    windows = dict(analysis.make_windows(settings, analysis.observations, issue, members))
    known_scaler = settings.known_scaler
    dry_scaled = (0 - known_scaler.mean_[0]) / known_scaler.scale_[0]
    assert np.allclose(windows["dry"][3][-PRED_LEN:, TIME_LEN], dry_scaled)
    assert np.allclose(windows["dry"][2], windows["observed"][2])


def test_ensemble_plots(trained, issue_times):
    analysis = Ensemble(runs=trained).filter("run == 'shift'")
    forecasts = analysis.predict(issue_times=issue_times)

    fig = analysis.plot_forecasts(forecasts)
    lines = [t for t in fig.data if t.name == "shift" and t.showlegend is False]
    assert len(lines) == len(issue_times)  # one trace per issue time ...
    for line in lines:  # ... holding both members, separated by a gap, in the issue time's color
        assert np.isnan(np.asarray(line.y, dtype=float)).sum() == 2
    assert len({t.line.color for t in lines}) == len(issue_times)

    by_label = analysis.plot_forecasts(forecasts, color_by="label")
    line = next(t for t in by_label.data if t.name == "shift")
    assert np.isnan(np.asarray(line.y, dtype=float)).sum() == len(issue_times) * 2  # a line per issue time and member

    fig, data = analysis.plot_issue(forecasts, issue_times[1], secondary="precipitation", return_data=True)
    member_lines = [t for t in fig.data if t.name == "shift"]
    assert len(member_lines) == 2
    assert [t.showlegend for t in member_lines] == [True, False]  # one legend entry per run
    assert set(data["member"].dropna()) == {"observed", "dry"}

    fig = analysis.plot_lead(forecasts, [1, PRED_LEN])
    assert len([t for t in fig.data if t.name == "shift"]) == 2 * 2  # members x steps


def test_ensemble_needs_a_period(trained):
    with pytest.raises(ValueError, match="start and end"):
        Ensemble(runs=trained).predict()


def test_unknown_member_column_raises(trained, issue_times):
    class Broken(fa.Analysis):
        def load_members(self, issue_time):
            return pd.DataFrame({"member": [0], "valid_time": [issue_time + pd.Timedelta(hours=1)], "rain": [1.0]})

    with pytest.raises(KeyError, match="rain"):
        Broken(runs=trained).predict(issue_times=issue_times[:1])
