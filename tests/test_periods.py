import pandas as pd
import torch
from conftest import PRED_LEN, SEQ_LEN, fit, make_model_args

from forecastlib.data_provider.data_loader import Dataset_MW
from forecastlib.data_provider.data_module import CustomDataModule
from forecastlib.models.LightningWrapper import CustomLightningModule
from forecastlib.utils.tools import load_yaml


def test_periods_cover_the_targets_of_each_split(data_args, csv_dir, tmp_path):
    dates = pd.to_datetime(pd.read_csv(csv_dir / "x.csv")["date"])
    dm = CustomDataModule(data_args)
    periods = {split: [pd.Timestamp(t) for t in span] for split, span in dm.target_periods()[0].items()}
    assert list(periods) == ["train", "val", "test"]
    assert periods["train"][0] == dates[SEQ_LEN]  # the first target after one input window
    assert periods["test"][1] == dates.iloc[-1]  # the last row of the file
    step = pd.Timedelta(hours=1)
    assert periods["val"][0] == periods["train"][1] + step  # splits join without gap or overlap
    assert periods["test"][0] == periods["val"][1] + step
    for split, dataset in (("train", dm.train_set), ("val", dm.val_set), ("test", dm.test_set)):
        n_targets = int((periods[split][1] - periods[split][0]) / step) + 1
        assert n_targets == len(dataset) + PRED_LEN - 1  # every target row of the split, once

    _, args = make_model_args(data_args, "DLinear")
    fit(CustomLightningModule(args), dm, tmp_path)
    stored = load_yaml((tmp_path / "hparams.yaml").read_text())
    assert stored["periods"] == dm.target_periods()[0]
    assert "station_periods" not in stored


def test_station_periods_of_a_multi_station_dataset():
    dataset = Dataset_MW.__new__(Dataset_MW)  # the period logic only; reading MW data needs the GEMS files
    dataset.seq_len, dataset.pred_len = 3, 2
    dataset.mw_list = ["MW_1", "MW_2", "MW_3"]
    dataset.well_data = {
        "MW_1": {"data_x_raw": pd.DataFrame(index=pd.date_range("2008-01-01", periods=10, freq="D"))},
        "MW_2": {"data_x_raw": pd.DataFrame(index=pd.date_range("2009-06-01", periods=6, freq="D"))},
        "MW_3": {"data_x_raw": pd.DataFrame(index=pd.date_range("2010-01-01", periods=4, freq="D"))},  # too short
    }
    stations = dataset.station_target_periods()
    assert stations == {
        "MW_1": (pd.Timestamp("2008-01-04"), pd.Timestamp("2008-01-10")),
        "MW_2": (pd.Timestamp("2009-06-04"), pd.Timestamp("2009-06-06")),
    }
    assert dataset.target_period() == (pd.Timestamp("2008-01-04"), pd.Timestamp("2009-06-06"))

    class Module:  # CustomDataModule.target_periods with this dataset in every split
        train_set = val_set = test_set = dataset

    periods, per_station = CustomDataModule.target_periods(Module())
    assert periods["train"] == ["2008-01-04 00:00", "2009-06-06 00:00"]
    assert per_station["MW_2"]["val"] == ["2009-06-04 00:00", "2009-06-06 00:00"]


def test_datasets_without_dates_are_left_out():
    class NoDates(torch.utils.data.Dataset):
        def __len__(self):
            return 5

    class Module:
        train_set = val_set = test_set = NoDates()

    assert CustomDataModule.target_periods(Module()) == ({}, {})
