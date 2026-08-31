"""Data loading utilities for time series forecasting datasets.

This module provides various PyTorch Dataset implementations for loading and
preprocessing time series data from different sources including:
- Electricity Transformer Temperature (ETT) datasets
- M4 competition data
- UEA time series classification datasets
- Custom datasets compatible with forecasting models

Supports multiple data splits (train/val/test), feature modes (univariate/multivariate),
scaling, time encoding, and data augmentation.
"""

import glob
import os
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from forecastlib.data_provider.m4 import M4Dataset, M4Meta
from forecastlib.data_provider.uea import Normalizer, interpolate_missing, subsample
from forecastlib.utils.augmentation import run_augmentation_single
from forecastlib.utils.timefeatures import FREQ_MAP, get_data_stamp, time_features, time_features_from_frequency_str

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):  # noqa: N801
    """PyTorch Dataset for Electricity Transformer Temperature (ETT) hourly data.

    Loads hourly ETT dataset with configurable train/val/test splits, feature modes,
    and preprocessing options. Supports univariate (single target) and multivariate
    (all features) forecasting, with optional scaling and augmentation.

    Args:
        args: Configuration namespace with attributes like augmentation_ratio.
        root_path: Directory containing the data CSV file.
        flag: Data split - "train", "val", or "test". Defaults to "train".
        size: Tuple of (seq_len, label_len, pred_len). Defaults to (96, 48, 96).
        features: Feature mode - "S" (univariate), "M" (multivariate all), or
            "MS" (multivariate with target). Defaults to "S".
        data_path: CSV filename. Defaults to "ETTh1.csv".
        target: Target column name for univariate mode. Defaults to "OT".
        scale: Whether to normalize data with StandardScaler. Defaults to True.
        timeenc: Time encoding mode (0=discrete, 1=continuous). Defaults to 0.
        freq: Frequency string for time features. Defaults to "h" (hourly).
        seasonal_patterns: Optional seasonal patterns info (unused). Defaults to None.

    """

    def __init__(
        self,
        args,
        root_path: str,
        flag: str = "train",
        size: tuple[int, int, int] | None = None,
        features: str = "S",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        seasonal_patterns: Any | None = None,
    ) -> None:
        """Initialize ETT hourly dataset.

        Args:
            args: Configuration object with augmentation parameters.
            root_path: Base path to data directory.
            flag: Dataset split ("train", "val", or "test").
            size: Window sizes as (input_len, label_len, forecast_len).
            features: Feature selection mode.
            data_path: Path to data CSV relative to root_path.
            target: Target column name.
            scale: Enable data normalization.
            timeenc: Time encoding method.
            freq: Time series frequency.
            seasonal_patterns: Optional metadata about seasonality.

        """
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self) -> None:
        """Load and preprocess ETT data from CSV file.

        Performs the following steps:
        1. Read CSV file and select features
        2. Split data into train/val/test based on configuration
        3. Fit scaler on training data if scaling enabled
        4. Generate time encodings for all timestamps
        5. Apply data augmentation if in training mode

        Returns:
            None (modifies self.data_x, self.data_y, self.data_stamp)

        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        data_stamp = get_data_stamp(df_stamp, self.timeenc, self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a single batch sample.

        Returns a window of size (seq_len, label_len, pred_len) containing:
        - Input sequence (seq_len timesteps)
        - Target sequence (label_len + pred_len timesteps)
        - Time encodings for input and target sequences

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (seq_x, seq_y, seq_x_mark, seq_y_mark) where:
                - seq_x: Input features of shape (seq_len, num_features)
                - seq_y: Target values of shape (label_len + pred_len, num_features)
                - seq_x_mark: Time encodings for input
                - seq_y_mark: Time encodings for target

        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            Number of non-overlapping windows that can be created.

        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.

        Args:
            data: Scaled data array.

        Returns:
            Original scale data.

        """
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        seasonal_patterns=None,
    ) -> None:
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ) -> None:
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date", *cols, self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",
        seasonal_patterns="Yearly",
    ) -> None:
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == "train":
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in dataset.values[dataset.groups == self.seasonal_patterns]]
        )  # split different frequencies
        self.ids = np.array(list(dataset.ids[dataset.groups == self.seasonal_patterns]))
        self.timeseries = list(training_values)

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit), high=len(sampled_timeseries), size=1
        )[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len) : cut_point]
        insample[-len(insample_window) :, 0] = insample_window
        insample_mask[-len(insample_window) :, 0] = 1.0
        outsample_window = sampled_timeseries[
            max(0, cut_point - self.label_len) : min(len(sampled_timeseries), cut_point + self.pred_len)
        ]
        outsample[: len(outsample_window), 0] = outsample_window
        outsample_mask[: len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self) -> int:
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len :]
            insample[i, -len(ts) :] = ts_last_window
            insample_mask[i, -len(ts) :] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train") -> None:
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, "train.csv"))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = pd.read_csv(os.path.join(root_path, "test_label.csv")).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self) -> int:
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        if self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        )


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train") -> None:
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self) -> int:
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        if self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        )


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train") -> None:
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self) -> int:

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        if self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        )


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train") -> None:
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self) -> int:
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        if self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        )


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train") -> None:
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "swat_train2.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "swat2.csv"))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self) -> int:
        """Number of images in the object dataset."""
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        if self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(self.test_labels[0 : self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size : index // self.step * self.win_size + self.win_size]
        )





class Dataset_Diff(Dataset):
    _supports_multicol = True  # default
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=(144, 48, 48),
        features="MS",
        data_path="example.csv",
        target="OT",
        timeenc=0,
        freq="h",
    ) -> None:
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            # TODO nicht size nehmen sonder args.seq_len etc.?
            self.seq_len = 144
            self.label_len = 48  # TODO???
            self.pred_len = 48
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.date_col = args.date_col
        self.diff = args.diff

        self.is_npy = self.data_path.endswith(".npy")
        if self.is_npy:
            self.stride = 100
            self.num_time_features = (
                FREQ_MAP[args.freq] if self.args.embed == "timeF" else len(time_features_from_frequency_str(args.freq))
            )

        else:
            self.stride = 1

        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.is_npy:
            data = np.load(os.path.join(self.root_path, self.data_path))
            df_raw = pd.DataFrame(data)
        elif self.data_path.endswith(".csv"):
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=[self.date_col])

        # fill missing values #TODO this is only for the LFU
        # max_fill = 200
        # na_count = df_raw.isna().sum(axis=0)
        # mask = (
        #     df_raw.columns.str.contains("NEW")
        #     | df_raw.columns.str.contains("NVh")
        #     | (df_raw.columns.str.startswith("N") & df_raw.columns.str.endswith("_mm"))
        # )

        # prec_cols = list(na_count[mask][na_count[mask] > 0].index)
        # if len(prec_cols) > 0:
        #     df_raw.loc[:, mask] = df_raw.loc[:, mask].fillna(0)

        # # interpolate data in all other columns
        # df_raw = df_raw.interpolate(limit=max_fill, limit_direction="both")

        # if df_raw.isna().sum().sum() > 0:
        #     msg = (
        #         f"Some columns were missing more than {max_fill} continuous values, either raise the limit or fill values manually."
        #         f"{df_raw.isna().sum().sum()} still missing,"
        #     )
        #     raise ValueError(msg)


        #df_raw.columns: ['date', ...(other features), target feature]
        if not self.is_npy:
            df_raw.rename(columns={self.date_col: "date"}, inplace=True)

        try:
            if self.args.borders_start is None:
                num_train = int(len(df_raw) * 0.7)
                num_test = int(len(df_raw) * 0.2)
                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
            else:
                # TODO assert compat boarders_start + dataset
                assert self.args.data_path in ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"], (
                    "This option should only be used in combination with the ETT datasets"
                )
                border1s = [
                    self.args.borders_start[0],
                    self.args.borders_start[1] - self.seq_len,
                    self.args.borders_start[2] - self.seq_len,
                ]
                border2s = self.args.borders_end
        except (KeyError,AttributeError):
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.args.diff:
            if self.features == "M":  # TODO other options
                cols = list(df_raw.columns)
                if not self.is_npy:
                    cols.remove("date")
            elif self.features in {"MS", "S"}:
                cols = [self.target]

            df_diff = df_raw[cols].diff()
            df_diff.iloc[0] = 0
            df_diff.columns = [str(x) + "_d1" for x in df_diff.columns]
            df_raw = pd.concat([df_raw, df_diff], axis=1)

        if not self.is_npy:
            cols = list(df_raw.columns)
            cols.remove("date")
            df_raw = df_raw[["date", *cols]]

        if self.features in {"M", "MS"}:
            if not self.is_npy:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            else:
                df_data = df_raw
        elif self.features == "S" and self.args.diff:
            df_data = df_raw[[self.target, f"{self.target}_d1"]]
        else:
            df_data = df_raw[[self.target]]

        self.data_x_raw = df_data[border1:border2]
        train_data = df_data[border1s[0] : border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        if not self.is_npy:
            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            self.index = df_stamp["date"].copy()
            data_stamp = get_data_stamp(df_stamp, self.timeenc, self.freq)
            self.data_stamp = torch.tensor(data_stamp, dtype=torch.float32)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        # TODO Alternativ könnte man auch im trainings loop auf float runter gehen. Oder das ganze in das data_module verschieben.
        self.data_x = torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y = torch.tensor(self.data_y, dtype=torch.float32)

        self.width = self.data_x.shape[1]

    def __getitem__(self, index):

        if index > len(self):
            msg = f"Index {index} out of range, max index is {len(self)}"
            raise IndexError(msg)

        s_begin = index
        if self.is_npy:
            #s_begin = index % n_timepoint  # select start timestamp
            s_begin = self.stride * s_begin

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if not self.is_npy:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        else:

            seq_x_mark = torch.zeros((seq_x.shape[0], self.num_time_features))
            seq_y_mark = torch.zeros((seq_x.shape[0], self.num_time_features))


        if self.diff:
            if self.features == "M":
                seq_x[0, self.width // 2 :] = 0
            elif self.features in {"MS", "S"}:
                seq_x[0, -1] = 0

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        if self.is_npy:
            return (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_pred_index(self):
        return self.index[self.seq_len + -1 : -self.pred_len]


class Dataset_MW(Dataset):
    def __init__(
        self,
        args,
        root_path,
        data_path,
        flag="train",
        size=(144, 48, 48),
        features="MS",
        target="GWL",
        timeenc=0,
        freq="D",
    ) -> None:
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 144
            self.label_len = 48
            self.pred_len = 48
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.train_end = "2007-12-31"
        self.val_end = "2012-12-31"

        if self.set_type == 0:
            self.dynamic_scaler = StandardScaler()
        else:
            self.dynamic_scaler = args.dynamic_scaler

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path  # folder path
        self.static_path = os.path.join(root_path, "static/static_features_MW_1toMW_3207.csv")

        self.diff = getattr(args, "diff", False)
        self.__read_data__()

        self.scaler = StandardScaler()
        self.scaler.mean_ = np.concat([self.dynamic_scaler.mean_, self.scaler_static.mean_])
        self.scaler.scale_ = np.concat([self.dynamic_scaler.scale_, self.scaler_static.scale_])

    def __read_data__(self):
        import glob

        # Get all MW files
        mw_files = glob.glob(os.path.join(self.root_path, self.data_path, "MW_*.csv"))
        self.mw_ids = sorted([os.path.basename(f).replace(".csv", "") for f in mw_files])

        # print("ACHTUNG NUR 100 BRUNNGEN data_loader read_data")
        # self.mw_ids = self.mw_ids[:80]
        self.train_mws = self.mw_ids
        self.val_mws = self.mw_ids
        self.test_mws = self.mw_ids
        # Split wells: 70% train, 20% val, 10% test
        # np.random.seed(42)  # for reproducibility
        # np.random.shuffle(self.mw_ids)
        # n_total = len(self.mw_ids)
        # n_train = int(0.7 * n_total)
        # n_val = int(0.2 * n_total)
        # self.train_mws = self.mw_ids[:n_train]
        # self.val_mws = self.mw_ids[n_train:n_train + n_val]
        # self.test_mws = self.mw_ids[n_train + n_val:]

        if self.set_type == 0:
            relevant_mws = self.train_mws
        elif self.set_type == 1:
            relevant_mws = self.val_mws
        else:
            relevant_mws = self.test_mws

        self.well_data = {}
        self.mw_list = []
        self.cutoffs = [0]

        static_df = pd.read_csv(self.static_path, index_col=0)
        static_df.set_index("MW_ID", inplace=True)

        static_df.drop(
            [
                "Proj_ID",
                "Operator",
                "Depth",
                "UpFilter",
                "LoFilter",
                "ScrLength",
                "PreState",  #'Pumping',
                "Easting (EPSG:3035)",
                "Northing (EPSG:3035)",
            ],
            axis=1,
            inplace=True,
        )

        # Encode categorical columns
        categorical_cols = [
            "AquiferMed",
            "HUEK250_HU",
            "HUEK250_RT",
            "HUEK250_CT",
            "HUEK250_DC",
            "HUEK250_GC",
            "HUMUS1000_OC",
        ]
        self.encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            static_df[col] = le.fit_transform(static_df[col])
            self.encoders[col] = le

        static_train = static_df.loc[self.train_mws]
        self.scaler_static = StandardScaler().fit(static_train.values)
        static_df = pd.DataFrame(
            self.scaler_static.transform(static_df), columns=static_df.columns, index=static_df.index
        )

        for mw_id in relevant_mws:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, f"{mw_id}.csv"))

            df_raw.rename(columns={"Unnamed: 0": "date"}, inplace=True)
            df_raw["date"] = pd.to_datetime(df_raw["date"])

            if self.set_type == 0:
                df_raw = df_raw[df_raw["date"] <= self.train_end]
            elif self.set_type == 1:
                df_raw = df_raw[(df_raw["date"] > self.train_end) & (df_raw["date"] <= self.val_end)]
            elif self.set_type == 2:
                df_raw = df_raw[df_raw["date"] > self.val_end]

            # Load static
            if mw_id not in static_df.index:
                print(f"Skipping {mw_id}, no static")
                continue
            static_row = static_df.loc[mw_id]
            static_features = self.scaler_static.transform(static_row.values.reshape(1, -1)).flatten()
            static_features = static_row.values

            if self.diff:
                cols = [self.target] if self.features in ["S", "MS"] else list(df_raw.columns[1:])
                df_diff = df_raw[cols].diff()
                df_diff.iloc[0] = 0
                df_diff.columns = df_diff.columns + "_d1"
                df_raw = pd.concat([df_raw, df_diff], axis=1)

            cols = list(df_raw.columns)
            cols.remove("date")
            df_raw = df_raw[["date", *cols]]

            if self.features in {"M", "MS"}:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == "S" and self.diff:
                df_data = df_raw[[self.target, f"{self.target}_d1"]]
            else:
                df_data = df_raw[[self.target]]
            if self.set_type == 0:
                self.dynamic_scaler.partial_fit(df_data.values)

            df_stamp = df_raw[["date"]]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            data_stamp = get_data_stamp(df_stamp, self.timeenc, self.freq)

            # data_x = torch.tensor(data, dtype=torch.float32)
            data_stamp = torch.tensor(data_stamp, dtype=torch.float32)

            # Add windows
            num_windows = len(df_data.values) - self.seq_len - self.pred_len + 1
            if num_windows > 0:
                self.well_data[mw_id] = {
                    "data_x": df_data.values,
                    "data_stamp": data_stamp,
                    "width_dynamic": df_data.values.shape[1],
                    "static": static_features,
                    "data_x_raw": df_raw.set_index("date"),  # df_data  # without static
                }

                self.mw_list.append(mw_id)
                self.cutoffs.append(self.cutoffs[-1] + num_windows)

        for mw_id in self.mw_list:
            self.well_data[mw_id]["data_x"] = torch.tensor(
                self.dynamic_scaler.transform(self.well_data[mw_id]["data_x"]), dtype=torch.float32
            )

    @property
    def data_x_raw(self):
        first_well = next(iter(self.well_data.keys()))
        return self.well_data[first_well]["data_x_raw"]

    def __getitem__(self, index):
        # Find the MW and the offset
        for j in range(len(self.cutoffs) - 1):
            if index < self.cutoffs[j + 1]:
                mw_id = self.mw_list[j]
                i = index - self.cutoffs[j]
                break
        data = self.well_data[mw_id]

        s_begin = i
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data["data_x"][s_begin:s_end]
        seq_y = data["data_x"][r_begin:r_end]
        seq_x_mark = data["data_stamp"][s_begin:s_end]
        seq_y_mark = data["data_stamp"][r_begin:r_end]

        if self.diff:
            if self.features == "M":
                seq_x[0, data["width_dynamic"] // 2 :] = 0
            elif self.features in {"MS", "S"}:
                seq_x[0, -1] = 0

        # Add static features
        static_repeated = np.tile(data["static"], (self.seq_len, 1))
        seq_x = torch.cat([seq_x, torch.tensor(static_repeated, dtype=torch.float32)], dim=1)

        # For seq_y, also add static
        static_repeated_y = np.tile(data["static"], (self.label_len + self.pred_len, 1))
        seq_y = torch.cat([seq_y, torch.tensor(static_repeated_y, dtype=torch.float32)], dim=1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return self.cutoffs[-1]

    def inverse_transform(self, data):
        # Since multiple scalers, this is not straightforward; perhaps implement per sample if needed
        return data

    def get_pred_index(self):
        # Per well, but for simplicity
        return []
