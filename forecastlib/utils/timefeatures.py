# From: gluonts/src/gluonts/time_feature/_base.py  # noqa: ERA001
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Time feature extraction utilities for time series forecasting.

This module provides classes and functions for extracting temporal features
from datetime indices. Features are encoded as normalized scalar values
(typically in range [-0.5, 0.5]) suitable for use as model inputs.

Features can be extracted at various granularities (second, minute, hour, day, week, month, year)
and are automatically selected based on the frequency of the time series.
"""

from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """Abstract base class for time feature extractors.

    Time features extract temporal information from datetime indices and encode
    them as normalized scalar values suitable for machine learning models.

    Subclasses should implement the __call__ method to extract a specific
    temporal feature (e.g., hour of day, day of week).
    """

    def __init__(self) -> None:
        """Initialize the TimeFeature extractor."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract time feature values from a datetime index.

        Args:
            index: A pandas DatetimeIndex containing timestamps to extract features from.

        Returns:
            Normalized array of feature values, typically in range [-0.5, 0.5].

        """

    def __repr__(self) -> str:
        """Return string representation of the TimeFeature instance.

        Returns:
            String representation showing the class name.

        """
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Extract the second of the minute as a normalized feature.

    Extracts which second within a minute (0-59) and normalizes it to [-0.5, 0.5].
    Normalization formula: (second / 59.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract second of minute feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized second values in range [-0.5, 0.5].

        """
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Extract the minute of the hour as a normalized feature.

    Extracts which minute within an hour (0-59) and normalizes it to [-0.5, 0.5].
    Normalization formula: (minute / 59.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract minute of hour feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized minute values in range [-0.5, 0.5].

        """
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Extract the hour of the day as a normalized feature.

    Extracts which hour within a day (0-23) and normalizes it to [-0.5, 0.5].
    Normalization formula: (hour / 23.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract hour of day feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized hour values in range [-0.5, 0.5].

        """
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Extract the day of the week as a normalized feature.

    Extracts which day within a week (0-6, Monday-Sunday) and normalizes it to [-0.5, 0.5].
    Normalization formula: (dayofweek / 6.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of week feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized day of week values in range [-0.5, 0.5].

        """
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Extract the day of the month as a normalized feature.

    Extracts which day within a month (1-31) and normalizes it to [-0.5, 0.5].
    Normalization formula: ((day - 1) / 30.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of month feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized day of month values in range [-0.5, 0.5].

        """
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Extract the day of the year as a normalized feature.

    Extracts which day within a year (1-365/366) and normalizes it to [-0.5, 0.5].
    Normalization formula: ((dayofyear - 1) / 365.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of year feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized day of year values in range [-0.5, 0.5].

        """
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Extract the month of the year as a normalized feature.

    Extracts which month within a year (1-12) and normalizes it to [-0.5, 0.5].
    Normalization formula: ((month - 1) / 11.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract month of year feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized month of year values in range [-0.5, 0.5].

        """
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Extract the week of the year as a normalized feature.

    Extracts which week within a year (1-52) and normalizes it to [-0.5, 0.5].
    Normalization formula: ((week - 1) / 52.0) - 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract week of year feature.

        Args:
            index: A pandas DatetimeIndex containing timestamps.

        Returns:
            Normalized week of year values in range [-0.5, 0.5].

        """
        return (index.isocalendar().week - 1) / 52.0 - 0.5


FREQ_MAP = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "W": 2, "d": 3, "D": 3, "b": 3}


def time_features_from_frequency_str(freq_str: str) -> list[TimeFeature]:
    """Return a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    """Extract time features for each datetime depending on the frequency.

    Returns:
        np.ndarray: Array with time features for each datetime.

    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def get_data_stamp(df_stamp: pd.DataFrame, time_enc: Literal[0, 1], freq: str) -> np.ndarray:
    """Extract and encode temporal features from a DataFrame with timestamps.

    Args:
        df_stamp: DataFrame containing a 'date' column with datetime values.
        time_enc: Encoding type for temporal features. 0 for manual encoding
            (month, day, weekday, hour/minute), 1 for Fourier features.
        freq: Frequency of the time series ('h' for hourly, 't' for minute, etc.).

    Returns:
        np.ndarray: Array of temporal features. Shape depends on time_enc:
            - If time_enc == 0: (n_samples, n_features) where features are
                month, day, weekday, and optionally hour/minute.
            - If time_enc == 1: (n_features, n_samples) transposed Fourier features.

    Raises:
        ValueError: If time_enc is not 0 or 1.

    """
    if time_enc == 0:
        df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        if freq in {"h", "t"}:
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
        elif freq == "t":
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)

        data_stamp = df_stamp.drop(["date"], axis=1).values
    elif time_enc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    else:
        msg = f"timeenc is {time_enc}, but must be 0 or 1"
        raise ValueError(msg)

    return data_stamp
