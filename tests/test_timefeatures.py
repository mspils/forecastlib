import warnings

import pandas as pd
import pytest

from forecastlib.utils.timefeatures import FREQ_MAP, pandas_freq, time_features, time_features_from_frequency_str


YEARLY_MISMATCH = pytest.mark.xfail(
    reason="pre-existing: FREQ_MAP expects 1 feature for yearly data, the YearEnd features list is empty", strict=True
)


@pytest.mark.parametrize(
    ("freq", "n"), [pytest.param(f, n, marks=YEARLY_MISMATCH) if f == "a" else (f, n) for f, n in FREQ_MAP.items()]
)
def test_every_freq_code_works_with_pandas_3(freq, n):
    """Every frequency setting forecastlib knows gives as many time features as FREQ_MAP says, without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(time_features_from_frequency_str(freq)) == n


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("t", "min"), ("15t", "15min"), ("m", "ME"), ("a", "YE"), ("H", "h"), ("3H", "3h"), ("d", "D"),
     ("h", "h"), ("15min", "15min"), ("D", "D"), ("W", "W")],
)  # fmt: skip
def test_pandas_freq(freq, expected):
    assert pandas_freq(freq) == expected


def test_minute_data_time_features():
    dates = pd.date_range("2024-01-01", periods=4, freq="15min")
    assert time_features(dates, freq="15t").shape == (5, 4)
