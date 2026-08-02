"""Feature engineering, with particular attention to temporal leakage.

Leakage is the failure mode that silently ruins time-series projects: the model
scores beautifully in validation and collapses in production because a feature
encoded information that would not have existed at prediction time. These tests
assert it cannot happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridpulse.config import FORECAST_HORIZON
from gridpulse.features.build import (
    FEATURE_COLUMNS,
    TARGET,
    build_features,
    chronological_split,
)


@pytest.fixture
def raw_frame(synthetic_grid: pd.DataFrame) -> pd.DataFrame:
    frame = synthetic_grid.copy()
    local = frame["period_utc"].dt.tz_convert("America/New_York")
    frame["date_local"] = local.dt.date
    frame["hour_local"] = local.dt.hour
    frame["day_of_week"] = local.dt.dayofweek
    frame["month"] = local.dt.month
    frame["year"] = local.dt.year
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["is_holiday"] = False
    frame["is_business_day"] = ~frame["is_weekend"]
    frame["is_day_before_holiday"] = False
    frame["is_day_after_holiday"] = False
    for column in ["apparent_temperature", "relative_humidity_2m", "dew_point_2m",
                   "cloud_cover", "wind_speed_10m", "shortwave_radiation"]:
        frame[column] = 10.0
    return frame


def test_all_declared_features_are_produced(raw_frame):
    featured = build_features(frame=raw_frame)
    missing = set(FEATURE_COLUMNS) - set(featured.columns)
    assert not missing, f"Declared but not built: {sorted(missing)}"


def test_no_nulls_remain_in_lag_features(raw_frame):
    featured = build_features(frame=raw_frame)
    lag_columns = [c for c in FEATURE_COLUMNS if c.startswith("demand_lag")]
    assert featured[lag_columns].isna().sum().sum() == 0


def test_cyclical_encoding_wraps_around(raw_frame):
    """Hour 23 and hour 0 must be neighbours in the encoded space."""
    featured = build_features(frame=raw_frame)
    hour_23 = featured[featured["hour_local"] == 23].iloc[0]
    hour_0 = featured[featured["hour_local"] == 0].iloc[0]
    hour_12 = featured[featured["hour_local"] == 12].iloc[0]

    def distance(a, b):
        return np.hypot(a["hour_sin"] - b["hour_sin"], a["hour_cos"] - b["hour_cos"])

    assert distance(hour_23, hour_0) < distance(hour_23, hour_12)


def test_lag_features_reference_the_correct_past_value(raw_frame):
    """demand_lag_24h at time t must equal actual demand at t - 24h."""
    featured = build_features(frame=raw_frame)
    single = featured[featured["ba_code"] == "PJM"].sort_values("period_utc")
    source = raw_frame[raw_frame["ba_code"] == "PJM"].set_index("period_utc")[TARGET]

    probe = single.iloc[500]
    expected = source.loc[probe["period_utc"] - pd.Timedelta(hours=24)]
    assert probe["demand_lag_24h"] == pytest.approx(expected)


def test_rolling_features_do_not_leak_the_present(raw_frame):
    """Rolling statistics must be shifted by at least the forecast horizon."""
    featured = build_features(frame=raw_frame)
    single = featured[featured["ba_code"] == "PJM"].sort_values("period_utc").reset_index(drop=True)
    source = raw_frame[raw_frame["ba_code"] == "PJM"].sort_values("period_utc").reset_index(drop=True)

    probe_index = 800
    probe = single.iloc[probe_index]
    position = source.index[source["period_utc"] == probe["period_utc"]][0]

    window = source[TARGET].iloc[position - FORECAST_HORIZON - 23 : position - FORECAST_HORIZON + 1]
    assert probe["demand_roll_mean_24h"] == pytest.approx(window.mean(), rel=1e-6)


def test_degree_days_split_the_temperature_response(raw_frame):
    featured = build_features(frame=raw_frame)
    # The two limbs are mutually exclusive: never both positive at once.
    both_positive = (featured["heating_degrees"] > 0) & (featured["cooling_degrees"] > 0)
    assert not both_positive.any()
    assert (featured["heating_degrees"] >= 0).all()
    assert (featured["cooling_degrees"] >= 0).all()


def test_chronological_split_never_overlaps(raw_frame):
    featured = build_features(frame=raw_frame)
    train, valid, test = chronological_split(featured, test_days=30, valid_days=30)

    assert train["period_utc"].max() <= valid["period_utc"].min()
    assert valid["period_utc"].max() <= test["period_utc"].min()
    assert len(train) + len(valid) + len(test) == len(featured)


def test_inference_frame_keeps_rows_without_a_target(raw_frame):
    """Future rows have no actual demand yet and must survive feature building."""
    frame = raw_frame.copy()
    frame.loc[frame.index[-12:], TARGET] = np.nan

    with_target = build_features(frame=frame, dropna_target=True)
    without = build_features(frame=frame, dropna_target=False)
    assert len(without) > len(with_target)
