"""Builds the 40 model features: calendar, lags, rolling stats and weather.

Everything derived from past demand is shifted back by the full forecast horizon,
so no feature can see anything that was not available at prediction time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gridpulse.config import FORECAST_HORIZON
from gridpulse.warehouse.duck import query

logger = logging.getLogger(__name__)

BALANCE_POINT_C = 18.0

DEMAND_PLAUSIBLE_LOWER = 0.2
DEMAND_PLAUSIBLE_UPPER = 5.0

LAG_HOURS = (24, 25, 26, 48, 72, 168, 336)
ROLLING_WINDOWS = (24, 168)

FEATURE_COLUMNS: list[str] = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "is_business_day",
    "is_day_before_holiday", "is_day_after_holiday",
    *[f"demand_lag_{h}h" for h in LAG_HOURS],
    *[f"demand_roll_mean_{w}h" for w in ROLLING_WINDOWS],
    *[f"demand_roll_std_{w}h" for w in ROLLING_WINDOWS],
    "demand_same_hour_last_week_delta",
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dew_point_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation",
    "heating_degrees", "cooling_degrees", "temp_squared",
    "temp_lag_24h", "temp_change_24h", "temp_roll_mean_24h",
    "cooling_x_business", "heating_x_business", "cooling_x_hour",
]

TARGET = "demand_mwh"


def load_modelling_frame(ba_codes: list[str] | None = None) -> pd.DataFrame:
    """Pull the gold fact table into memory for feature construction."""
    where = ""
    if ba_codes:
        codes = ", ".join(f"'{c.upper()}'" for c in ba_codes)
        where = f"WHERE ba_code IN ({codes})"

    frame = query(f"""
        SELECT period_utc, ba_code, date_local, hour_local,
               -- The warehouse is the single source of truth for what counts as a
               -- trustworthy reading. Modelling consumes demand_clean_mwh so the
               -- cleaning rules live in one place rather than being reimplemented here.
               demand_clean_mwh AS demand_mwh, demand_forecast_mwh,
               temperature_2m, apparent_temperature, relative_humidity_2m,
               dew_point_2m, cloud_cover, wind_speed_10m, shortwave_radiation,
               day_of_week, is_weekend, is_holiday, is_business_day,
               is_day_before_holiday, is_day_after_holiday, month, year
        FROM fact_demand_hourly
        {where}
        ORDER BY ba_code, period_utc
    """)
    frame["period_utc"] = pd.to_datetime(frame["period_utc"], utc=True)
    return frame


def flag_implausible_demand(frame: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """True where demand is impossible compared to that region's own median."""
    median = frame.groupby("ba_code")[target].transform("median")
    return (frame[target] < median * DEMAND_PLAUSIBLE_LOWER) | (
        frame[target] > median * DEMAND_PLAUSIBLE_UPPER
    )


def _cyclical(frame: pd.DataFrame, column: pd.Series, period: int, prefix: str) -> None:
    radians = 2 * np.pi * column / period
    frame[f"{prefix}_sin"] = np.sin(radians)
    frame[f"{prefix}_cos"] = np.cos(radians)


def _engineer_one_ba(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features for a single BA. Assumes the frame is sorted by period."""
    out = frame.sort_values("period_utc").copy()

    _cyclical(out, out["hour_local"], 24, "hour")
    _cyclical(out, out["day_of_week"], 7, "dow")
    _cyclical(out, pd.to_datetime(out["date_local"]).dt.dayofyear, 365.25, "doy")

    for flag in ("is_weekend", "is_holiday", "is_business_day",
                 "is_day_before_holiday", "is_day_after_holiday"):
        out[flag] = out[flag].fillna(False).astype(int)

    demand = out[TARGET]
    for lag in LAG_HOURS:
        out[f"demand_lag_{lag}h"] = demand.shift(lag)

    shifted = demand.shift(FORECAST_HORIZON)
    for window in ROLLING_WINDOWS:
        out[f"demand_roll_mean_{window}h"] = shifted.rolling(window, min_periods=window // 4).mean()
        out[f"demand_roll_std_{window}h"] = shifted.rolling(window, min_periods=window // 4).std()

    out["demand_same_hour_last_week_delta"] = (
        out["demand_lag_168h"] - out["demand_lag_336h"]
    )

    temp = out["temperature_2m"]
    out["heating_degrees"] = (BALANCE_POINT_C - temp).clip(lower=0)
    out["cooling_degrees"] = (temp - BALANCE_POINT_C).clip(lower=0)
    out["temp_squared"] = temp**2
    out["temp_lag_24h"] = temp.shift(24)
    out["temp_change_24h"] = temp - out["temp_lag_24h"]
    out["temp_roll_mean_24h"] = temp.rolling(24, min_periods=6).mean()

    out["cooling_x_business"] = out["cooling_degrees"] * out["is_business_day"]
    out["heating_x_business"] = out["heating_degrees"] * out["is_business_day"]
    out["cooling_x_hour"] = out["cooling_degrees"] * out["hour_local"]

    return out


def build_features(
    frame: pd.DataFrame | None = None,
    ba_codes: list[str] | None = None,
    dropna_target: bool = True,
) -> pd.DataFrame:
    """Build the full feature table the models train on.

    Set ``dropna_target=False`` when building features for future hours, where
    there is no actual demand to compare against yet.
    """
    source = load_modelling_frame(ba_codes) if frame is None else frame

    engineered = pd.concat(
        [_engineer_one_ba(group) for _, group in source.groupby("ba_code", sort=True)],
        ignore_index=True,
    )

    weather_columns = [
        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "dew_point_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation",
    ]
    engineered[weather_columns] = (
        engineered.groupby("ba_code")[weather_columns]
        .transform(lambda s: s.interpolate(limit=6, limit_direction="both"))
    )

    if dropna_target:
        implausible = flag_implausible_demand(engineered)
        if implausible.any():
            by_ba = engineered.loc[implausible].groupby("ba_code").size().to_dict()
            logger.warning(
                "Excluding %d implausible demand reading(s) from modelling: %s",
                int(implausible.sum()), by_ba,
            )
        engineered = engineered[~implausible]
        engineered = engineered.dropna(subset=[TARGET])

    required = [c for c in FEATURE_COLUMNS if c.startswith("demand_lag")]
    engineered = engineered.dropna(subset=required)

    logger.info(
        "Features built: %s rows x %d columns across %d BA(s)",
        f"{len(engineered):,}", len(FEATURE_COLUMNS), engineered["ba_code"].nunique(),
    )
    return engineered.reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, test_days: int = 90, valid_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date only, never randomly, so the model cannot see the future."""
    cutoff_test = frame["period_utc"].max() - pd.Timedelta(days=test_days)
    cutoff_valid = cutoff_test - pd.Timedelta(days=valid_days)

    train = frame[frame["period_utc"] <= cutoff_valid]
    valid = frame[(frame["period_utc"] > cutoff_valid) & (frame["period_utc"] <= cutoff_test)]
    test = frame[frame["period_utc"] > cutoff_test]

    logger.info(
        "Split -> train %s | valid %s | test %s (test window from %s)",
        f"{len(train):,}", f"{len(valid):,}", f"{len(test):,}", cutoff_test.date(),
    )
    return train, valid, test
