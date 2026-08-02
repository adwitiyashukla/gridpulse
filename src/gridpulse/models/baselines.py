"""Naive and classical statistical baselines.

Every forecasting claim is meaningless without a floor to compare against. These
three establish it:

**Seasonal naive (24h)**
    Tomorrow at 3pm equals today at 3pm. Trivial, and startlingly hard to beat.

**Weekly naive (168h)**
    Tomorrow at 3pm equals the same weekday last week at 3pm. Usually stronger than
    the daily variant because it preserves the weekday/weekend regime.

**Holt-Winters**
    Triple exponential smoothing with a daily seasonal cycle: a real statistical
    model, no exogenous inputs, representing what a utility analyst could build in
    a spreadsheet.

If a deep network cannot beat weekly naive, the deep network is not working.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seasonal_naive(frame: pd.DataFrame, season_hours: int = 24, target: str = "demand_mwh") -> pd.Series:
    """Predict each hour as the observation ``season_hours`` earlier."""
    return frame.groupby("ba_code")[target].shift(season_hours)


def weekly_naive(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.Series:
    return seasonal_naive(frame, season_hours=168, target=target)


def drift_naive(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.Series:
    """Weekly naive nudged by the recent week-on-week trend."""
    last_week = frame.groupby("ba_code")[target].shift(168)
    two_weeks = frame.groupby("ba_code")[target].shift(336)
    return last_week + 0.5 * (last_week - two_weeks)


def holt_winters(
    train: pd.Series, horizon: int, seasonal_periods: int = 24
) -> np.ndarray:
    """Fit Holt-Winters on one series and forecast ``horizon`` steps.

    Falls back to the seasonal mean if statsmodels is unavailable or the fit fails
    to converge, which it occasionally does on series with long flat stretches.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        clean = train.dropna()
        if len(clean) < seasonal_periods * 3:
            raise ValueError("insufficient history for a seasonal fit")

        model = ExponentialSmoothing(
            clean,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.asarray(model.forecast(horizon))

    except Exception as exc:  # noqa: BLE001
        logger.warning("Holt-Winters fell back to seasonal mean: %s", exc)
        tail = train.dropna().tail(seasonal_periods * 4)
        if tail.empty:
            return np.full(horizon, np.nan)
        pattern = tail.groupby(np.arange(len(tail)) % seasonal_periods).mean()
        return np.asarray([pattern.iloc[i % seasonal_periods] for i in range(horizon)])


def build_all_baselines(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.DataFrame:
    """Attach every naive baseline plus EIA's official forecast as columns."""
    out = frame.copy()
    out["pred_seasonal_naive"] = seasonal_naive(out, 24, target)
    out["pred_weekly_naive"] = weekly_naive(out, target)
    out["pred_drift_naive"] = drift_naive(out, target)
    if "demand_forecast_mwh" in out.columns:
        out["pred_eia_official"] = out["demand_forecast_mwh"]
    return out
