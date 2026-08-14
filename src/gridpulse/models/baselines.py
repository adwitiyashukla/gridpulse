from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def seasonal_naive(frame: pd.DataFrame, season_hours: int = 24, target: str = "demand_mwh") -> pd.Series:
    return frame.groupby("ba_code")[target].shift(season_hours)


def weekly_naive(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.Series:
    return seasonal_naive(frame, season_hours=168, target=target)


def drift_naive(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.Series:
    last_week = frame.groupby("ba_code")[target].shift(168)
    two_weeks = frame.groupby("ba_code")[target].shift(336)
    return last_week + 0.5 * (last_week - two_weeks)


def holt_winters(
    train: pd.Series, horizon: int, seasonal_periods: int = 24
) -> np.ndarray:
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

    except Exception as exc:
        logger.warning("Holt-Winters fell back to seasonal mean: %s", exc)
        tail = train.dropna().tail(seasonal_periods * 4)
        if tail.empty:
            return np.full(horizon, np.nan)
        pattern = tail.groupby(np.arange(len(tail)) % seasonal_periods).mean()
        return np.asarray([pattern.iloc[i % seasonal_periods] for i in range(horizon)])


def build_all_baselines(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.DataFrame:
    out = frame.copy()
    out["pred_seasonal_naive"] = seasonal_naive(out, 24, target)
    out["pred_weekly_naive"] = weekly_naive(out, target)
    out["pred_drift_naive"] = drift_naive(out, target)
    if "demand_forecast_mwh" in out.columns:
        out["pred_eia_official"] = out["demand_forecast_mwh"]
    return out
