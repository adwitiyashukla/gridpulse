"""Forecast accuracy metrics, framed the way a utility actually scores them.

MAPE is the industry lingua franca for load forecasting -- a system operator will
quote "we run about 2 percent MAPE" without further qualification -- so it leads
here. It is reported alongside MAE and RMSE because MAPE alone hides the cost
asymmetry: RMSE punishes the large misses that trigger expensive peaker starts,
while MAPE treats a 500 MW miss at 3am the same as one at 5pm on a heatwave.

``skill_vs_benchmark`` expresses accuracy as percentage improvement over EIA's own
published day-ahead forecast, which is the only comparison a hiring manager in this
industry will care about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred) & (true > 0)
    return true[mask], pred[mask]


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error."""
    true, pred = _clean(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs((true - pred) / true)) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE; bounded and does not explode near zero."""
    true, pred = _clean(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    denominator = (np.abs(true) + np.abs(pred)) / 2
    return float(np.mean(np.abs(true - pred) / denominator) * 100)


def mae(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(true - pred))) if true.size else float("nan")


def rmse(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2))) if true.size else float("nan")


def r2(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")


def peak_hour_mape(frame: pd.DataFrame, actual: str, predicted: str) -> float:
    """MAPE restricted to each day's peak-demand hour.

    Peak accuracy drives capacity procurement and is where forecast error becomes
    genuinely expensive, so it is scored separately from the all-hours average.
    """
    if frame.empty:
        return float("nan")
    peaks = frame.loc[frame.groupby(frame["period_utc"].dt.date)[actual].idxmax()]
    return mape(peaks[actual], peaks[predicted])


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss -- the correct score for probabilistic forecasts."""
    true, pred = _clean(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    delta = true - pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def coverage(y_true, lower, upper) -> float:
    """Share of actuals falling inside the predicted interval.

    A well-calibrated 80 percent interval should contain roughly 80 percent of
    outcomes. Much higher means the interval is uselessly wide.
    """
    true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(true) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    return float(np.mean((true[mask] >= lo[mask]) & (true[mask] <= hi[mask])) * 100)


def evaluate_forecast(y_true, y_pred, label: str = "model") -> dict:
    """Standard metric bundle for one model on one dataset.

    ``n_obs`` counts the rows the metrics were actually computed on, after
    dropping non-finite and non-positive actuals. Reporting the raw input length
    here would overstate the sample behind every other number in the bundle.
    """
    scored, _ = _clean(y_true, y_pred)
    return {
        "model": label,
        "mape_pct": round(mape(y_true, y_pred), 4),
        "smape_pct": round(smape(y_true, y_pred), 4),
        "mae_mwh": round(mae(y_true, y_pred), 2),
        "rmse_mwh": round(rmse(y_true, y_pred), 2),
        "r2": round(r2(y_true, y_pred), 5),
        "n_obs": int(scored.size),
    }


def skill_vs_benchmark(model_mape: float, benchmark_mape: float) -> float:
    """Percentage improvement in MAPE over a benchmark.

    Positive means the model beats the benchmark. This is the headline number:
    ``skill_vs_benchmark(1.62, 2.14)`` -> ``24.3`` reads as "24 percent more
    accurate than EIA's own published day-ahead forecast".
    """
    if not np.isfinite(model_mape) or not np.isfinite(benchmark_mape) or benchmark_mape == 0:
        return float("nan")
    return round((benchmark_mape - model_mape) / benchmark_mape * 100, 2)
