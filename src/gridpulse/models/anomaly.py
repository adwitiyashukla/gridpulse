"""Grid anomaly detection: bad telemetry, demand shocks and abnormal load days.

Three detectors vote, because each alone has a characteristic blind spot:

**Robust seasonal z-score**
    Residual against the median load for that BA, hour-of-day and month, scaled by
    median absolute deviation. MAD is used rather than standard deviation precisely
    because the outliers we are hunting would inflate a standard deviation and hide
    themselves. Catches point spikes and drops; blind to correlated multi-hour drift.

**Isolation Forest**
    Unsupervised, multivariate, over demand level, ramp rate, temperature and the
    temperature/demand relationship. Catches combinations that are individually
    unremarkable -- moderate load at a moderate temperature can still be anomalous
    if that pairing never otherwise occurs.

**Daily-profile autoencoder**
    A small dense autoencoder compresses each 24-hour shape to an 8-dimensional
    bottleneck and reconstructs it. Days whose *shape* is unlike anything in the
    training set reconstruct badly, even when every individual hour sits inside its
    normal range. This is the detector that finds holidays behaving like weekends,
    storm days and demand-response events.

Consensus scoring keeps false positives manageable: an hour is flagged ``high``
severity only when at least two independent detectors agree.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridpulse.config import PATHS
from gridpulse.warehouse.duck import connect, query

logger = logging.getLogger(__name__)

Z_THRESHOLD = 4.0          # robust z beyond which an hour is suspicious
CONTAMINATION = 0.01       # expected anomaly rate for Isolation Forest
AE_PERCENTILE = 99.0       # reconstruction-error percentile defining "unusual shape"


# ---------------------------------------------------------------------------
# Detector 1: robust seasonal z-score
# ---------------------------------------------------------------------------
def robust_seasonal_z(frame: pd.DataFrame, target: str = "demand_mwh") -> pd.Series:
    """Median-absolute-deviation z-score within (BA, hour-of-day, month) cells."""
    work = frame[["ba_code", "hour_local", "month", target]].copy()
    grouped = work.groupby(["ba_code", "hour_local", "month"])[target]

    median = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    # 0.6745 rescales MAD to be comparable with a standard deviation under normality.
    scale = (mad / 0.6745).replace(0, np.nan)
    return ((work[target] - median) / scale).abs().fillna(0.0)


# ---------------------------------------------------------------------------
# Detector 2: Isolation Forest
# ---------------------------------------------------------------------------
ISO_FEATURES = ["demand_mwh", "ramp_mwh", "ramp_pct", "temperature_2m", "demand_per_degree"]


def _isolation_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ramp_mwh"] = work.groupby("ba_code")["demand_mwh"].diff()
    work["ramp_pct"] = work["ramp_mwh"] / work.groupby("ba_code")["demand_mwh"].shift(1) * 100
    # Demand normalised by distance from the comfort balance point: how much load
    # each degree of heating or cooling demand is buying.
    departure = (work["temperature_2m"] - 18.0).abs().clip(lower=0.5)
    work["demand_per_degree"] = work["demand_mwh"] / departure
    return work[ISO_FEATURES].replace([np.inf, -np.inf], np.nan)


def isolation_forest_scores(frame: pd.DataFrame, contamination: float = CONTAMINATION) -> tuple[pd.Series, object]:
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = _isolation_features(frame)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            n_estimators=150, contamination=contamination, random_state=42, n_jobs=2
        ),
    )
    model.fit(features)
    # decision_function is high for normal points; negate so high means anomalous.
    raw = -model[-1].decision_function(model[:-1].transform(features))
    return pd.Series(raw, index=frame.index), model


# ---------------------------------------------------------------------------
# Detector 3: daily-profile autoencoder
# ---------------------------------------------------------------------------
def _daily_profiles(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Reshape into one normalised 24-value vector per (BA, local date).

    Each day is divided by its own **median** so the autoencoder learns load
    *shape* rather than which BA is largest. The median is used instead of the
    mean because a single corrupt hour drags a mean toward itself; if that mean
    lands near zero the division explodes and the reconstruction loss becomes
    meaningless. Days whose level is not comfortably positive are dropped rather
    than rescued, since their shape cannot be trusted anyway.
    """
    pivot = (
        frame.pivot_table(index=["ba_code", "date_local"], columns="hour_local",
                          values="demand_mwh", aggfunc="mean")
        .dropna()
        .sort_index()
    )
    if pivot.empty:
        return np.empty((0, 24), dtype=np.float32), pd.DataFrame()

    values = pivot.to_numpy(dtype=np.float64)
    level = np.median(values, axis=1, keepdims=True)

    usable = (level.ravel() > 1.0) & np.isfinite(values).all(axis=1)
    values, level = values[usable], level[usable]
    index = pivot.index.to_frame(index=False).loc[usable].reset_index(drop=True)

    if values.size == 0:
        return np.empty((0, 24), dtype=np.float32), pd.DataFrame()

    normalised = values / level
    # A normalised day should sit near 1.0 throughout. Anything beyond this band
    # is a corrupt reading, not a load shape, and would dominate the loss.
    keep = (normalised > 0.05).all(axis=1) & (normalised < 20.0).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("  dropped %d degenerate daily profile(s) before autoencoder fit", dropped)

    return normalised[keep].astype(np.float32), index.loc[keep].reset_index(drop=True)


def autoencoder_scores(frame: pd.DataFrame, quick: bool = False) -> pd.DataFrame:
    """Per-day reconstruction error from a small dense autoencoder."""
    profiles, index = _daily_profiles(frame)
    if profiles.shape[0] < 100:
        logger.warning("Too few complete days (%d) for the autoencoder; skipping", profiles.shape[0])
        return pd.DataFrame()

    try:
        import torch
        from torch import nn
    except ImportError:
        logger.warning("PyTorch unavailable; autoencoder detector skipped")
        return pd.DataFrame()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    n_hours = profiles.shape[1]

    model = nn.Sequential(
        nn.Linear(n_hours, 32), nn.ReLU(),
        nn.Linear(32, 8), nn.ReLU(),          # bottleneck
        nn.Linear(8, 32), nn.ReLU(),
        nn.Linear(32, n_hours),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    tensor = torch.from_numpy(profiles)

    epochs = 30 if quick else 150
    batch = 128
    for epoch in range(epochs):
        permutation = torch.randperm(len(tensor))
        total = 0.0
        for start in range(0, len(tensor), batch):
            chunk = tensor[permutation[start : start + batch]]
            optimiser.zero_grad()
            loss = criterion(model(chunk), chunk)
            loss.backward()
            optimiser.step()
            total += loss.item()
        if (epoch + 1) % 50 == 0:
            logger.info("    autoencoder epoch %3d  loss %.6f", epoch + 1, total / max(1, len(tensor) // batch))

    model.eval()
    with torch.no_grad():
        errors = ((model(tensor) - tensor) ** 2).mean(dim=1).numpy()

    out = index.copy()
    out["ae_reconstruction_error"] = errors
    out["ae_threshold"] = np.percentile(errors, AE_PERCENTILE)
    out["ae_anomalous_day"] = errors > out["ae_threshold"]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def classify_anomaly(row: pd.Series) -> str:
    """Human-readable label so an operator knows what they are looking at."""
    if row.get("flag_frozen_reading"):
        return "frozen_telemetry"
    if row.get("flag_nonpositive_demand"):
        return "invalid_reading"
    if row.get("flag_missing_demand"):
        return "missing_interval"
    if abs(row.get("ramp_pct", 0) or 0) > 40:
        return "extreme_ramp"
    if (row.get("robust_z", 0) or 0) > Z_THRESHOLD:
        return "demand_spike" if row.get("residual_sign", 0) > 0 else "demand_drop"
    if row.get("ae_anomalous_day"):
        return "unusual_daily_shape"
    return "multivariate_outlier"


def run_anomaly_detection(quick: bool = False, persist: bool = True) -> pd.DataFrame:
    """Run every detector, combine by consensus and persist to ``anomaly_scores``."""
    frame = query("""
        SELECT period_utc, ba_code, date_local, hour_local, month,
               demand_mwh, temperature_2m,
               flag_missing_demand, flag_nonpositive_demand,
               flag_frozen_reading, flag_extreme_ramp,
               coalesce(flag_implausible_magnitude, false) AS flag_implausible_magnitude
        FROM fact_demand_hourly
        ORDER BY ba_code, period_utc
    """)
    if frame.empty:
        raise FileNotFoundError("No warehouse data. Run `gridpulse build` first.")

    frame["period_utc"] = pd.to_datetime(frame["period_utc"], utc=True)

    # Physically impossible readings are already flagged by the warehouse and
    # reported by the quality suite. Feeding them to statistical detectors would
    # let them define the very distribution used to judge everything else.
    implausible = frame["flag_implausible_magnitude"].fillna(False).astype(bool)
    if implausible.any():
        logger.info("  excluding %d implausible reading(s) from scoring", int(implausible.sum()))
    scored = frame[frame["demand_mwh"].notna() & ~implausible].copy()
    logger.info("Scoring %s hours for anomalies", f"{len(scored):,}")

    # Detector 1
    logger.info("  detector 1/3: robust seasonal z-score")
    scored["robust_z"] = robust_seasonal_z(scored)
    grouped = scored.groupby(["ba_code", "hour_local", "month"])["demand_mwh"]
    scored["residual_sign"] = np.sign(scored["demand_mwh"] - grouped.transform("median"))

    # Detector 2
    logger.info("  detector 2/3: isolation forest")
    scored["iso_score"], _ = isolation_forest_scores(scored)
    scored["iso_flag"] = scored["iso_score"] > scored["iso_score"].quantile(1 - CONTAMINATION)
    scored["ramp_mwh"] = scored.groupby("ba_code")["demand_mwh"].diff()
    scored["ramp_pct"] = scored["ramp_mwh"] / scored.groupby("ba_code")["demand_mwh"].shift(1) * 100

    # Detector 3
    logger.info("  detector 3/3: daily-profile autoencoder")
    daily = autoencoder_scores(scored, quick=quick)
    if not daily.empty:
        scored = scored.merge(
            daily[["ba_code", "date_local", "ae_reconstruction_error", "ae_anomalous_day"]],
            on=["ba_code", "date_local"], how="left",
        )
    else:
        scored["ae_reconstruction_error"] = np.nan
        scored["ae_anomalous_day"] = False

    scored["ae_anomalous_day"] = scored["ae_anomalous_day"].astype("boolean").fillna(False).astype(bool)

    # Consensus
    scored["z_flag"] = scored["robust_z"] > Z_THRESHOLD
    def _as_bool(column: str) -> pd.Series:
        return scored[column].astype("boolean").fillna(False).astype(bool)

    scored["rule_flag"] = (
        _as_bool("flag_frozen_reading")
        | _as_bool("flag_nonpositive_demand")
        | _as_bool("flag_extreme_ramp")
    )
    scored["detector_votes"] = (
        scored["z_flag"].astype(int)
        + scored["iso_flag"].astype(int)
        + scored["ae_anomalous_day"].astype(int)
        + scored["rule_flag"].astype(int)
    )
    scored["is_anomaly"] = scored["detector_votes"] >= 1
    scored["severity"] = pd.cut(
        scored["detector_votes"], bins=[-1, 0, 1, 2, 10],
        labels=["none", "low", "medium", "high"],
    ).astype(str)
    scored["anomaly_type"] = scored.apply(classify_anomaly, axis=1)
    scored.loc[~scored["is_anomaly"], "anomaly_type"] = "normal"

    output = scored[[
        "period_utc", "ba_code", "date_local", "hour_local", "demand_mwh", "temperature_2m",
        "robust_z", "iso_score", "ae_reconstruction_error", "ramp_pct",
        "z_flag", "iso_flag", "ae_anomalous_day", "rule_flag",
        "detector_votes", "is_anomaly", "severity", "anomaly_type",
    ]].copy()

    flagged = int(output["is_anomaly"].sum())
    logger.info(
        "Anomalies: %s of %s hours (%.2f%%); %s high severity",
        f"{flagged:,}", f"{len(output):,}", 100 * flagged / len(output),
        f"{int((output['severity'] == 'high').sum()):,}",
    )
    logger.info("Breakdown:\n%s", output.loc[output["is_anomaly"], "anomaly_type"].value_counts().to_string())

    if persist:
        with connect() as con:
            con.register("_anom", output)
            con.execute("CREATE OR REPLACE TABLE anomaly_scores AS SELECT * FROM _anom")
            con.unregister("_anom")

        summary_path = PATHS.artifacts / "anomaly_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({
            "hours_scored": int(len(output)),
            "anomalies_found": flagged,
            "anomaly_rate_pct": round(100 * flagged / len(output), 3),
            "by_type": output.loc[output["is_anomaly"], "anomaly_type"].value_counts().to_dict(),
            "by_severity": output["severity"].value_counts().to_dict(),
        }, indent=2))

    return output
