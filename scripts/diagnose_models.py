"""Diagnostic: find where the model predictions go wrong.

Prints the target scaler's fitted statistics, per-balancing-authority error, and
the worst individual predictions. Run after `gridpulse train`::

    python scripts/diagnose_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridpulse.features.build import build_features  # noqa: E402
from gridpulse.models.gbm import TrainedGBM  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

TEST_DAYS = 90
VALID_DAYS = 60


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def main() -> int:
    rule("1. Loading model artifacts")
    gbm = TrainedGBM.load()
    print(f"  best_iteration : {gbm.best_iteration}")
    print(f"  features       : {len(gbm.feature_names)}")
    print(f"  ba_categories  : {gbm.ba_categories}")

    rule("2. Target scaler statistics (fitted on training rows)")
    stats = pd.DataFrame(
        [{"ba_code": k, "mean_mw": v[0], "std_mw": v[1]} for k, v in gbm.target_scaler.stats.items()]
    ).sort_values("mean_mw", ascending=False)
    print(stats.to_string(index=False))
    print(f"\n  global_mean : {gbm.target_scaler.global_mean:,.1f}")
    print(f"  global_std  : {gbm.target_scaler.global_std:,.1f}")
    print(f"  scale ratio (largest/smallest BA mean): "
          f"{stats['mean_mw'].max() / stats['mean_mw'].min():.1f}x")

    rule("3. Rebuilding the test split")
    frame = build_features()
    test_start = frame["period_utc"].max() - pd.Timedelta(days=TEST_DAYS)
    test = frame[frame["period_utc"] >= test_start].copy()
    print(f"  test rows : {len(test):,}")
    print(f"  BAs       : {sorted(test['ba_code'].unique())}")

    rule("4. Does every test BA resolve in the scaler?")
    missing = sorted(set(test["ba_code"].unique()) - set(gbm.target_scaler.stats))
    if missing:
        print(f"  !! MISSING, will silently fall back to global stats: {missing}")
    else:
        print("  OK - every test BA is present in the scaler.")

    rule("5. Raw model output, BEFORE the inverse transform")
    matrix = test[gbm.feature_names].copy()
    matrix["ba_code"] = pd.Categorical(test["ba_code"], categories=gbm.ba_categories)
    raw = gbm.point_model.predict(matrix)
    print(f"  z-score predictions   min={raw.min():9.3f}  max={raw.max():9.3f}  "
          f"mean={raw.mean():8.3f}  std={raw.std():7.3f}")
    print("  (a healthy z-score prediction sits roughly within -4 .. +4)")

    actual_z = gbm.target_scaler.transform(test)
    print(f"  z-score actuals       min={actual_z.min():9.3f}  max={actual_z.max():9.3f}  "
          f"mean={actual_z.mean():8.3f}  std={actual_z.std():7.3f}")

    rule("6. Predictions AFTER the inverse transform")
    predicted = gbm.predict(test)
    test = test.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)
    test["pred"] = predicted["pred_gbm"].to_numpy()
    test["ape"] = (test["pred"] - test["demand_mwh"]).abs() / test["demand_mwh"] * 100

    print(f"  predicted MW  min={test['pred'].min():12,.0f}  max={test['pred'].max():12,.0f}")
    print(f"  actual    MW  min={test['demand_mwh'].min():12,.0f}  max={test['demand_mwh'].max():12,.0f}")

    rule("7. Error by balancing authority")
    by_ba = (
        test.groupby("ba_code")
        .agg(
            hours=("demand_mwh", "size"),
            actual_mean_mw=("demand_mwh", "mean"),
            pred_mean_mw=("pred", "mean"),
            mape_pct=("ape", "mean"),
            worst_ape_pct=("ape", "max"),
        )
        .sort_values("mape_pct", ascending=False)
        .round(2)
    )
    print(by_ba.to_string())

    rule("8. The 15 worst individual predictions")
    worst = test.nlargest(15, "ape")[
        ["period_utc", "ba_code", "demand_mwh", "pred", "ape", "demand_lag_24h", "demand_lag_168h"]
    ]
    print(worst.to_string(index=False))

    rule("9. How concentrated is the damage?")
    for threshold in (10, 25, 50, 100, 500):
        share = (test["ape"] > threshold).mean() * 100
        print(f"  rows with APE > {threshold:4d}% : {share:6.2f}%")

    contribution = test.nlargest(int(len(test) * 0.01), "ape")["ape"].sum() / test["ape"].sum() * 100
    print(f"\n  the worst 1% of rows account for {contribution:.1f}% of total absolute error")
    print(f"  overall MAPE {test['ape'].mean():.3f}%   median APE {test['ape'].median():.3f}%")
    print("  (a median far below the mean confirms a small number of extreme outliers)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
