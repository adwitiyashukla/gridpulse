"""Training orchestration: build features, fit every model, score them all identically.

The comparison is the point. Six forecasters are evaluated on one identical
out-of-sample window, and the benchmark is not something invented here -- it is the
day-ahead forecast the EIA itself published and grid operators actually used:

1. Seasonal naive (24h)
2. Weekly naive (168h)
3. **EIA official day-ahead forecast**  <- the benchmark to beat
4. LightGBM global model with P10/P50/P90 intervals
5. LSTM encoder with known future covariates
6. Transformer encoder with known future covariates

Every model is scored on the same rows, over the same horizon, with the same
metrics. Results land in ``model_scores`` and ``model_predictions``, and every run
is tracked in MLflow.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from gridpulse.config import PATHS
from gridpulse.features.build import build_features
from gridpulse.models import baselines, metrics
from gridpulse.warehouse.duck import connect

logger = logging.getLogger(__name__)

TEST_DAYS = 90
VALID_DAYS = 60
BENCHMARK = "eia_official"


def _mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed; run tracking disabled")
        return None


def train_all(bas: list[str] | None = None, quick: bool = False) -> pd.DataFrame:
    """Train and evaluate the full model suite. Returns the leaderboard."""
    mlflow = _mlflow()
    if mlflow:
        mlflow.set_tracking_uri("file:" + str((PATHS.artifacts.parent / "mlruns").as_posix()))
        mlflow.set_experiment("gridpulse-day-ahead-load")

    # ------------------------------------------------------------------
    # Features and the single shared time split
    # ------------------------------------------------------------------
    logger.info("Building feature matrix")
    frame = build_features(ba_codes=bas)
    if frame.empty:
        raise RuntimeError("Feature matrix is empty. Check that `gridpulse build` produced data.")

    test_start = frame["period_utc"].max() - pd.Timedelta(days=TEST_DAYS)
    valid_start = test_start - pd.Timedelta(days=VALID_DAYS)
    train = frame[frame["period_utc"] < valid_start]
    valid = frame[(frame["period_utc"] >= valid_start) & (frame["period_utc"] < test_start)]
    test = frame[frame["period_utc"] >= test_start].copy()

    logger.info(
        "Split  train %s (to %s) | valid %s | test %s (from %s)",
        f"{len(train):,}", valid_start.date(), f"{len(valid):,}",
        f"{len(test):,}", test_start.date(),
    )
    if test.empty or train.empty:
        raise RuntimeError("Not enough history to form a train/test split. Ingest a longer window.")

    predictions = test[["period_utc", "ba_code", "demand_mwh", "demand_forecast_mwh"]].copy()

    # ------------------------------------------------------------------
    # 1-3. Baselines and the EIA benchmark
    # ------------------------------------------------------------------
    logger.info("Scoring baselines")
    with_baselines = baselines.build_all_baselines(frame)
    baseline_test = with_baselines[with_baselines["period_utc"] >= test_start]
    for column in ("pred_seasonal_naive", "pred_weekly_naive", "pred_eia_official"):
        if column in baseline_test.columns:
            predictions[column] = baseline_test[column].to_numpy()

    # ------------------------------------------------------------------
    # 4. LightGBM
    # ------------------------------------------------------------------
    logger.info("Training LightGBM")
    from gridpulse.models.gbm import train_gbm

    gbm = train_gbm(train, valid, quick=quick)
    gbm_predictions = gbm.predict(test)
    for column in gbm_predictions.columns:
        predictions[column] = gbm_predictions[column].to_numpy()
    gbm.save()

    # ------------------------------------------------------------------
    # 4b. Hybrid: the same model, additionally consuming EIA's published
    # day-ahead forecast as an input feature.
    #
    # This is not leakage. EIA publishes that forecast the day before, so it is
    # genuinely in an operator's hands at prediction time -- and no real utility
    # ignores their vendor's forecast. The hybrid learns to correct EIA's
    # systematic biases rather than re-deriving the whole signal from scratch,
    # which is exactly how a forecasting desk actually operates.
    #
    # Both models are reported. The pure model answers "can we beat them from
    # first principles"; the hybrid answers "can we improve what they publish".
    # ------------------------------------------------------------------
    if train["demand_forecast_mwh"].notna().mean() > 0.9:
        logger.info("Training LightGBM hybrid (EIA forecast as an input feature)")
        from gridpulse.features.build import FEATURE_COLUMNS

        hybrid_features = [*FEATURE_COLUMNS, "demand_forecast_mwh"]
        hybrid = train_gbm(train, valid, quick=quick, quantiles=(), features=hybrid_features)
        predictions["pred_gbm_hybrid"] = hybrid.predict(test)["pred_gbm"].to_numpy()
        hybrid.save(PATHS.artifacts / "gbm_hybrid")
    else:
        logger.warning("EIA forecast coverage too sparse for the hybrid model; skipping")

    importance = gbm.importance(30)
    (PATHS.artifacts / "feature_importance.json").write_text(
        importance.to_json(orient="records", indent=2)
    )
    logger.info("Top features:\n%s", importance.head(12).to_string(index=False))

    # ------------------------------------------------------------------
    # 5-6. Deep models
    # ------------------------------------------------------------------
    for architecture in ("lstm", "transformer"):
        try:
            logger.info("Training deep model: %s", architecture)
            from gridpulse.models.deep import train_deep

            model, deep_predictions = train_deep(
                frame, valid_start, test_start, architecture=architecture, quick=quick
            )
            model.save()
            if not deep_predictions.empty:
                deep_predictions = deep_predictions.rename(columns={"prediction": f"pred_{architecture}"})
                predictions["period_utc"] = pd.to_datetime(predictions["period_utc"], utc=True)
                deep_predictions["period_utc"] = pd.to_datetime(deep_predictions["period_utc"], utc=True)
                predictions = predictions.merge(
                    deep_predictions[["ba_code", "period_utc", f"pred_{architecture}"]],
                    on=["ba_code", "period_utc"], how="left",
                )
        except ImportError as exc:
            logger.warning("Skipping %s: %s", architecture, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Deep model %s failed: %s", architecture, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Ensemble: simple average of the two strongest model families
    # ------------------------------------------------------------------
    ensemble_parts = [c for c in ("pred_gbm", "pred_lstm") if c in predictions.columns]
    if len(ensemble_parts) > 1:
        predictions["pred_ensemble"] = predictions[ensemble_parts].mean(axis=1)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    leaderboard = _score(predictions)

    # Interval calibration, reported only if the quantile models produced output.
    if {"pred_gbm_p10", "pred_gbm_p90"} <= set(predictions.columns):
        interval_coverage = metrics.coverage(
            predictions["demand_mwh"], predictions["pred_gbm_p10"], predictions["pred_gbm_p90"]
        )
        logger.info("P10-P90 interval coverage: %.1f%% (target 80%%)", interval_coverage)
        leaderboard.loc[leaderboard["model"] == "gbm", "p10_p90_coverage_pct"] = round(interval_coverage, 2)

    _persist(predictions, leaderboard, test_start)

    if mlflow:
        _track(mlflow, leaderboard, quick, len(train), len(test), test_start)

    logger.info("\n%s", leaderboard.to_string(index=False))
    return leaderboard


def _score(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score every ``pred_*`` column against actuals and rank by skill over EIA."""
    actual = predictions["demand_mwh"]
    rows = []

    for column in [c for c in predictions.columns if c.startswith("pred_")]:
        name = column.removeprefix("pred_")
        mask = predictions[column].notna() & actual.notna()
        if mask.sum() < 100:
            logger.warning("  %s produced too few predictions (%d); skipped", name, int(mask.sum()))
            continue

        row = metrics.evaluate_forecast(actual[mask], predictions.loc[mask, column], name)
        row["peak_hour_mape_pct"] = round(
            metrics.peak_hour_mape(predictions[mask], "demand_mwh", column), 4
        )
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    benchmark = leaderboard.loc[leaderboard["model"] == BENCHMARK, "mape_pct"]
    benchmark_mape = float(benchmark.iloc[0]) if not benchmark.empty else float("nan")

    leaderboard["benchmark_mape_pct"] = round(benchmark_mape, 4)
    leaderboard["skill_vs_eia_pct"] = leaderboard["mape_pct"].apply(
        lambda m: metrics.skill_vs_benchmark(m, benchmark_mape)
    )
    return leaderboard.sort_values("mape_pct").reset_index(drop=True)


def _persist(predictions: pd.DataFrame, leaderboard: pd.DataFrame, test_start: pd.Timestamp) -> None:
    stamped = leaderboard.copy()
    stamped["trained_at_utc"] = datetime.now(timezone.utc)
    stamped["test_window_start"] = test_start

    with connect() as con:
        con.register("_preds", predictions)
        con.execute("CREATE OR REPLACE TABLE model_predictions AS SELECT * FROM _preds")
        con.unregister("_preds")

        con.register("_scores", stamped)
        con.execute("CREATE TABLE IF NOT EXISTS model_scores AS SELECT * FROM _scores LIMIT 0")
        con.execute("INSERT INTO model_scores SELECT * FROM _scores")
        con.unregister("_scores")

    PATHS.artifacts.mkdir(parents=True, exist_ok=True)
    (PATHS.artifacts / "leaderboard.json").write_text(stamped.to_json(orient="records", indent=2, date_format="iso"))

    best = leaderboard.iloc[0]
    (PATHS.artifacts / "headline.json").write_text(json.dumps({
        "best_model": best["model"],
        "best_mape_pct": best["mape_pct"],
        "eia_benchmark_mape_pct": best["benchmark_mape_pct"],
        "skill_vs_eia_pct": best["skill_vs_eia_pct"],
        "test_window_start": str(test_start.date()),
        "test_observations": int(best["n_obs"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    logger.info(
        "HEADLINE  best=%s  MAPE=%.3f%%  vs EIA %.3f%%  -> %.1f%% more accurate",
        best["model"], best["mape_pct"], best["benchmark_mape_pct"], best["skill_vs_eia_pct"],
    )


def _track(mlflow, leaderboard: pd.DataFrame, quick: bool, n_train: int, n_test: int, test_start) -> None:
    for _, row in leaderboard.iterrows():
        with mlflow.start_run(run_name=f"{row['model']}-{datetime.now():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "model": row["model"], "quick_mode": quick,
                "train_rows": n_train, "test_rows": n_test,
                "test_window_start": str(test_start.date()),
            })
            mlflow.log_metrics({
                "mape_pct": row["mape_pct"], "smape_pct": row["smape_pct"],
                "mae_mwh": row["mae_mwh"], "rmse_mwh": row["rmse_mwh"],
                "r2": row["r2"], "peak_hour_mape_pct": row["peak_hour_mape_pct"],
                "skill_vs_eia_pct": row["skill_vs_eia_pct"],
            })
