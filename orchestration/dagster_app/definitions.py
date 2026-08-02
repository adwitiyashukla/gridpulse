"""Dagster software-defined assets for the GridPulse platform.

Dagster is modelled around *assets* -- the tables and files that must exist --
rather than *tasks* that must run. That maps naturally onto a lakehouse: each
node below is a real materialised artifact, so the UI lineage graph is a literal
picture of the warehouse rather than a schedule diagram.

Run the UI locally::

    dagster dev -f orchestration/dagster_app/definitions.py

An equivalent Airflow DAG lives in ``orchestration/airflow/dags/`` for teams
standardised on Airflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetSelection,
    Definitions,
    MetadataValue,
    Output,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

GROUP_EXTRACT = "01_extract"
GROUP_WAREHOUSE = "02_warehouse"
GROUP_QUALITY = "03_quality"
GROUP_ML = "04_machine_learning"
GROUP_SERVE = "05_serving"


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
@asset(
    group_name=GROUP_EXTRACT,
    compute_kind="python",
    description="Hourly demand, EIA's day-ahead forecast, net generation and interchange "
                "for every configured balancing authority. Incremental via stored watermarks.",
)
def eia_bronze(context: AssetExecutionContext) -> Output[dict]:
    from gridpulse.ingestion import ingest_eia

    counts = ingest_eia()
    total = sum(counts.values())
    return Output(
        counts,
        metadata={
            "total_rows": MetadataValue.int(total),
            "balancing_authorities": MetadataValue.int(len(counts)),
            "rows_per_ba": MetadataValue.json(counts),
        },
    )


@asset(
    group_name=GROUP_EXTRACT,
    compute_kind="python",
    description="Hourly weather for each BA's load centre. Stitches the ERA5 archive "
                "(authoritative, ~5 day lag) with the forecast endpoint (recent + future).",
)
def weather_bronze(context: AssetExecutionContext) -> Output[dict]:
    from gridpulse.ingestion import ingest_weather

    counts = ingest_weather()
    return Output(
        counts,
        metadata={
            "total_rows": MetadataValue.int(sum(counts.values())),
            "rows_per_ba": MetadataValue.json(counts),
        },
    )


# ---------------------------------------------------------------------------
# Warehouse
# ---------------------------------------------------------------------------
@asset(
    group_name=GROUP_WAREHOUSE,
    compute_kind="duckdb",
    deps=[eia_bronze, weather_bronze],
    description="Silver conformed layer plus the gold Kimball star schema: dim_ba, "
                "dim_date, fact_demand_hourly and fact_forecast_accuracy.",
)
def gold_warehouse(context: AssetExecutionContext) -> Output[dict]:
    from gridpulse.warehouse import build_warehouse

    counts = build_warehouse()
    return Output(
        counts,
        metadata={
            "tables": MetadataValue.int(len(counts)),
            "row_counts": MetadataValue.json(counts),
            "fact_rows": MetadataValue.int(counts.get("fact_demand_hourly", 0)),
        },
    )


@asset(
    group_name=GROUP_WAREHOUSE,
    compute_kind="dbt",
    deps=[gold_warehouse],
    description="dbt analytics marts: daily demand, load profiles, EIA forecast "
                "scorecard, temperature response curves and peak events.",
)
def dbt_marts(context: AssetExecutionContext) -> Output[str]:
    import subprocess

    project = REPO_ROOT / "dbt" / "gridpulse"
    result = subprocess.run(
        ["dbt", "build", "--profiles-dir", "."],
        cwd=project, capture_output=True, text=True,
    )
    context.log.info(result.stdout[-4000:])
    if result.returncode != 0:
        context.log.error(result.stderr[-4000:])
        raise RuntimeError(f"dbt build failed with exit code {result.returncode}")

    return Output(
        "ok",
        metadata={"dbt_output": MetadataValue.text(result.stdout[-2000:])},
    )


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
@asset(
    group_name=GROUP_QUALITY,
    compute_kind="python",
    deps=[gold_warehouse],
    description="Thirteen declarative quality checks across completeness, validity, "
                "consistency, timeliness, uniqueness and accuracy.",
)
def data_quality(context: AssetExecutionContext) -> Output[dict]:
    from gridpulse.quality import run_quality_suite

    report = run_quality_suite()
    failures = [r.check.name for r in report.results if not r.passed]
    if failures:
        context.log.warning("Checks failed: %s", ", ".join(failures))

    return Output(
        {"score": report.score, "passed": report.passed, "failures": failures},
        metadata={
            "quality_score_pct": MetadataValue.float(report.score),
            "suite_passed": MetadataValue.bool(report.passed),
            "checks_run": MetadataValue.int(len(report.results)),
            "failed_checks": MetadataValue.json(failures),
            "results": MetadataValue.md(report.to_frame().to_markdown(index=False)),
        },
    )


@asset_check(asset=gold_warehouse, description="The fact table is non-empty and covers every configured BA.")
def fact_table_populated() -> AssetCheckResult:
    from gridpulse.config import active_bas
    from gridpulse.warehouse.duck import query

    frame = query("SELECT count(*) AS rows, count(DISTINCT ba_code) AS bas FROM fact_demand_hourly")
    rows = int(frame.iloc[0]["rows"])
    bas = int(frame.iloc[0]["bas"])
    expected = len(active_bas())

    return AssetCheckResult(
        passed=rows > 0 and bas == expected,
        metadata={"rows": rows, "balancing_authorities": bas, "expected": expected},
    )


@asset_check(asset=gold_warehouse, description="EIA's benchmark forecast is present for scoring.")
def benchmark_present() -> AssetCheckResult:
    from gridpulse.warehouse.duck import query

    rows = int(query("SELECT count(*) AS n FROM fact_forecast_accuracy").iloc[0]["n"])
    return AssetCheckResult(passed=rows > 1000, metadata={"benchmark_rows": rows})


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------
@asset(
    group_name=GROUP_ML,
    compute_kind="python",
    deps=[gold_warehouse, data_quality],
    description="Trains baselines, LightGBM with quantiles, LSTM and Transformer, then "
                "scores all of them against EIA's own published day-ahead forecast.",
)
def forecasting_models(context: AssetExecutionContext) -> Output[dict]:
    from gridpulse.models.pipeline import train_all

    leaderboard = train_all()
    best = leaderboard.iloc[0]

    return Output(
        leaderboard.to_dict(orient="records"),
        metadata={
            "best_model": MetadataValue.text(str(best["model"])),
            "best_mape_pct": MetadataValue.float(float(best["mape_pct"])),
            "eia_benchmark_mape_pct": MetadataValue.float(float(best["benchmark_mape_pct"])),
            "skill_vs_eia_pct": MetadataValue.float(float(best["skill_vs_eia_pct"])),
            "leaderboard": MetadataValue.md(leaderboard.to_markdown(index=False)),
        },
    )


@asset(
    group_name=GROUP_ML,
    compute_kind="python",
    deps=[gold_warehouse],
    description="Three-detector anomaly consensus: robust seasonal z-score, Isolation "
                "Forest and a daily-load-shape autoencoder.",
)
def anomaly_scores(context: AssetExecutionContext) -> Output[dict]:
    from gridpulse.models.anomaly import run_anomaly_detection

    frame = run_anomaly_detection()
    flagged = int(frame["is_anomaly"].sum())

    return Output(
        {"hours_scored": len(frame), "anomalies": flagged},
        metadata={
            "hours_scored": MetadataValue.int(len(frame)),
            "anomalies_found": MetadataValue.int(flagged),
            "anomaly_rate_pct": MetadataValue.float(round(100 * flagged / max(len(frame), 1), 3)),
            "by_type": MetadataValue.json(
                frame.loc[frame["is_anomaly"], "anomaly_type"].value_counts().to_dict()
            ),
        },
    )


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
@asset(
    group_name=GROUP_SERVE,
    compute_kind="python",
    deps=[forecasting_models, anomaly_scores, data_quality],
    description="Slim DuckDB artifact the public Streamlit app ships with.",
)
def app_export(context: AssetExecutionContext) -> Output[str]:
    from gridpulse.warehouse.export import export_for_app

    path = export_for_app()
    size_mb = path.stat().st_size / 1e6

    return Output(
        str(path),
        metadata={
            "path": MetadataValue.path(str(path)),
            "size_mb": MetadataValue.float(round(size_mb, 2)),
        },
    )


# ---------------------------------------------------------------------------
# Jobs and schedules
# ---------------------------------------------------------------------------
daily_refresh = define_asset_job(
    name="daily_refresh",
    selection=AssetSelection.all(),
    description="Full platform refresh: extract, transform, validate, retrain, export.",
)

incremental_refresh = define_asset_job(
    name="incremental_refresh",
    selection=AssetSelection.assets(eia_bronze, weather_bronze, gold_warehouse, data_quality),
    description="Data-only refresh without retraining. Cheap enough to run hourly.",
)

# EIA publishes on a lag, so 06:00 UTC comfortably captures the previous full day.
daily_schedule = ScheduleDefinition(
    job=daily_refresh,
    cron_schedule="0 6 * * *",
    execution_timezone="UTC",
    description="Nightly full refresh after EIA publishes the previous day.",
)

hourly_schedule = ScheduleDefinition(
    job=incremental_refresh,
    cron_schedule="15 * * * *",
    execution_timezone="UTC",
    description="Hourly top-up of grid and weather data.",
    default_status=None,
)

defs = Definitions(
    assets=[
        eia_bronze, weather_bronze, gold_warehouse, dbt_marts,
        data_quality, forecasting_models, anomaly_scores, app_export,
    ],
    asset_checks=[fact_table_populated, benchmark_present],
    jobs=[daily_refresh, incremental_refresh],
    schedules=[daily_schedule, hourly_schedule],
)
