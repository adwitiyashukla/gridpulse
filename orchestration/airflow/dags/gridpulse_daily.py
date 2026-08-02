"""Airflow mirror of the GridPulse Dagster asset graph.

Dagster is the primary orchestrator for this project (asset-based lineage suits a
lakehouse, and it is far lighter on constrained hardware). This DAG exists because
a large share of production data platforms are standardised on Airflow, and the
same pipeline should be portable to either.

The task graph is identical to the Dagster asset graph:

    eia_ingest ──┐
                 ├─► build_warehouse ──┬─► data_quality ──┬─► train_models ──┐
    weather_ingest┘                    │                  │                  ├─► export_app
                                       ├─► dbt_marts      └─► detect_anomalies┘
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_ARGS = {
    "owner": "gridpulse",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
    "depends_on_past": False,
}


@dag(
    dag_id="gridpulse_daily_refresh",
    description="Extract EIA and weather, rebuild the warehouse, validate, retrain, export.",
    default_args=DEFAULT_ARGS,
    # 06:00 UTC: EIA has published the previous full day by then.
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["gridpulse", "energy", "forecasting", "elt"],
    doc_md=__doc__,
)
def gridpulse_daily_refresh():

    @task(task_id="eia_ingest", doc_md="Incremental EIA-930 extract, watermark driven.")
    def eia_ingest() -> dict:
        from gridpulse.ingestion import ingest_eia

        return ingest_eia()

    @task(task_id="weather_ingest", doc_md="ERA5 archive plus forecast, stitched.")
    def weather_ingest() -> dict:
        from gridpulse.ingestion import ingest_weather

        return ingest_weather()

    @task(task_id="build_warehouse", doc_md="Bronze to silver to the gold star schema.")
    def build_warehouse(eia: dict, weather: dict) -> dict:
        from gridpulse.warehouse import build_warehouse as build

        return build()

    @task(task_id="data_quality", doc_md="Thirteen checks across six quality dimensions.")
    def data_quality(_: dict) -> dict:
        from gridpulse.quality import run_quality_suite

        report = run_quality_suite()
        if not report.passed:
            failed = [r.check.name for r in report.results
                      if not r.passed and r.check.severity.value == "critical"]
            raise ValueError(f"Critical data quality checks failed: {failed}")
        return {"score": report.score}

    dbt_marts = BashOperator(
        task_id="dbt_marts",
        bash_command=f"cd {REPO_ROOT / 'dbt' / 'gridpulse'} && dbt build --profiles-dir .",
        doc_md="Build and test the dbt analytics marts.",
    )

    @task(task_id="train_models", doc_md="Baselines, LightGBM, LSTM, Transformer, ensemble.")
    def train_models(_: dict) -> dict:
        from gridpulse.models.pipeline import train_all

        leaderboard = train_all()
        best = leaderboard.iloc[0]
        return {
            "best_model": str(best["model"]),
            "mape_pct": float(best["mape_pct"]),
            "skill_vs_eia_pct": float(best["skill_vs_eia_pct"]),
        }

    @task(task_id="detect_anomalies", doc_md="Three-detector consensus anomaly scoring.")
    def detect_anomalies(_: dict) -> dict:
        from gridpulse.models.anomaly import run_anomaly_detection

        frame = run_anomaly_detection()
        return {"anomalies": int(frame["is_anomaly"].sum())}

    @task(task_id="export_app", doc_md="Write the slim DuckDB artifact for the public app.")
    def export_app(models: dict, anomalies: dict) -> str:
        from gridpulse.warehouse.export import export_for_app

        return str(export_for_app())

    eia = eia_ingest()
    weather = weather_ingest()
    warehouse = build_warehouse(eia, weather)
    quality = data_quality(warehouse)

    warehouse >> dbt_marts

    models = train_models(quality)
    anomalies = detect_anomalies(quality)
    export_app(models, anomalies)


gridpulse_daily_refresh()
