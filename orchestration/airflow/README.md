# Airflow deployment

Dagster is the primary orchestrator for GridPulse (see
`orchestration/dagster_app/definitions.py`). This directory mirrors the same
pipeline as an Airflow DAG for teams standardised on Airflow.

## Run with Docker

```bash
cd orchestration/airflow
docker compose up
# UI at http://localhost:8080  (user: airflow, password: airflow)
```

## Run locally (Linux, macOS or WSL2)

Airflow does not run natively on Windows; use WSL2 or the Docker option above.

```bash
export AIRFLOW_HOME=$(pwd)/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

AIRFLOW_VERSION=2.10.4
PY=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PY}.txt"

airflow standalone
```

## Task graph

```
eia_ingest ──────┐
                 ├─► build_warehouse ──┬─► data_quality ──┬─► train_models ─────┐
weather_ingest ──┘                     │                  │                     ├─► export_app
                                       └─► dbt_marts      └─► detect_anomalies ─┘
```

Retries use exponential backoff, capped at 30 minutes. `data_quality` raises on a
critical check failure, which halts the run before a bad model is trained and
exported to the public site.
