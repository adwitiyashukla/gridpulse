"""GridPulse REST API.

A thin, documented HTTP surface over the warehouse, the trained models and the
analytics agent. FastAPI generates OpenAPI docs at ``/docs`` automatically, so the
service is self-describing.

Run locally::

    uvicorn gridpulse.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gridpulse.config import BALANCING_AUTHORITIES, FORECAST_HORIZON, PATHS
from gridpulse.warehouse.duck import connect

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GridPulse API",
    description=(
        "Day-ahead electricity demand forecasting for US balancing authorities, "
        "benchmarked against the EIA's own published forecast."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _database():
    slim = PATHS.gold / "gridpulse_app.duckdb"
    return slim if slim.exists() else PATHS.duckdb


def _query(sql: str, params: list | None = None):
    try:
        with connect(_database(), read_only=True) as con:
            return con.execute(sql, params or []).df()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Warehouse unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    warehouse_present: bool
    models_present: bool
    balancing_authorities: int
    server_time_utc: str


class ForecastRequest(BaseModel):
    ba_code: str = Field(..., description="Balancing authority code, e.g. PJM", examples=["PJM"])
    horizon_hours: int = Field(FORECAST_HORIZON, ge=1, le=48)
    allow_network: bool = Field(True, description="Fetch live weather; falls back to replay if unavailable")


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Which balancing authority has the highest average demand?"])
    summarise: bool = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {"service": "GridPulse API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    from gridpulse.models.inference import artifacts_available

    return HealthResponse(
        status="ok",
        warehouse_present=_database().exists(),
        models_present=artifacts_available(),
        balancing_authorities=len(BALANCING_AUTHORITIES),
        server_time_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/balancing-authorities", tags=["reference"])
def list_bas() -> list[dict[str, Any]]:
    """Every balancing authority the platform covers."""
    return [
        {
            "code": ba.code, "name": ba.name, "region": ba.region,
            "load_centre": ba.load_centre, "timezone": ba.timezone,
            "latitude": ba.latitude, "longitude": ba.longitude,
        }
        for ba in BALANCING_AUTHORITIES.values()
    ]


@app.get("/demand/{ba_code}", tags=["data"])
def demand(
    ba_code: str,
    hours: int = Query(168, ge=1, le=8760, description="How many recent hours to return"),
) -> dict[str, Any]:
    """Recent observed demand, weather and EIA's forecast for one BA."""
    code = ba_code.upper()
    if code not in BALANCING_AUTHORITIES:
        raise HTTPException(404, f"Unknown balancing authority '{ba_code}'")

    frame = _query(
        """
        SELECT period_utc, demand_mwh, demand_forecast_mwh, temperature_2m
        FROM fact_demand_hourly WHERE ba_code = ?
        ORDER BY period_utc DESC LIMIT ?
        """,
        [code, hours],
    ).sort_values("period_utc")

    if frame.empty:
        raise HTTPException(404, f"No stored data for {code}")

    frame["period_utc"] = frame["period_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"ba_code": code, "rows": len(frame), "data": frame.to_dict(orient="records")}


@app.post("/forecast", tags=["models"])
def make_forecast(request: ForecastRequest) -> dict[str, Any]:
    """Generate a 24-hour-ahead demand forecast with P10/P90 bands."""
    from gridpulse.models.inference import forecast as run_forecast

    try:
        result = run_forecast(
            request.ba_code, horizon=request.horizon_hours, allow_network=request.allow_network
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model artifacts missing: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc

    return {
        "ba_code": result.ba_code,
        "model": result.model,
        "mode": result.mode,
        "generated_at_utc": result.generated_at_utc.isoformat(),
        "notes": result.notes or [],
        "forecast": result.to_records(),
    }


@app.get("/leaderboard", tags=["models"])
def leaderboard() -> dict[str, Any]:
    """Model accuracy versus EIA's own published day-ahead forecast."""
    frame = _query("""
        SELECT * FROM model_scores
        WHERE trained_at_utc = (SELECT max(trained_at_utc) FROM model_scores)
        ORDER BY mape_pct
    """)
    if frame.empty:
        raise HTTPException(404, "No model scores yet. Run `gridpulse train`.")
    return {"models": frame.to_dict(orient="records")}


@app.get("/forecast-accuracy", tags=["models"])
def forecast_accuracy() -> dict[str, Any]:
    """EIA's own forecast error, aggregated per balancing authority."""
    frame = _query("""
        SELECT ba_code,
               round(avg(abs_pct_error), 3) AS eia_mape_pct,
               round(avg(error_mwh), 1)     AS mean_bias_mwh,
               count(*)                     AS hours
        FROM fact_forecast_accuracy
        GROUP BY ba_code ORDER BY eia_mape_pct
    """)
    return {"by_balancing_authority": frame.to_dict(orient="records")}


@app.get("/anomalies", tags=["monitoring"])
def anomalies(
    severity: str = Query("high", pattern="^(low|medium|high|all)$"),
    limit: int = Query(200, ge=1, le=5000),
) -> dict[str, Any]:
    """Recently detected grid anomalies."""
    clause = "" if severity == "all" else "AND severity = ?"
    params: list = [limit] if severity == "all" else [severity, limit]
    frame = _query(
        f"""
        SELECT period_utc, ba_code, demand_mwh, temperature_2m,
               anomaly_type, severity, detector_votes, round(robust_z, 2) AS robust_z
        FROM anomaly_scores WHERE is_anomaly {clause}
        ORDER BY period_utc DESC LIMIT ?
        """,
        params,
    )
    if not frame.empty:
        frame["period_utc"] = frame["period_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"severity": severity, "count": len(frame), "anomalies": frame.to_dict(orient="records")}


@app.get("/data-quality", tags=["monitoring"])
def data_quality() -> dict[str, Any]:
    """Latest data quality scorecard."""
    scorecard = _query("SELECT * FROM dq_scorecard ORDER BY dimension")
    detail = _query("""
        SELECT check_name, dimension, severity, failed_rows, total_rows,
               failure_rate_pct, passed
        FROM dq_results
        WHERE run_at_utc = (SELECT max(run_at_utc) FROM dq_results)
        ORDER BY passed, severity
    """)
    return {
        "scorecard": scorecard.to_dict(orient="records"),
        "checks": detail.to_dict(orient="records"),
    }


@app.post("/ask", tags=["ai"])
def ask(request: AskRequest) -> dict[str, Any]:
    """Natural-language question answered by generating and running guarded SQL."""
    from gridpulse.agent import GridAgent

    agent = GridAgent(database=_database())
    if not agent.available:
        raise HTTPException(503, "AI agent disabled: no GROQ_API_KEY configured.")

    answer = agent.ask(request.question, summarise=request.summarise)
    if not answer.ok:
        raise HTTPException(400, answer.error or "Agent failed")

    return {
        "question": answer.question,
        "sql": answer.sql,
        "summary": answer.summary,
        "row_count": len(answer.data),
        "warnings": answer.warnings,
        "data": answer.data.head(500).to_dict(orient="records"),
    }
