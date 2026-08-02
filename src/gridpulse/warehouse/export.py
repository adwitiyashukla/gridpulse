"""Build the slim deployment artifact the public app ships with.

The development warehouse holds years of hourly history across every table. Git and
free-tier hosting do not want that. This module distils it into a compact DuckDB
file containing only what the public site actually reads:

* a rolling window of recent hourly demand, weather and forecasts,
* pre-computed model predictions over the evaluation window,
* the model leaderboard and data quality scorecard,
* flagged anomalies.

The app therefore starts instantly with no retraining and no warehouse dependency,
while still calling the live EIA API for anything newer than the last export.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from gridpulse.config import PATHS, WEATHER_VARIABLES
from gridpulse.warehouse.duck import connect, row_count

logger = logging.getLogger(__name__)

APP_DB_NAME = "gridpulse_app.duckdb"
EXPORT_WINDOW_DAYS = 400  # a little over a year, so the app can show YoY comparisons


def export_for_app(window_days: int = EXPORT_WINDOW_DAYS, destination: Path | None = None) -> Path:
    """Write the slim app database and its manifest. Returns the database path."""
    target = Path(destination) if destination else PATHS.gold / APP_DB_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    wal = target.with_suffix(target.suffix + ".wal")
    if wal.exists():
        wal.unlink()

    source = PATHS.duckdb
    if not source.exists():
        raise FileNotFoundError(f"Warehouse not found at {source}. Run `gridpulse build` first.")

    manifest: dict[str, int] = {}

    with connect(target) as con:
        con.execute(f"ATTACH '{source.as_posix()}' AS wh (READ_ONLY)")

        # Dimensions are small; copy wholesale.
        for table in ("dim_ba", "dim_date"):
            if _exists_in(con, "wh", table):
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM wh.{table}")
                manifest[table] = row_count(con, table)

        # The main fact, trimmed to the rolling window. The column list is derived
        # from WEATHER_VARIABLES rather than hand-written: a hand-written list
        # silently drops columns the feature builder needs, and the failure only
        # surfaces at inference time on the deployed site.
        if _exists_in(con, "wh", "fact_demand_hourly"):
            columns = ", ".join([
                "period_utc", "ba_code", "date_local", "hour_local",
                "demand_mwh", "demand_forecast_mwh", "net_generation_mwh",
                "total_interchange_mwh", "demand_clean_mwh",
                "flag_isolated_spike", "flag_implausible_magnitude",
                *WEATHER_VARIABLES,
                "day_of_week", "is_weekend", "is_holiday", "is_business_day",
                "is_day_before_holiday", "is_day_after_holiday",
                "season", "month", "year",
            ])
            con.execute(f"""
                CREATE TABLE fact_demand_hourly AS
                SELECT {columns}
                FROM wh.fact_demand_hourly
                WHERE period_utc >= (SELECT max(period_utc) FROM wh.fact_demand_hourly)
                                     - INTERVAL {window_days} DAY
            """)
            manifest["fact_demand_hourly"] = row_count(con, "fact_demand_hourly")

        for table in ("fact_forecast_accuracy", "model_predictions", "anomaly_scores"):
            if _exists_in(con, "wh", table):
                con.execute(f"""
                    CREATE TABLE {table} AS SELECT * FROM wh.{table}
                    WHERE period_utc >= (SELECT max(period_utc) FROM wh.{table})
                                         - INTERVAL {window_days} DAY
                """)
                manifest[table] = row_count(con, table)

        for table in ("model_scores", "dq_results", "dq_scorecard"):
            if _exists_in(con, "wh", table):
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM wh.{table}")
                manifest[table] = row_count(con, table)

        con.execute("DETACH wh")
        con.execute("VACUUM")

    size_mb = target.stat().st_size / 1e6
    (PATHS.artifacts / "export_manifest.json").write_text(json.dumps({
        "database": target.name,
        "size_mb": round(size_mb, 2),
        "window_days": window_days,
        "tables": manifest,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    logger.info("App database written: %s (%.1f MB)", target, size_mb)
    for table, count in manifest.items():
        logger.info("  %-26s %10s rows", table, f"{count:,}")
    if size_mb > 90:
        logger.warning(
            "Export is %.0f MB, close to GitHub's 100 MB file limit. "
            "Re-run with a smaller --window-days or enable Git LFS.", size_mb,
        )
    return target


def _exists_in(con, schema: str, table: str) -> bool:
    found = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
        [schema, table],
    ).fetchone()
    if found and found[0]:
        return True
    # DuckDB reports attached databases via the catalog column in some versions.
    found = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_catalog = ? AND table_name = ?",
        [schema, table],
    ).fetchone()
    return bool(found and found[0])
