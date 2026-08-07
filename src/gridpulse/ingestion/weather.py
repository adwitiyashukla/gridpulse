"""Downloads weather from Open-Meteo, no API key needed.

Joins the ERA5 archive, which lags about 5 days, to the forecast endpoint, which
covers the gap and supplies tomorrow's weather. Overlapping hours use the archive.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

from gridpulse.config import PATHS, SETTINGS, WEATHER_VARIABLES, BalancingAuthority, active_bas
from gridpulse.ingestion.http import fetch_json

logger = logging.getLogger(__name__)

ARCHIVE_ROUTE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_ROUTE = "https://api.open-meteo.com/v1/forecast"

BRONZE_SUBDIR = "weather"
ARCHIVE_LAG_DAYS = 6
FORECAST_PAST_DAYS = 92
FORECAST_AHEAD_DAYS = 16

INTER_REQUEST_PAUSE = 2.0


def bronze_path(ba_code: str) -> Path:
    return PATHS.bronze / BRONZE_SUBDIR / f"ba={ba_code}" / "data.parquet"


def read_bronze(ba_code: str) -> pd.DataFrame:
    path = bronze_path(ba_code)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _flatten(payload: dict, ba: BalancingAuthority, source: str) -> pd.DataFrame:
    """Open-Meteo returns column-oriented arrays; pivot them into tidy rows."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()

    frame = pd.DataFrame({"period_utc": pd.to_datetime(times, utc=True)})
    for variable in WEATHER_VARIABLES:
        frame[variable] = pd.to_numeric(pd.Series(hourly.get(variable, [None] * len(times))), errors="coerce")

    frame["ba_code"] = ba.code
    frame["source"] = source
    frame["ingested_at_utc"] = pd.Timestamp.now(tz="UTC")
    return frame


async def _fetch_archive(
    client: httpx.AsyncClient, ba: BalancingAuthority, start: str, end: str, sem: asyncio.Semaphore
) -> pd.DataFrame:
    if pd.Timestamp(start) > pd.Timestamp(end):
        return pd.DataFrame()
    payload = await fetch_json(
        client,
        ARCHIVE_ROUTE,
        {
            "latitude": ba.latitude,
            "longitude": ba.longitude,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "UTC",
        },
        semaphore=sem,
        label=f"Weather-archive:{ba.code}",
    )
    return _flatten(payload, ba, "era5_archive")


async def _fetch_forecast(
    client: httpx.AsyncClient, ba: BalancingAuthority, sem: asyncio.Semaphore
) -> pd.DataFrame:
    payload = await fetch_json(
        client,
        FORECAST_ROUTE,
        {
            "latitude": ba.latitude,
            "longitude": ba.longitude,
            "hourly": ",".join(WEATHER_VARIABLES),
            "past_days": FORECAST_PAST_DAYS,
            "forecast_days": FORECAST_AHEAD_DAYS,
            "timezone": "UTC",
        },
        semaphore=sem,
        label=f"Weather-forecast:{ba.code}",
    )
    return _flatten(payload, ba, "forecast")


def _merge(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine sources, preferring the archive where both cover the same hour."""
    populated = [f for f in frames if f is not None and not f.empty]
    if not populated:
        return pd.DataFrame()

    combined = pd.concat(populated, ignore_index=True)
    combined["period_utc"] = pd.to_datetime(combined["period_utc"], utc=True)
    combined["_priority"] = (combined["source"] == "era5_archive").astype(int)
    combined = (
        combined.sort_values(["period_utc", "_priority"])
        .drop_duplicates(subset=["ba_code", "period_utc"], keep="last")
        .drop(columns="_priority")
        .sort_values("period_utc")
        .reset_index(drop=True)
    )
    return combined


async def _ingest_async(bas: list[BalancingAuthority], full_refresh: bool) -> dict[str, int]:
    sem = asyncio.Semaphore(1)
    today = datetime.now(timezone.utc).date()
    archive_end = today - timedelta(days=ARCHIVE_LAG_DAYS)
    written: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(SETTINGS.request_timeout), follow_redirects=True) as client:
        for ba in bas:
            existing = pd.DataFrame() if full_refresh else read_bronze(ba.code)

            if existing.empty:
                archive_start = SETTINGS.start_date
            else:
                archived = existing.loc[existing["source"] == "era5_archive", "period_utc"]
                archive_start = (
                    (pd.to_datetime(archived, utc=True).max().date() - timedelta(days=1)).isoformat()
                    if not archived.empty
                    else SETTINGS.start_date
                )

            archive = await _fetch_archive(client, ba, archive_start, archive_end.isoformat(), sem)
            await asyncio.sleep(INTER_REQUEST_PAUSE)
            forecast = await _fetch_forecast(client, ba, sem)
            await asyncio.sleep(INTER_REQUEST_PAUSE)

            merged = _merge(existing, archive, forecast)

            path = bronze_path(ba.code)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(path, index=False, compression="snappy")
            written[ba.code] = len(merged)
            logger.info(
                "  %-5s %7d rows (%s -> %s)",
                ba.code, len(merged),
                merged["period_utc"].min().date() if len(merged) else "-",
                merged["period_utc"].max().date() if len(merged) else "-",
            )

    return written


def ingest_weather(ba_codes: list[str] | None = None, full_refresh: bool = False) -> dict[str, int]:
    """Extract hourly weather for every active BA's load centre into bronze."""
    PATHS.ensure()
    bas = active_bas()
    if ba_codes:
        wanted = {c.upper() for c in ba_codes}
        bas = [b for b in bas if b.code in wanted]

    logger.info(
        "Ingesting weather for %d BA(s) [%s]",
        len(bas), "FULL REFRESH" if full_refresh else "incremental",
    )
    result = asyncio.run(_ingest_async(bas, full_refresh))
    logger.info("Weather ingestion complete: %s rows total", f"{sum(result.values()):,}")
    return result
