"""Downloading the hourly grid data from the EIA API.

For each region I pull four hourly series from the ``region-data`` endpoint:

==========  ====================================================================
``D``       Demand in MWh, which is what I am trying to predict
``DF``      EIA's own day-ahead forecast, which is what I compare against
``NG``      Net generation in MWh
``TI``      Power traded with neighbouring regions, in MWh
==========  ====================================================================

The ``DF`` series is the important one. Instead of making up my own easy baseline,
I score every model against the forecast the US government actually published and
actually ran the grid against that day.

Downloads only fetch what is new. Each run looks at the latest timestamp already
saved and asks for periods after that, so a scheduled run costs a handful of
requests rather than downloading everything again. Running it twice does not create
duplicates.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

from gridpulse.config import EIA_MEASURES, PATHS, SETTINGS, BalancingAuthority, active_bas
from gridpulse.ingestion.http import fetch_json

logger = logging.getLogger(__name__)

EIA_BASE = "https://api.eia.gov/v2"
REGION_DATA_ROUTE = f"{EIA_BASE}/electricity/rto/region-data/data/"

# EIA-930 collection began 1 July 2015; requests before this return nothing.
EIA_EPOCH = "2015-07-01"

BRONZE_SUBDIR = "eia_region"
_SCHEMA = ["period_utc", "ba_code", "measure_code", "value_mwh", "ingested_at_utc"]


# ---------------------------------------------------------------------------
# Bronze layout helpers
# ---------------------------------------------------------------------------
def bronze_path(ba_code: str) -> Path:
    return PATHS.bronze / BRONZE_SUBDIR / f"ba={ba_code}" / "data.parquet"


def read_bronze(ba_code: str) -> pd.DataFrame:
    """Existing bronze rows for a BA, or an empty correctly-typed frame."""
    path = bronze_path(ba_code)
    if not path.exists():
        return pd.DataFrame(columns=_SCHEMA)
    return pd.read_parquet(path)


def watermark(ba_code: str) -> str | None:
    """Latest period already stored for a BA, as an EIA ``YYYY-MM-DDTHH`` string."""
    existing = read_bronze(ba_code)
    if existing.empty:
        return None
    latest = pd.to_datetime(existing["period_utc"], utc=True).max()
    return latest.strftime("%Y-%m-%dT%H")


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def _build_params(
    ba_code: str, start: str, end: str, offset: int, length: int
) -> list[tuple[str, str]]:
    """EIA v2 uses repeated bracketed keys, so params must be a list of tuples."""
    params: list[tuple[str, str]] = [
        ("api_key", SETTINGS.require_eia_key()),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", ba_code),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(length)),
    ]
    params += [("facets[type][]", code) for code in EIA_MEASURES]
    return params


def _normalise(records: list[dict]) -> pd.DataFrame:
    """Coerce raw EIA records into the bronze schema.

    EIA returns numerics as strings and occasionally omits ``value`` entirely for
    hours a BA failed to report, so both are handled defensively.
    """
    if not records:
        return pd.DataFrame(columns=_SCHEMA)

    df = pd.DataFrame(records)
    out = pd.DataFrame(
        {
            # EIA hourly periods are UTC, formatted YYYY-MM-DDTHH
            "period_utc": pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True, errors="coerce"),
            "ba_code": df["respondent"].astype("string"),
            "measure_code": df["type"].astype("string"),
            "value_mwh": pd.to_numeric(df.get("value"), errors="coerce"),
            "ingested_at_utc": pd.Timestamp.now(tz="UTC"),
        }
    )
    return out.dropna(subset=["period_utc", "ba_code", "measure_code"])


def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows in. If a row already exists, the newest download wins.

    Written this way so that running the download twice does not create duplicate
    rows, which matters because the scheduled job can overlap with a manual run.
    """
    if existing.empty:
        combined = fresh
    elif fresh.empty:
        combined = existing
    else:
        combined = pd.concat([existing, fresh], ignore_index=True)

    if combined.empty:
        return combined

    combined["period_utc"] = pd.to_datetime(combined["period_utc"], utc=True)
    combined = (
        combined.sort_values("ingested_at_utc")
        .drop_duplicates(subset=["period_utc", "ba_code", "measure_code"], keep="last")
        .sort_values(["period_utc", "measure_code"])
        .reset_index(drop=True)
    )
    return combined


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------
async def _fetch_ba(
    client: httpx.AsyncClient,
    ba: BalancingAuthority,
    start: str,
    end: str,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """Page through every row EIA holds for one BA across the requested window."""
    first = await fetch_json(
        client,
        REGION_DATA_ROUTE,
        _build_params(ba.code, start, end, offset=0, length=SETTINGS.page_size),
        semaphore=semaphore,
        label=f"EIA:{ba.code}",
    )

    payload = first.get("response", {})
    total = int(payload.get("total", 0) or 0)
    records = list(payload.get("data", []))

    if total == 0:
        logger.info("  %-5s no new rows", ba.code)
        return pd.DataFrame(columns=_SCHEMA)

    offsets = range(SETTINGS.page_size, total, SETTINGS.page_size)
    if offsets:
        pages = await asyncio.gather(
            *[
                fetch_json(
                    client,
                    REGION_DATA_ROUTE,
                    _build_params(ba.code, start, end, offset=off, length=SETTINGS.page_size),
                    semaphore=semaphore,
                    label=f"EIA:{ba.code}@{off}",
                )
                for off in offsets
            ]
        )
        for page in pages:
            records.extend(page.get("response", {}).get("data", []))

    logger.info("  %-5s %7d rows across %d page(s)", ba.code, len(records), 1 + len(offsets))
    return _normalise(records)


async def _ingest_async(bas: list[BalancingAuthority], full_refresh: bool) -> dict[str, int]:
    end = (datetime.now(timezone.utc) + timedelta(hours=48)).strftime("%Y-%m-%dT%H")
    semaphore = asyncio.Semaphore(SETTINGS.max_concurrency)
    written: dict[str, int] = {}

    timeout = httpx.Timeout(SETTINGS.request_timeout, connect=15.0)
    limits = httpx.Limits(max_connections=SETTINGS.max_concurrency + 2)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        for ba in bas:
            mark = None if full_refresh else watermark(ba.code)
            if mark:
                # Re-request the final stored day: EIA revises recent hours in place.
                start_ts = pd.Timestamp(mark, tz="UTC") - pd.Timedelta(hours=24)
                start = start_ts.strftime("%Y-%m-%dT%H")
            else:
                start = f"{max(SETTINGS.start_date, EIA_EPOCH)}T00"

            fresh = await _fetch_ba(client, ba, start, end, semaphore)
            existing = pd.DataFrame(columns=_SCHEMA) if full_refresh else read_bronze(ba.code)
            merged = _merge(existing, fresh)

            path = bronze_path(ba.code)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(path, index=False, compression="snappy")
            written[ba.code] = len(merged)

    return written


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def ingest_eia(ba_codes: list[str] | None = None, full_refresh: bool = False) -> dict[str, int]:
    """Extract EIA-930 hourly telemetry into the bronze zone.

    Parameters
    ----------
    ba_codes
        Restrict to these balancing authorities. Defaults to ``GRIDPULSE_BAS``.
    full_refresh
        Ignore the stored watermark and re-download the whole history.

    Returns
    -------
    dict
        Total bronze row count per BA after the merge.
    """
    PATHS.ensure()
    bas = active_bas()
    if ba_codes:
        wanted = {c.upper() for c in ba_codes}
        bas = [b for b in bas if b.code in wanted]

    mode = "FULL REFRESH" if full_refresh else "incremental"
    logger.info("Ingesting EIA-930 for %d BA(s) [%s]", len(bas), mode)
    result = asyncio.run(_ingest_async(bas, full_refresh))
    logger.info("EIA ingestion complete: %s rows total", f"{sum(result.values()):,}")
    return result


def probe_eia() -> dict:
    """Single tiny request that validates the API key and response contract.

    Always run this before a full ingestion: it fails in two seconds with a clear
    message instead of thirty minutes into a download.
    """

    async def _run() -> dict:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            return await fetch_json(
                client,
                REGION_DATA_ROUTE,
                _build_params("PJM", "2024-01-01T00", "2024-01-01T05", 0, 10),
                label="EIA:probe",
            )

    payload = _run_sync(_run())
    response = payload.get("response", {})
    sample = response.get("data", [])
    return {
        "api_version": payload.get("apiVersion"),
        "total_rows_matched": response.get("total"),
        "returned": len(sample),
        "columns": sorted(sample[0].keys()) if sample else [],
        "sample_row": sample[0] if sample else None,
    }


def _run_sync(coro):
    """Run a coroutine whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
