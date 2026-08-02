"""DuckDB connection management.

DuckDB is the warehouse engine because it gives columnar OLAP performance over
hundreds of millions of rows inside a single embedded file, with no server to run.
On an 8GB laptop that is the difference between a project that runs and a project
that swaps. The same SQL ports to Snowflake, BigQuery or Azure Synapse unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from gridpulse.config import PATHS

logger = logging.getLogger(__name__)

# Tuned for a constrained laptop: cap DuckDB well below total system RAM so the OS,
# the Python process and any concurrently running training job all still fit.
DEFAULT_MEMORY_LIMIT = "2GB"
DEFAULT_THREADS = 4


@contextmanager
def connect(
    path: Path | str | None = None,
    read_only: bool = False,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a configured DuckDB connection, always closed on exit."""
    target = Path(path) if path else PATHS.duckdb
    target.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(target), read_only=read_only)
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        con.execute("SET preserve_insertion_order=false")  # lower peak memory on big sorts
        yield con
    finally:
        con.close()


def query(sql: str, params: list | None = None, path: Path | str | None = None) -> pd.DataFrame:
    """Run a read-only SELECT and return a DataFrame."""
    with connect(path, read_only=True) as con:
        return con.execute(sql, params or []).df()


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    found = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(found and found[0])


def row_count(con: duckdb.DuckDBPyConnection, name: str) -> int:
    if not table_exists(con, name):
        return 0
    return int(con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])


def summarise(path: Path | str | None = None) -> pd.DataFrame:
    """One row per table with its row count. Used by the CLI and the dashboard."""
    with connect(path, read_only=True) as con:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' ORDER BY table_name"
        ).df()["table_name"].tolist()
        return pd.DataFrame(
            [{"table": t, "rows": row_count(con, t)} for t in tables]
        )
