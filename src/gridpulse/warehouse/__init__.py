"""Lakehouse layer: bronze Parquet -> silver conformed -> gold DuckDB star schema."""

from gridpulse.warehouse.build import build_warehouse
from gridpulse.warehouse.duck import connect, query

__all__ = ["build_warehouse", "connect", "query"]
