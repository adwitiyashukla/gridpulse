"""The warehouse: raw Parquet in bronze, cleaned in silver, star schema in gold."""

from gridpulse.warehouse.build import build_warehouse
from gridpulse.warehouse.duck import connect, query

__all__ = ["build_warehouse", "connect", "query"]
