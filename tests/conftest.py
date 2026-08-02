"""Shared fixtures. Every test runs without network access or API keys."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("EIA_API_KEY", "test-key-not-used")
os.environ.setdefault("GRIDPULSE_BAS", "PJM,ERCO,CISO")


@pytest.fixture(scope="session")
def synthetic_grid() -> pd.DataFrame:
    """A realistic synthetic hourly demand series for three balancing authorities.

    Construction mirrors the real physics so downstream tests are meaningful:
    a daily cycle, a weekly cycle, an annual temperature cycle, a V-shaped
    demand response around the comfort balance point, and gaussian noise.
    """
    rng = np.random.default_rng(42)
    periods = pd.date_range("2022-01-01", "2024-06-30 23:00", freq="h", tz="UTC")
    frames = []

    for ba, scale, offset in [("PJM", 90_000, 0.0), ("ERCO", 55_000, 0.35), ("CISO", 30_000, -0.2)]:
        hour = periods.hour.to_numpy()
        dow = periods.dayofweek.to_numpy()
        doy = periods.dayofyear.to_numpy()

        temperature = 15 + 12 * np.sin(2 * np.pi * (doy - 100) / 365.25 + offset) \
            + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 2.0, len(periods))

        daily = 1 + 0.18 * np.sin(2 * np.pi * (hour - 8) / 24)
        weekly = np.where(dow >= 5, 0.88, 1.0)
        heating = np.clip(18 - temperature, 0, None)
        cooling = np.clip(temperature - 18, 0, None)
        weather_effect = 1 + 0.012 * cooling + 0.008 * heating

        demand = scale * daily * weekly * weather_effect * (1 + rng.normal(0, 0.02, len(periods)))

        frames.append(pd.DataFrame({
            "period_utc": periods,
            "ba_code": ba,
            "demand_mwh": demand.round(1),
            # EIA's forecast: the truth plus a realistic ~2 percent error.
            "demand_forecast_mwh": (demand * (1 + rng.normal(0, 0.021, len(periods)))).round(1),
            "net_generation_mwh": (demand * 1.03).round(1),
            "total_interchange_mwh": rng.normal(0, scale * 0.02, len(periods)).round(1),
            "temperature_2m": temperature.round(2),
        }))

    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def bronze_dir(tmp_path_factory, synthetic_grid: pd.DataFrame) -> Path:
    """Write the synthetic series into a bronze layout the warehouse can read."""
    root = tmp_path_factory.mktemp("bronze")
    measures = {
        "D": "demand_mwh", "DF": "demand_forecast_mwh",
        "NG": "net_generation_mwh", "TI": "total_interchange_mwh",
    }

    for ba, group in synthetic_grid.groupby("ba_code"):
        long = pd.concat([
            pd.DataFrame({
                "period_utc": group["period_utc"],
                "ba_code": ba,
                "measure_code": code,
                "value_mwh": group[column],
                "ingested_at_utc": pd.Timestamp.now(tz="UTC"),
            })
            for code, column in measures.items()
        ], ignore_index=True)

        target = root / "eia_region" / f"ba={ba}"
        target.mkdir(parents=True, exist_ok=True)
        long.to_parquet(target / "data.parquet", index=False)

        weather = root / "weather" / f"ba={ba}"
        weather.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "period_utc": group["period_utc"],
            "ba_code": ba,
            "temperature_2m": group["temperature_2m"],
            "relative_humidity_2m": 60.0,
            "dew_point_2m": group["temperature_2m"] - 5,
            "apparent_temperature": group["temperature_2m"] - 1,
            "cloud_cover": 40.0,
            "wind_speed_10m": 12.0,
            "shortwave_radiation": 200.0,
            "source": "era5_archive",
            "ingested_at_utc": pd.Timestamp.now(tz="UTC"),
        }).to_parquet(weather / "data.parquet", index=False)

    return root


@pytest.fixture(scope="session")
def warehouse(bronze_dir: Path, tmp_path_factory):
    """A fully built DuckDB warehouse over the synthetic bronze data.

    Paths are patched at module level rather than via environment variables,
    because the config dataclasses resolve their paths at import time.
    """
    from gridpulse.config import Paths
    from gridpulse.warehouse import build as build_module
    from gridpulse.warehouse import duck as duck_module

    database = tmp_path_factory.mktemp("gold") / "test.duckdb"

    class TestPaths(Paths):
        @property
        def bronze(self):
            return bronze_dir

        @property
        def duckdb(self):
            return database

        @property
        def gold(self):
            return database.parent

    test_paths = TestPaths(data=bronze_dir.parent)

    original_build, original_duck = build_module.PATHS, duck_module.PATHS
    build_module.PATHS = test_paths
    duck_module.PATHS = test_paths
    try:
        build_module.build_warehouse(rebuild=True)
        yield database
    finally:
        build_module.PATHS = original_build
        duck_module.PATHS = original_duck
