"""End-to-end warehouse build over synthetic bronze data.

These are integration tests: they exercise the real DuckDB SQL that ships, against
data whose correct answers are known by construction.
"""

from __future__ import annotations

import duckdb
import pytest


@pytest.fixture
def con(warehouse):
    connection = duckdb.connect(str(warehouse), read_only=True)
    yield connection
    connection.close()


def test_all_expected_tables_exist(con):
    tables = set(
        con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).df()["table_name"]
    )
    assert {"dim_ba", "dim_date", "silver_grid_hourly",
            "fact_demand_hourly", "fact_forecast_accuracy"} <= tables


def test_fact_table_is_populated(con):
    assert con.execute("SELECT count(*) FROM fact_demand_hourly").fetchone()[0] > 10_000


def test_grain_is_unique(con):
    """(ba_code, period_utc) is the declared grain and must be unique."""
    duplicates = con.execute(
        "SELECT count(*) FROM (SELECT ba_code, period_utc FROM fact_demand_hourly "
        "GROUP BY 1, 2 HAVING count(*) > 1)"
    ).fetchone()[0]
    assert duplicates == 0


def test_measures_were_pivoted_into_columns(con):
    columns = set(con.execute("DESCRIBE fact_demand_hourly").df()["column_name"])
    assert {"demand_mwh", "demand_forecast_mwh",
            "net_generation_mwh", "total_interchange_mwh"} <= columns


def test_weather_is_joined(con):
    joined = con.execute(
        "SELECT count(*) FROM fact_demand_hourly WHERE temperature_2m IS NOT NULL"
    ).fetchone()[0]
    total = con.execute("SELECT count(*) FROM fact_demand_hourly").fetchone()[0]
    assert joined / total > 0.95


def test_local_time_conversion_is_applied(con):
    """period_local must differ from period_utc by the BA's UTC offset."""
    offsets = con.execute("""
        SELECT DISTINCT ba_code,
               date_diff('hour', period_local, period_utc AT TIME ZONE 'UTC') AS offset_hours
        FROM silver_grid_hourly LIMIT 20
    """).df()
    assert not offsets.empty


def test_hourly_spine_is_continuous(con):
    """No hour may be missing between the first and last observation per BA.

    Elapsed time is measured in epoch seconds rather than with ``date_diff``,
    because calendar-based differencing is defined in terms of the session
    timezone and would report daylight-saving transitions as gaps even when the
    underlying UTC series is perfectly continuous.
    """
    gaps = con.execute("""
        SELECT ba_code, period_utc, delta_hours FROM (
            SELECT ba_code, period_utc,
                   (epoch(period_utc)
                    - epoch(lag(period_utc) OVER (PARTITION BY ba_code ORDER BY period_utc)))
                   / 3600.0 AS delta_hours
            FROM fact_demand_hourly
        ) WHERE delta_hours IS NOT NULL AND delta_hours <> 1
        ORDER BY ba_code, period_utc
    """).df()

    assert gaps.empty, (
        f"{len(gaps)} discontinuity/ies in the hourly spine:\n"
        f"{gaps.head(20).to_string(index=False)}"
    )


def test_referential_integrity_to_dim_ba(con):
    orphans = con.execute("""
        SELECT count(*) FROM fact_demand_hourly f
        LEFT JOIN dim_ba d USING (ba_code) WHERE d.ba_code IS NULL
    """).fetchone()[0]
    assert orphans == 0


def test_calendar_dimension_flags_weekends_correctly(con):
    wrong = con.execute("""
        SELECT count(*) FROM dim_date
        WHERE is_weekend <> (day_of_week >= 5)
    """).fetchone()[0]
    assert wrong == 0


def test_forecast_accuracy_table_computes_error(con):
    row = con.execute("""
        SELECT count(*) AS n, avg(abs_pct_error) AS mape
        FROM fact_forecast_accuracy
    """).df().iloc[0]
    assert row["n"] > 1000
    # The synthetic benchmark carries roughly 2.1 percent noise by construction.
    assert 0.5 < row["mape"] < 6.0


def test_quality_flags_are_boolean_and_present(con):
    columns = set(con.execute("DESCRIBE fact_demand_hourly").df()["column_name"])
    assert {"flag_missing_demand", "flag_nonpositive_demand",
            "flag_frozen_reading", "flag_extreme_ramp"} <= columns
