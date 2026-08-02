"""Diagnostic: inspect the tail of a BA's series to identify the chart spikes."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BA = sys.argv[1] if len(sys.argv) > 1 else "BPAT"

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

for label, path in [
    ("APP EXPORT", ROOT / "data" / "gold" / "gridpulse_app.duckdb"),
    ("FULL WAREHOUSE", ROOT / "data" / "gold" / "gridpulse.duckdb"),
]:
    print(f"\n{'=' * 100}\n  {label}: {path.name}\n{'=' * 100}")
    if not path.exists():
        print("  not found")
        continue

    con = duckdb.connect(str(path), read_only=True)

    cols = set(con.execute("DESCRIBE fact_demand_hourly").df()["column_name"])
    print(f"  spike flag present: {'flag_isolated_spike' in cols}")

    select = [
        "period_utc", "demand_mwh",
        "demand_clean_mwh" if "demand_clean_mwh" in cols else "NULL AS demand_clean_mwh",
    ]
    for flag in ("flag_isolated_spike", "flag_extreme_ramp",
                 "flag_frozen_reading", "flag_implausible_magnitude"):
        select.append(flag if flag in cols else f"NULL AS {flag}")

    tail = con.execute(f"""
        SELECT {', '.join(select)}
        FROM fact_demand_hourly
        WHERE ba_code = '{BA}'
        ORDER BY period_utc DESC
        LIMIT 60
    """).df().sort_values("period_utc")

    tail["step_pct"] = tail["demand_mwh"].pct_change() * 100
    print(f"\n  Last 60 hours for {BA}:\n")
    print(tail.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print(f"\n  demand_mwh       nulls: {int(tail['demand_mwh'].isna().sum())} of 60")
    print(f"  demand_clean_mwh nulls: {int(tail['demand_clean_mwh'].isna().sum())} of 60")
    big = tail["step_pct"].abs() > 20
    print(f"  hour-on-hour steps over 20%: {int(big.sum())}")

    con.close()
