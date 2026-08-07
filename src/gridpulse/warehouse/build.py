"""Turning raw downloads into the warehouse: bronze, then silver, then gold.

Each of the three layers has one job:

**Bronze**
    The raw data exactly as it was downloaded, one row per region per hour per
    measurement. I never edit it, only add to it, so if something goes wrong later
    I can always go back and reproduce it.

**Silver**
    Cleaned up and reshaped. The measurements become columns, the weather gets
    joined on, a full list of every hour makes any missing hours obvious, local
    time is worked out for each region, and anything suspicious is *flagged rather
    than deleted*. Deleting a bad reading also deletes the proof that a meter was
    broken, and that proof is the useful part.

**Gold**
    A star schema, which is what the dashboard and the models read from. It has
    lookup tables around `fact_demand_hourly`, plus a separate table that scores
    EIA's own published forecast against what actually happened.

All the heavy work is done in SQL over whole tables at once, rather than looping
through rows in pandas, so memory stays flat no matter how much history there is.
"""

from __future__ import annotations

import logging

import pandas as pd

from gridpulse.config import EIA_MEASURES, PATHS, WEATHER_VARIABLES, active_bas
from gridpulse.warehouse.duck import connect, row_count, table_exists

logger = logging.getLogger(__name__)

# Physically implausible demand. Real BA demand never legitimately hits zero;
# a zero or negative reading is a telemetry failure, not a quiet grid.
MIN_PLAUSIBLE_MWH = 1.0
# Hour-on-hour swings beyond this are almost always bad data, not real load.
MAX_HOURLY_RAMP_PCT = 40.0

# I find spikes by comparing each point against a rolling median centred on it,
# rather than against the point before and after. Comparing to neighbours broke in
# two ways when I tried it. It needs a threshold low enough to catch a spike that
# is only extreme on one side, but high enough not to flag genuine fast changes.
# And it cannot judge the very first or very last row at all, which is exactly
# where the newest and least reliable data sits.
#
# A rolling median has neither problem. Total demand moves smoothly over five
# hours, so a normal day never strays far from its local median, while a single
# bad reading stands out no matter which side it falls on.
SPIKE_WINDOW_HOURS = 2      # rows either side, so a 5 hour window centred on each point
SPIKE_DEVIATION_PCT = 20.0


def _as_utc(value) -> pd.Timestamp:
    """Coerce a timestamp to tz-aware UTC.

    DuckDB returns tz-aware timestamps from ``.df()``, but the exact tzinfo object
    varies by version and platform. Passing such a value to ``pd.date_range``
    alongside ``tz="UTC"`` trips pandas' consistency assertion, so the endpoints
    are normalised here instead.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _dim_ba_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ba_code": ba.code,
                "ba_name": ba.name,
                "timezone": ba.timezone,
                "region": ba.region,
                "load_centre": ba.load_centre,
                "latitude": ba.latitude,
                "longitude": ba.longitude,
            }
            for ba in active_bas()
        ]
    )


def _dim_date_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Calendar dimension including US federal holidays and a holiday-adjacency flag.

    Load on the working day either side of a holiday behaves differently from a
    normal working day, so the adjacency flags earn their place as model features.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar

    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    calendar = USFederalHolidayCalendar()
    holidays = set(calendar.holidays(start=days.min(), end=days.max()))

    frame = pd.DataFrame({"date_day": days})
    frame["year"] = days.year
    frame["quarter"] = days.quarter
    frame["month"] = days.month
    frame["day_of_month"] = days.day
    frame["day_of_week"] = days.dayofweek           # Monday = 0
    frame["day_of_year"] = days.dayofyear
    frame["week_of_year"] = days.isocalendar().week.astype(int)
    frame["is_weekend"] = days.dayofweek >= 5
    frame["is_holiday"] = days.isin(holidays)
    frame["is_business_day"] = ~(frame["is_weekend"] | frame["is_holiday"])
    frame["season"] = days.month.map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )
    holiday_flags = frame["is_holiday"].to_numpy()
    frame["is_day_before_holiday"] = pd.Series(holiday_flags).shift(-1, fill_value=False)
    frame["is_day_after_holiday"] = pd.Series(holiday_flags).shift(1, fill_value=False)
    return frame


def build_warehouse(rebuild: bool = False) -> dict[str, int]:
    """Build the silver and gold layers from the raw Parquet files in bronze.

    Parameters
    ----------
    rebuild
        Drop every managed table before rebuilding.

    Returns
    -------
    dict
        Row counts keyed by table name.
    """
    PATHS.ensure()

    eia_glob = str(PATHS.bronze / "eia_region" / "**" / "*.parquet").replace("\\", "/")
    weather_glob = str(PATHS.bronze / "weather" / "**" / "*.parquet").replace("\\", "/")

    if not list((PATHS.bronze / "eia_region").rglob("*.parquet")):
        raise FileNotFoundError(
            "No bronze EIA data found. Run `gridpulse ingest` before `gridpulse build`."
        )
    has_weather = bool(list((PATHS.bronze / "weather").rglob("*.parquet")))
    if not has_weather:
        logger.warning("No bronze weather data found; weather columns will be NULL.")

    measure_columns = ",\n            ".join(
        f"max(CASE WHEN measure_code = '{code}' THEN value_mwh END) AS {column}"
        for code, column in EIA_MEASURES.items()
    )
    weather_columns = ",\n            ".join(f"w.{v}" for v in WEATHER_VARIABLES)
    weather_nulls = ",\n            ".join(f"CAST(NULL AS DOUBLE) AS {v}" for v in WEATHER_VARIABLES)

    with connect() as con:
        if rebuild:
            # Only the tables this function owns are dropped. Model scores,
            # predictions and anomaly results belong to other commands; wiping
            # them here would silently discard an hour of training because
            # someone rebuilt the data layer.
            logger.info("Rebuild requested: dropping data-layer tables")
            for table in (
                "fact_forecast_accuracy", "fact_demand_hourly",
                "dim_date", "dim_ba", "silver_grid_hourly",
            ):
                con.execute(f"DROP TABLE IF EXISTS {table}")

            stale = [
                t for t in ("model_scores", "model_predictions", "anomaly_scores")
                if table_exists(con, t)
            ]
            if stale:
                logger.warning(
                    "Model outputs %s were computed against the previous warehouse "
                    "and are now stale. Re-run `gridpulse train` and `gridpulse anomalies`.",
                    ", ".join(stale),
                )

        # ------------------------------------------------------------------
        # Dimensions
        # ------------------------------------------------------------------
        logger.info("Building dim_ba")
        con.register("_dim_ba", _dim_ba_frame())
        con.execute("CREATE OR REPLACE TABLE dim_ba AS SELECT * FROM _dim_ba")
        con.unregister("_dim_ba")

        # ------------------------------------------------------------------
        # Silver: pivot measures, attach weather, derive local civil time
        # ------------------------------------------------------------------
        logger.info("Building silver_grid_hourly")
        con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _eia_wide AS
        SELECT
            period_utc,
            ba_code,
            {measure_columns}
        FROM read_parquet('{eia_glob}', hive_partitioning = true, union_by_name = true)
        GROUP BY period_utc, ba_code
        """)

        if has_weather:
            con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _weather AS
            SELECT period_utc, ba_code, {", ".join(WEATHER_VARIABLES)}
            FROM read_parquet('{weather_glob}', hive_partitioning = true, union_by_name = true)
            """)
            weather_join = "LEFT JOIN _weather w ON w.ba_code = s.ba_code AND w.period_utc = s.period_utc"
            weather_select = weather_columns
        else:
            weather_join = ""
            weather_select = weather_nulls

        # A dense hourly spine per BA. Anti-joining against it is the only reliable
        # way to distinguish "reported zero" from "never reported at all".
        # Built in pandas rather than SQL: generate_series over TIMESTAMPTZ depends
        # on the ICU extension and has shifted across DuckDB releases, whereas
        # date_range is deterministic on every platform.
        bounds = con.execute(
            "SELECT ba_code, min(period_utc) AS lo, max(period_utc) AS hi "
            "FROM _eia_wide GROUP BY ba_code"
        ).df()
        spine = pd.concat(
            [
                pd.DataFrame({
                    "ba_code": row.ba_code,
                    "period_utc": pd.date_range(
                        _as_utc(row.lo), _as_utc(row.hi), freq="h"
                    ),
                })
                for row in bounds.itertuples()
            ],
            ignore_index=True,
        )
        logger.info("Hourly spine: %s rows across %d BA(s)", f"{len(spine):,}", len(bounds))
        con.register("_spine_df", spine)
        con.execute("CREATE OR REPLACE TEMP TABLE _spine AS SELECT * FROM _spine_df")
        con.unregister("_spine_df")

        con.execute(f"""
        CREATE OR REPLACE TABLE silver_grid_hourly AS
        SELECT
            s.period_utc,
            s.ba_code,
            d.timezone,
            timezone(d.timezone, s.period_utc)                       AS period_local,
            CAST(timezone(d.timezone, s.period_utc) AS DATE)         AS date_local,
            hour(timezone(d.timezone, s.period_utc))                 AS hour_local,
            isodow(timezone(d.timezone, s.period_utc))               AS iso_dow_local,

            e.demand_mwh,
            e.demand_forecast_mwh,
            e.net_generation_mwh,
            e.total_interchange_mwh,
            {weather_select},

            -- Data quality flags. Diagnosis, not deletion.
            (e.demand_mwh IS NULL)                                   AS flag_missing_demand,
            (e.demand_mwh IS NOT NULL
             AND e.demand_mwh < {MIN_PLAUSIBLE_MWH})                 AS flag_nonpositive_demand,
            (e.period_utc IS NULL)                                   AS flag_absent_interval,
            (e.demand_mwh IS NOT NULL
             AND e.demand_mwh = lag(e.demand_mwh) OVER w
             AND e.demand_mwh = lag(e.demand_mwh, 2) OVER w)         AS flag_frozen_reading
        FROM _spine s
        JOIN dim_ba d              ON d.ba_code = s.ba_code
        LEFT JOIN _eia_wide e      ON e.ba_code = s.ba_code AND e.period_utc = s.period_utc
        {weather_join}
        WINDOW w AS (PARTITION BY s.ba_code ORDER BY s.period_utc)
        ORDER BY s.ba_code, s.period_utc
        """)

        # ------------------------------------------------------------------
        # Gold: calendar dimension sized to the observed data
        # ------------------------------------------------------------------
        span = con.execute(
            "SELECT min(date_local), max(date_local) FROM silver_grid_hourly"
        ).fetchone()
        logger.info("Building dim_date for %s -> %s", span[0], span[1])
        con.register(
            "_dim_date",
            _dim_date_frame(pd.Timestamp(span[0]), pd.Timestamp(span[1]) + pd.Timedelta(days=20)),
        )
        con.execute(
            "CREATE OR REPLACE TABLE dim_date AS "
            "SELECT CAST(date_day AS DATE) AS date_day, * EXCLUDE (date_day) FROM _dim_date"
        )
        con.unregister("_dim_date")

        # ------------------------------------------------------------------
        # Gold: central fact
        # ------------------------------------------------------------------
        logger.info("Building fact_demand_hourly")
        # Robust per-BA bounds. Computed from the median so that the outliers being
        # detected cannot influence the threshold that detects them.
        con.execute("""
        CREATE OR REPLACE TEMP TABLE _ba_bounds AS
        SELECT ba_code,
               median(demand_mwh)       AS median_demand,
               median(demand_mwh) * 0.2 AS lower_bound,
               median(demand_mwh) * 5.0 AS upper_bound
        FROM silver_grid_hourly
        WHERE demand_mwh IS NOT NULL AND demand_mwh > 0
        GROUP BY ba_code
        """)

        # Local level for spike detection, computed once so the window function is
        # not repeated across several expressions.
        con.execute("""
        CREATE OR REPLACE TEMP TABLE _neighbours AS
        SELECT
            period_utc, ba_code,
            quantile_cont(demand_mwh, 0.5) OVER (
                PARTITION BY ba_code ORDER BY period_utc
                ROWS BETWEEN {window} PRECEDING AND {window} FOLLOWING
            ) AS local_median
        FROM silver_grid_hourly
        """.replace("{window}", str(SPIKE_WINDOW_HOURS)))

        con.execute(f"""
        CREATE OR REPLACE TABLE fact_demand_hourly AS
        SELECT
            s.period_utc,
            s.ba_code,
            s.date_local,
            s.hour_local,
            s.demand_mwh,
            s.demand_forecast_mwh,
            s.net_generation_mwh,
            s.total_interchange_mwh,
            {", ".join("s." + v for v in WEATHER_VARIABLES)},

            -- Interpolated demand: a modelling-ready series with short gaps bridged.
            -- The raw column is retained untouched alongside it.
            CASE WHEN s.demand_mwh IS NOT NULL
                      AND NOT s.flag_nonpositive_demand
                      AND s.demand_mwh BETWEEN b.lower_bound AND b.upper_bound
                      AND NOT (
                          n.local_median IS NOT NULL
                          AND abs(s.demand_mwh - n.local_median)
                              / nullif(n.local_median, 0) * 100 > {SPIKE_DEVIATION_PCT}
                      )
                 THEN s.demand_mwh END                                    AS demand_clean_mwh,

            s.flag_missing_demand,
            s.flag_nonpositive_demand,
            s.flag_frozen_reading,
            (abs(s.demand_mwh - lag(s.demand_mwh) OVER w)
                / nullif(lag(s.demand_mwh) OVER w, 0) * 100
                > {MAX_HOURLY_RAMP_PCT})                                  AS flag_extreme_ramp,

            -- Physically impossible magnitude. Flagged, never deleted: the reading
            -- is the evidence that upstream telemetry failed.
            (s.demand_mwh IS NOT NULL
             AND (s.demand_mwh < b.lower_bound
                  OR s.demand_mwh > b.upper_bound))                       AS flag_implausible_magnitude,

            -- Isolated excursion: departs sharply from the local 5-hour median.
            (s.demand_mwh IS NOT NULL
             AND n.local_median IS NOT NULL
             AND abs(s.demand_mwh - n.local_median)
                 / nullif(n.local_median, 0) * 100 > {SPIKE_DEVIATION_PCT}) AS flag_isolated_spike,

            d.day_of_week, d.is_weekend, d.is_holiday, d.is_business_day,
            d.season, d.month, d.year, d.is_day_before_holiday, d.is_day_after_holiday
        FROM silver_grid_hourly s
        LEFT JOIN dim_date d  ON d.date_day = s.date_local
        LEFT JOIN _ba_bounds b  ON b.ba_code = s.ba_code
        LEFT JOIN _neighbours n ON n.ba_code = s.ba_code AND n.period_utc = s.period_utc
        WINDOW w AS (PARTITION BY s.ba_code ORDER BY s.period_utc)
        ORDER BY s.ba_code, s.period_utc
        """)

        # ------------------------------------------------------------------
        # Gold: the benchmark table. This is the one that holds EIA's own forecast
        # next to what actually happened, so anyone can check my headline claim.
        # ------------------------------------------------------------------
        logger.info("Building fact_forecast_accuracy")
        con.execute("""
        CREATE OR REPLACE TABLE fact_forecast_accuracy AS
        SELECT
            period_utc,
            ba_code,
            date_local,
            hour_local,
            demand_clean_mwh                                     AS actual_mwh,
            demand_forecast_mwh                                  AS eia_forecast_mwh,
            demand_forecast_mwh - demand_clean_mwh                AS error_mwh,
            abs(demand_forecast_mwh - demand_clean_mwh)           AS abs_error_mwh,
            abs(demand_forecast_mwh - demand_clean_mwh)
                / nullif(demand_clean_mwh, 0) * 100               AS abs_pct_error,
            CASE WHEN demand_forecast_mwh > demand_clean_mwh
                 THEN 'over' ELSE 'under' END                     AS bias_direction
        FROM fact_demand_hourly
        WHERE demand_clean_mwh IS NOT NULL
          AND demand_forecast_mwh IS NOT NULL
        """)

        con.execute("CREATE INDEX IF NOT EXISTS idx_fact_ba_period ON fact_demand_hourly(ba_code, period_utc)")

        counts = {
            table: row_count(con, table)
            for table in ("dim_ba", "dim_date", "silver_grid_hourly",
                          "fact_demand_hourly", "fact_forecast_accuracy")
        }

    for table, n in counts.items():
        logger.info("  %-24s %10s rows", table, f"{n:,}")
    return counts
