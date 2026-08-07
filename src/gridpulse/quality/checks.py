"""The data quality checks.

Electricity meter data goes wrong in ways that general purpose testing tools do not
look for. A meter reporting the exact same value for six hours is not steady, it is
stuck. An hour that disappears every March is not missing data, it is the clocks
going forward. A negative demand reading does not mean the grid was quiet, it means
someone got a plus and minus the wrong way round further upstream. Every check
below is written around one of those specific problems.

Each check is scored against one of six categories: completeness, validity,
consistency, timeliness, duplicates and accuracy. The results get saved into
``dq_results`` and ``dq_scorecard`` rather than just printed, so I can look back at
how data quality changed over time instead of only seeing today's answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd

from gridpulse.warehouse.duck import connect, table_exists

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "critical"  # fails the pipeline
    WARNING = "warning"    # logged and tracked, does not block
    INFO = "info"          # observability only


class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    ACCURACY = "accuracy"


@dataclass(frozen=True)
class Check:
    """A single quality assertion.

    ``sql`` must return exactly one row with columns ``failed`` and ``total``.
    The check passes when ``failed / total <= threshold``.
    """

    name: str
    dimension: Dimension
    severity: Severity
    description: str
    sql: str
    threshold: float = 0.0  # tolerated failure fraction


@dataclass
class CheckResult:
    check: Check
    failed: int
    total: int
    error: str | None = None

    @property
    def failure_rate(self) -> float:
        return (self.failed / self.total) if self.total else 0.0

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return self.failure_rate <= self.check.threshold


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)
    run_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        """The suite passes when no CRITICAL check has failed."""
        return not any(
            r.check.severity is Severity.CRITICAL and not r.passed for r in self.results
        )

    @property
    def score(self) -> float:
        """Share of checks passing, weighted so critical checks count triple."""
        weights = {Severity.CRITICAL: 3.0, Severity.WARNING: 1.0, Severity.INFO: 0.5}
        total = sum(weights[r.check.severity] for r in self.results)
        earned = sum(weights[r.check.severity] for r in self.results if r.passed)
        return round(100 * earned / total, 1) if total else 100.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "run_at_utc": self.run_at,
                    "check_name": r.check.name,
                    "dimension": r.check.dimension.value,
                    "severity": r.check.severity.value,
                    "description": r.check.description,
                    "failed_rows": r.failed,
                    "total_rows": r.total,
                    "failure_rate_pct": round(100 * r.failure_rate, 4),
                    "threshold_pct": round(100 * r.check.threshold, 4),
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in self.results
            ]
        )


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------
CHECKS: list[Check] = [
    Check(
        name="demand_not_null",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.WARNING,
        description="Actual demand is reported for every hour on the spine.",
        sql="SELECT count(*) FILTER (WHERE demand_mwh IS NULL) AS failed, count(*) AS total FROM fact_demand_hourly",
        threshold=0.02,  # BAs genuinely miss the occasional hour
    ),
    Check(
        name="demand_positive",
        dimension=Dimension.VALIDITY,
        severity=Severity.CRITICAL,
        description="Demand is strictly positive; zero or negative indicates a sign or telemetry fault.",
        sql="SELECT count(*) FILTER (WHERE demand_mwh <= 0) AS failed, count(*) FILTER (WHERE demand_mwh IS NOT NULL) AS total FROM fact_demand_hourly",
        threshold=0.001,
    ),
    Check(
        name="demand_magnitude_plausible",
        dimension=Dimension.VALIDITY,
        severity=Severity.CRITICAL,
        description=(
            "Demand sits within 0.2x-5x the balancing authority's own median. Real system "
            "load never leaves this band; values outside it are telemetry faults that "
            "silently destroy any statistic computed from them."
        ),
        sql="""
            WITH bounds AS (
                SELECT ba_code, median(demand_mwh) AS med
                FROM fact_demand_hourly
                WHERE demand_mwh IS NOT NULL AND demand_mwh > 0
                GROUP BY ba_code
            )
            SELECT
                count(*) FILTER (
                    WHERE f.demand_mwh < b.med * 0.2 OR f.demand_mwh > b.med * 5.0
                ) AS failed,
                count(*) AS total
            FROM fact_demand_hourly f
            JOIN bounds b USING (ba_code)
            WHERE f.demand_mwh IS NOT NULL
        """,
        threshold=0.0005,
    ),
    Check(
        name="demand_dispersion_sane",
        dimension=Dimension.VALIDITY,
        severity=Severity.CRITICAL,
        description=(
            "Every BA's standard deviation stays below its mean on the cleaned series. "
            "A standard deviation exceeding the mean is the signature of extreme outliers, "
            "and any statistic derived from such a series is meaningless. This check runs "
            "against demand_clean_mwh because that is the contract downstream modelling "
            "depends on; the raw column deliberately retains flagged rows as evidence."
        ),
        sql="""
            SELECT
                count(*) FILTER (WHERE sd > mu) AS failed,
                count(*) AS total
            FROM (
                SELECT ba_code,
                       avg(demand_clean_mwh)          AS mu,
                       stddev_samp(demand_clean_mwh)  AS sd
                FROM fact_demand_hourly
                WHERE demand_clean_mwh IS NOT NULL
                GROUP BY ba_code
            )
        """,
    ),
    Check(
        name="cleaned_series_is_usable",
        dimension=Dimension.VALIDITY,
        severity=Severity.CRITICAL,
        description=(
            "demand_clean_mwh retains the overwhelming majority of rows. If cleaning "
            "discarded a large share of the series, the bounds are wrong, not the data."
        ),
        sql="""
            SELECT count(*) FILTER (WHERE demand_clean_mwh IS NULL
                                      AND demand_mwh IS NOT NULL) AS failed,
                   count(*) FILTER (WHERE demand_mwh IS NOT NULL) AS total
            FROM fact_demand_hourly
        """,
        threshold=0.01,
    ),
    Check(
        name="no_duplicate_intervals",
        dimension=Dimension.UNIQUENESS,
        severity=Severity.CRITICAL,
        description="The grain (ba_code, period_utc) is unique. Duplicates usually mean a DST fall-back was stored twice.",
        sql="""
            SELECT coalesce(sum(n - 1), 0) AS failed, count(*) AS total FROM (
                SELECT ba_code, period_utc, count(*) AS n
                FROM fact_demand_hourly GROUP BY 1, 2
            )
        """,
    ),
    Check(
        name="no_missing_hours",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.WARNING,
        description="No hour is absent from the continuous spine between first and last observation.",
        sql="SELECT count(*) FILTER (WHERE flag_missing_demand) AS failed, count(*) AS total FROM fact_demand_hourly",
        threshold=0.02,
    ),
    Check(
        name="no_frozen_readings",
        dimension=Dimension.VALIDITY,
        severity=Severity.WARNING,
        description="Demand does not repeat identically for three or more consecutive hours (stuck telemetry).",
        sql="SELECT count(*) FILTER (WHERE flag_frozen_reading) AS failed, count(*) AS total FROM fact_demand_hourly",
        threshold=0.005,
    ),
    Check(
        name="no_extreme_ramps",
        dimension=Dimension.CONSISTENCY,
        severity=Severity.WARNING,
        description="Hour-on-hour demand change stays within 40 percent; larger swings are usually bad data.",
        sql="SELECT count(*) FILTER (WHERE flag_extreme_ramp) AS failed, count(*) AS total FROM fact_demand_hourly",
        threshold=0.01,
    ),
    Check(
        name="referential_integrity_ba",
        dimension=Dimension.CONSISTENCY,
        severity=Severity.CRITICAL,
        description="Every fact row resolves to a row in dim_ba.",
        sql="""
            SELECT count(*) FILTER (WHERE d.ba_code IS NULL) AS failed, count(*) AS total
            FROM fact_demand_hourly f LEFT JOIN dim_ba d USING (ba_code)
        """,
    ),
    Check(
        name="referential_integrity_date",
        dimension=Dimension.CONSISTENCY,
        severity=Severity.CRITICAL,
        description="Every fact row resolves to a row in dim_date.",
        sql="""
            SELECT count(*) FILTER (WHERE d.date_day IS NULL) AS failed, count(*) AS total
            FROM fact_demand_hourly f LEFT JOIN dim_date d ON d.date_day = f.date_local
        """,
    ),
    Check(
        name="weather_coverage",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.WARNING,
        description="Temperature is joined for the overwhelming majority of hours.",
        sql="SELECT count(*) FILTER (WHERE temperature_2m IS NULL) AS failed, count(*) AS total FROM fact_demand_hourly",
        threshold=0.05,
    ),
    Check(
        name="temperature_physically_plausible",
        dimension=Dimension.VALIDITY,
        severity=Severity.CRITICAL,
        description="Temperature sits inside the physically possible range for the continental US.",
        sql="""
            SELECT count(*) FILTER (WHERE temperature_2m < -60 OR temperature_2m > 60) AS failed,
                   count(*) FILTER (WHERE temperature_2m IS NOT NULL) AS total
            FROM fact_demand_hourly
        """,
    ),
    Check(
        name="data_freshness",
        dimension=Dimension.TIMELINESS,
        severity=Severity.WARNING,
        description="The warehouse holds data within the last 48 hours.",
        sql="""
            SELECT CASE WHEN max(period_utc) < now() - INTERVAL 48 HOUR THEN 1 ELSE 0 END AS failed,
                   1 AS total
            FROM fact_demand_hourly WHERE demand_mwh IS NOT NULL
        """,
    ),
    Check(
        name="eia_benchmark_available",
        dimension=Dimension.ACCURACY,
        severity=Severity.CRITICAL,
        description="EIA's published day-ahead forecast is present, since every model is scored against it.",
        sql="""
            SELECT CASE WHEN count(*) < 1000 THEN 1 ELSE 0 END AS failed, 1 AS total
            FROM fact_forecast_accuracy
        """,
    ),
    Check(
        name="all_bas_present",
        dimension=Dimension.COMPLETENESS,
        severity=Severity.CRITICAL,
        description="Every configured balancing authority contributed rows.",
        sql="""
            SELECT count(*) FILTER (WHERE f.ba_code IS NULL) AS failed, count(*) AS total
            FROM dim_ba d
            LEFT JOIN (SELECT DISTINCT ba_code FROM fact_demand_hourly) f USING (ba_code)
        """,
    ),
]


def run_quality_suite(persist: bool = True, database=None) -> QualityReport:
    """Execute every check and optionally persist the results.

    Parameters
    ----------
    persist
        Write results to ``dq_results`` and rebuild ``dq_scorecard``.
    database
        Override the warehouse path. Used by the test suite.

    Returns
    -------
    QualityReport
        ``report.passed`` is False when any CRITICAL check failed.
    """
    report = QualityReport()

    with connect(database, read_only=not persist) as con:
        if not table_exists(con, "fact_demand_hourly"):
            raise FileNotFoundError("Warehouse not built. Run `gridpulse build` first.")

        for check in CHECKS:
            try:
                row = con.execute(check.sql).fetchone()
                result = CheckResult(check, int(row[0] or 0), int(row[1] or 0))
            except Exception as exc:  # noqa: BLE001
                result = CheckResult(check, 0, 0, error=str(exc)[:300])
            report.results.append(result)

            marker = "PASS" if result.passed else ("FAIL" if check.severity is Severity.CRITICAL else "WARN")
            logger.info(
                "  [%s] %-32s %-13s %8s/%-10s (%.3f%%)",
                marker, check.name, check.dimension.value,
                f"{result.failed:,}", f"{result.total:,}", 100 * result.failure_rate,
            )

        if persist:
            frame = report.to_frame()
            con.register("_dq", frame)
            con.execute("CREATE TABLE IF NOT EXISTS dq_results AS SELECT * FROM _dq LIMIT 0")
            con.execute("INSERT INTO dq_results SELECT * FROM _dq")
            con.unregister("_dq")

            con.execute("""
            CREATE OR REPLACE TABLE dq_scorecard AS
            WITH latest AS (SELECT max(run_at_utc) AS r FROM dq_results)
            SELECT dimension,
                   count(*)                                        AS checks_run,
                   count(*) FILTER (WHERE passed)                  AS checks_passed,
                   round(100.0 * count(*) FILTER (WHERE passed) / count(*), 1) AS pass_pct,
                   max(run_at_utc)                                 AS run_at_utc
            FROM dq_results, latest
            WHERE run_at_utc = latest.r
            GROUP BY dimension ORDER BY dimension
            """)

    passed_n = sum(r.passed for r in report.results)
    logger.info(
        "Quality score %.1f%%  (%d/%d checks passed, suite %s)",
        report.score, passed_n, len(report.results), "PASSED" if report.passed else "FAILED",
    )
    return report
