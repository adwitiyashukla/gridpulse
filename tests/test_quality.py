"""Data quality suite definition and execution."""

from __future__ import annotations

from gridpulse.quality.checks import CHECKS, Dimension, Severity


def test_suite_is_not_empty():
    assert len(CHECKS) >= 10


def test_check_names_are_unique():
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))


def test_every_quality_dimension_is_covered():
    covered = {c.dimension for c in CHECKS}
    assert covered == set(Dimension), f"Uncovered dimensions: {set(Dimension) - covered}"


def test_critical_checks_exist():
    assert any(c.severity is Severity.CRITICAL for c in CHECKS)


def test_thresholds_are_valid_fractions():
    for check in CHECKS:
        assert 0.0 <= check.threshold <= 1.0, f"{check.name} has an invalid threshold"


def test_every_check_declares_a_description():
    for check in CHECKS:
        assert check.description.strip(), f"{check.name} is missing a description"


def test_every_check_returns_failed_and_total():
    """The contract is one row with exactly the columns `failed` and `total`."""
    for check in CHECKS:
        lowered = check.sql.lower()
        assert "failed" in lowered, f"{check.name} does not select `failed`"
        assert "total" in lowered, f"{check.name} does not select `total`"


class TestSuiteExecution:
    def test_suite_runs_and_scores(self, warehouse):
        from gridpulse.quality.checks import run_quality_suite

        report = run_quality_suite(persist=False, database=warehouse)
        assert len(report.results) == len(CHECKS)
        assert 0 <= report.score <= 100

    def test_clean_synthetic_data_passes_critical_checks(self, warehouse):
        from gridpulse.quality.checks import run_quality_suite

        report = run_quality_suite(persist=False, database=warehouse)
        critical_failures = [
            r.check.name
            for r in report.results
            if r.check.severity is Severity.CRITICAL
            and not r.passed
            # Freshness is expected to fail: the synthetic series ends in 2024.
            and r.check.name != "data_freshness"
        ]
        assert not critical_failures, f"Unexpected critical failures: {critical_failures}"

    def test_report_frame_has_one_row_per_check(self, warehouse):
        from gridpulse.quality.checks import run_quality_suite

        report = run_quality_suite(persist=False, database=warehouse)
        frame = report.to_frame()
        assert len(frame) == len(CHECKS)
        assert set(frame.columns) >= {"check_name", "dimension", "severity", "passed"}
