"""Security tests for the LLM SQL guard.

This is the highest-risk surface in the project: text from a language model is
executed against a database. These tests assert the guard blocks the obvious
attacks and, equally importantly, does not block legitimate analytics.
"""

from __future__ import annotations

import pytest

from gridpulse.agent.text2sql import SQLGuardError, guard_sql


class TestAllowsLegitimateQueries:
    def test_plain_select(self):
        assert "LIMIT" in guard_sql("SELECT ba_code FROM fact_demand_hourly")

    def test_cte(self):
        sql = (
            "WITH averages AS (SELECT ba_code, avg(demand_mwh) d FROM fact_demand_hourly GROUP BY 1) "
            "SELECT * FROM averages ORDER BY d DESC LIMIT 5"
        )
        assert guard_sql(sql).startswith("WITH")

    def test_join_between_allowed_tables(self):
        sql = "SELECT * FROM fact_forecast_accuracy JOIN dim_ba USING (ba_code) LIMIT 10"
        assert guard_sql(sql)

    def test_markdown_fences_are_stripped(self):
        assert not guard_sql("```sql\nSELECT 1 FROM dim_ba LIMIT 1\n```").startswith("`")

    def test_limit_is_injected_when_missing(self):
        assert "LIMIT" in guard_sql("SELECT * FROM dim_ba")

    def test_existing_limit_is_preserved(self):
        assert guard_sql("SELECT * FROM dim_ba LIMIT 7").count("LIMIT") == 1


class TestBlocksAttacks:
    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE dim_ba",
            "DELETE FROM fact_demand_hourly",
            "INSERT INTO dim_ba VALUES (1)",
            "UPDATE dim_ba SET ba_code = 'X'",
            "CREATE TABLE evil AS SELECT 1",
            "ALTER TABLE dim_ba ADD COLUMN x INT",
        ],
    )
    def test_ddl_and_dml_are_refused(self, sql):
        with pytest.raises(SQLGuardError):
            guard_sql(sql)

    def test_stacked_statement_is_refused(self):
        with pytest.raises(SQLGuardError, match="Multiple SQL statements"):
            guard_sql("SELECT * FROM dim_ba; DROP TABLE dim_ba;")

    def test_attach_is_refused(self):
        with pytest.raises(SQLGuardError):
            guard_sql("ATTACH '/etc/passwd' AS leak")

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM duckdb_settings()",
            "SELECT * FROM sqlite_master",
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM secret_table",
        ],
    )
    def test_unknown_tables_are_refused(self, sql):
        with pytest.raises(SQLGuardError, match="unknown table"):
            guard_sql(sql)

    def test_empty_input_is_refused(self):
        with pytest.raises(SQLGuardError):
            guard_sql("   ")

    def test_comment_hidden_payload_is_neutralised(self):
        """A DELETE hidden behind a comment must not survive into the executed SQL."""
        result = guard_sql("SELECT * FROM dim_ba -- ; DELETE FROM dim_ba")
        assert "delete" not in result.lower()
