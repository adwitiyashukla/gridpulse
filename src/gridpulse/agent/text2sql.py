"""Letting people ask questions in plain English instead of writing SQL.

You can ask "which region had the worst forecast error last month?" and get a
chart back rather than a SQL editor. The part I actually had to think about is not
calling the LLM, which is about four lines of code. It is not trusting what comes
back from it.

There are six checks, in this order:

1. **The connection is read-only.** I open the database with ``read_only=True``,
   so even if someone completely tricked the model, it still could not change
   anything.
2. **Only one statement, and it has to be a SELECT.** It must start with
   ``SELECT`` or ``WITH``. Anything else gets rejected before it runs.
3. **Banned keywords.** Anything that creates, changes or deletes data is refused,
   including when it is hidden inside a comment or tacked on after a semicolon.
4. **A list of allowed tables.** Only my actual warehouse tables can be queried,
   which stops anyone reading DuckDB's internal system tables.
5. **A forced LIMIT.** If the model forgets one, I add it, so no single query can
   return a huge amount of data.
6. **The real schema goes into the prompt.** I read the actual table and column
   names out of the database and put them in the prompt, so the model describes
   real columns instead of inventing ones that sound plausible.

The SQL it generated always comes back along with the answer. If you cannot see
the query, you have no way to judge whether the answer is right.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from gridpulse.config import PATHS, SETTINGS
from gridpulse.warehouse.duck import connect

logger = logging.getLogger(__name__)

MAX_ROWS = 5000

ALLOWED_TABLES = {
    "fact_demand_hourly", "fact_forecast_accuracy", "dim_ba", "dim_date",
    "model_predictions", "model_scores", "anomaly_scores",
    "dq_results", "dq_scorecard", "silver_grid_hourly",
}

FORBIDDEN = {
    "insert", "update", "delete", "drop", "create", "alter", "truncate", "replace",
    "attach", "detach", "copy", "export", "import", "install", "load", "pragma",
    "set", "call", "grant", "revoke", "vacuum", "checkpoint",
}

SYSTEM_PROMPT = """You are a senior analytics engineer for GridPulse, a US electricity grid data platform.
You translate questions into a single DuckDB SQL query.

RULES
- Return ONLY SQL. No prose, no markdown fences, no explanation.
- Exactly one statement. It must start with SELECT or WITH.
- Never write, modify or drop anything.
- Only reference the tables and columns given in the schema below.
- Always include a LIMIT clause (at most {max_rows}).
- Timestamps are TIMESTAMP WITH TIME ZONE in UTC. For "local" questions use date_local/hour_local.
- Demand and generation are in megawatthours (MWh).
- Prefer readable aliases and round percentages to 2 decimals.

DOMAIN NOTES
- fact_demand_hourly is the central fact, one row per (ba_code, period_utc).
- demand_mwh is the actual observed load; demand_forecast_mwh is EIA's own published day-ahead forecast.
- fact_forecast_accuracy pre-computes EIA's error: abs_pct_error is their MAPE contribution per hour.
- model_scores holds the leaderboard; skill_vs_eia_pct is percentage improvement over EIA (higher is better).
- anomaly_scores flags suspect hours; severity is one of none/low/medium/high.

SCHEMA
{schema}

EXAMPLES
Q: Which BA had the highest peak demand last summer?
A: SELECT ba_code, max(demand_mwh) AS peak_mwh FROM fact_demand_hourly WHERE season = 'Summer' GROUP BY ba_code ORDER BY peak_mwh DESC LIMIT 10;

Q: How accurate is EIA's forecast by balancing authority?
A: SELECT ba_code, round(avg(abs_pct_error), 3) AS eia_mape_pct, count(*) AS hours FROM fact_forecast_accuracy GROUP BY ba_code ORDER BY eia_mape_pct LIMIT 20;

Q: Show me demand versus temperature for ERCOT.
A: SELECT round(temperature_2m) AS temp_c, round(avg(demand_mwh)) AS avg_demand_mwh, count(*) AS hours FROM fact_demand_hourly WHERE ba_code = 'ERCO' AND temperature_2m IS NOT NULL GROUP BY temp_c ORDER BY temp_c LIMIT 100;
"""


class SQLGuardError(RuntimeError):
    """Raised when generated SQL violates a safety rule."""


@dataclass
class AgentAnswer:
    question: str
    sql: str
    data: pd.DataFrame
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    text = re.sub(r"^\s*```(?:sql)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", text).strip()


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)


def guard_sql(raw: str, allowed_tables: set[str] | None = None) -> str:
    """Validate and normalise model-generated SQL, or raise :class:`SQLGuardError`."""
    allowed = allowed_tables if allowed_tables is not None else ALLOWED_TABLES
    sql = _strip_fences(raw)
    if not sql:
        raise SQLGuardError("The model returned an empty query.")

    bare = _strip_comments(sql)

    statements = [s for s in bare.split(";") if s.strip()]
    if len(statements) > 1:
        raise SQLGuardError("Multiple SQL statements are not allowed.")

    body = statements[0].strip()
    if not re.match(r"^\s*(select|with)\b", body, flags=re.IGNORECASE):
        raise SQLGuardError("Only SELECT and WITH queries are permitted.")

    tokens = set(re.findall(r"\b[a-z_]+\b", body.lower()))
    banned = tokens & FORBIDDEN
    if banned:
        raise SQLGuardError(f"Query contains forbidden keyword(s): {', '.join(sorted(banned))}")

    referenced = {
        match.lower()
        for match in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", body, flags=re.IGNORECASE)
    }
    # CTE names are defined inline and are legitimate targets.
    cte_names = {m.lower() for m in re.findall(r"(?:with|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", body, flags=re.IGNORECASE)}
    unknown = referenced - allowed - cte_names
    if unknown:
        raise SQLGuardError(
            f"Query references unknown table(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(allowed))}"
        )

    if not re.search(r"\blimit\s+\d+", body, flags=re.IGNORECASE):
        body = f"{body.rstrip()}\nLIMIT {MAX_ROWS}"

    return body


# ---------------------------------------------------------------------------
# Schema grounding
# ---------------------------------------------------------------------------
def introspect_schema(database=None, tables: set[str] | None = None) -> str:
    """Render a compact schema description for the prompt."""
    wanted = tables or ALLOWED_TABLES
    lines: list[str] = []
    with connect(database, read_only=True) as con:
        available = set(
            con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).df()["table_name"]
        )
        for table in sorted(wanted & available):
            columns = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).df()
            rendered = ", ".join(f"{r.column_name} {r.data_type}" for r in columns.itertuples())
            lines.append(f"{table}({rendered})")
    return "\n".join(lines) if lines else "(warehouse is empty)"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class GridAgent:
    """Question in, validated SQL and a DataFrame out."""

    def __init__(self, database=None, model: str | None = None):
        self.database = database or (PATHS.gold / "gridpulse_app.duckdb"
                                     if (PATHS.gold / "gridpulse_app.duckdb").exists()
                                     else PATHS.duckdb)
        self.model = model or SETTINGS.groq_model
        self._schema: str | None = None
        self._client = None

    @property
    def available(self) -> bool:
        return SETTINGS.has_llm

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = introspect_schema(self.database)
        return self._schema

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=SETTINGS.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        return self._client

    def generate_sql(self, question: str) -> str:
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(schema=self.schema, max_rows=MAX_ROWS)},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        return response.choices[0].message.content or ""

    def summarise(self, question: str, sql: str, data: pd.DataFrame) -> str:
        """Two-sentence plain-English reading of the result set."""
        try:
            preview = data.head(20).to_markdown(index=False)
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a grid analyst. In at most two sentences, state what the "
                            "result shows. Quote concrete numbers with units (MWh, percent). "
                            "Do not describe the SQL. Do not speculate beyond the data."
                        ),
                    },
                    {"role": "user", "content": f"Question: {question}\n\nResult ({len(data)} rows):\n{preview}"},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summary generation failed: %s", exc)
            return ""

    def ask(self, question: str, summarise: bool = True) -> AgentAnswer:
        """Full round trip with guardrails and a single self-repair retry."""
        if not self.available:
            return AgentAnswer(
                question, "", pd.DataFrame(),
                error="No GROQ_API_KEY configured. Add one to .env to enable the agent.",
            )

        warnings: list[str] = []
        sql = ""
        try:
            raw = self.generate_sql(question)
            try:
                sql = guard_sql(raw)
            except SQLGuardError as first_failure:
                # One repair attempt: hand the model its own error and ask again.
                warnings.append(f"First attempt rejected: {first_failure}")
                logger.info("Guard rejected SQL, retrying: %s", first_failure)
                repair = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.format(schema=self.schema, max_rows=MAX_ROWS)},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"That was rejected: {first_failure}. Return corrected SQL only."},
                    ],
                    temperature=0.0,
                    max_tokens=700,
                )
                sql = guard_sql(repair.choices[0].message.content or "")

            with connect(self.database, read_only=True) as con:
                data = con.execute(sql).df()

            if len(data) >= MAX_ROWS:
                warnings.append(f"Result truncated to {MAX_ROWS} rows.")

            summary = self.summarise(question, sql, data) if (summarise and not data.empty) else ""
            return AgentAnswer(question, sql, data, summary, warnings)

        except SQLGuardError as exc:
            return AgentAnswer(question, sql, pd.DataFrame(), warnings=warnings, error=f"Blocked by SQL guard: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent failed: %s", exc)
            return AgentAnswer(question, sql, pd.DataFrame(), warnings=warnings, error=str(exc)[:400])


SAMPLE_QUESTIONS = [
    "Which balancing authority has the highest average demand?",
    "How accurate is EIA's own day-ahead forecast for each balancing authority?",
    "Show average demand by hour of day for ERCOT in summer.",
    "Which model performs best on the leaderboard and by how much does it beat EIA?",
    "What were the ten highest demand hours ever recorded and where?",
    "How does average demand vary with temperature in California?",
    "How many high severity anomalies were detected per balancing authority?",
    "Compare weekday versus weekend demand for PJM.",
]
