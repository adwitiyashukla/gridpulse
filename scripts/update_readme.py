"""Regenerate the README results table from the trained model leaderboard.

Keeping the headline number in sync with the artifacts by hand is how READMEs end
up lying. Run this after ``gridpulse train``::

    python scripts/update_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
LEADERBOARD = REPO_ROOT / "artifacts" / "leaderboard.json"
HEADLINE = REPO_ROOT / "artifacts" / "headline.json"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

PRETTY = {
    "gbm": "**LightGBM** (global, quantile)",
    "gbm_hybrid": "**LightGBM hybrid** (+ EIA forecast as input)",
    "lstm": "**LSTM** encoder",
    "transformer": "**Transformer** encoder",
    "ensemble": "**Ensemble** (GBM + LSTM)",
    "eia_official": "_EIA official forecast_ ⭐",
    "seasonal_naive": "Seasonal naive (24h)",
    "weekly_naive": "Weekly naive (168h)",
    "drift_naive": "Drift naive",
}


# P10/P50/P90 are the edges of a prediction interval, not competing point
# forecasts. Ranking them by MAPE compares things that answer different questions.
QUANTILE_MODELS = {"gbm_p10", "gbm_p50", "gbm_p90"}


def render_table(rows: list[dict]) -> str:
    lines = [
        "| Model | MAPE % | MAE (MW) | RMSE (MW) | R² | Peak-hour MAPE % | Skill vs EIA |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r.get("mape_pct", 999)):
        if row["model"] in QUANTILE_MODELS:
            continue
        name = PRETTY.get(row["model"], row["model"])
        skill = row.get("skill_vs_eia_pct")
        if row["model"] == "eia_official":
            skill_cell = "- (benchmark)"
        elif isinstance(skill, (int, float)):
            skill_cell = f"**{skill:+.1f}%**" if skill > 0 else f"{skill:+.1f}%"
        else:
            skill_cell = "-"
        lines.append(
            f"| {name} | {row['mape_pct']:.3f} | {row['mae_mwh']:,.0f} | "
            f"{row['rmse_mwh']:,.0f} | {row['r2']:.4f} | "
            f"{row.get('peak_hour_mape_pct', float('nan')):.3f} | {skill_cell} |"
        )
    return "\n".join(lines)


def main() -> int:
    if not LEADERBOARD.exists():
        print(f"No leaderboard at {LEADERBOARD}. Run `gridpulse train` first.", file=sys.stderr)
        return 1

    rows = json.loads(LEADERBOARD.read_text())
    table = render_table(rows)

    if HEADLINE.exists():
        head = json.loads(HEADLINE.read_text())
        skill = head.get("skill_vs_eia_pct")
        if isinstance(skill, (int, float)) and skill > 0:
            label = PRETTY.get(head["best_model"], head["best_model"]).replace("*", "")
            table = (
                f"> ### {skill:.1f}% more accurate than the EIA's own day-ahead forecast\n"
                f">\n"
                f"> **{label}** reaches **{head['best_mape_pct']:.3f}% MAPE** against the "
                f"EIA's **{head['eia_benchmark_mape_pct']:.3f}%**, measured over "
                f"**{head['test_observations']:,}** out-of-sample hours across 12 balancing "
                f"authorities.\n"
                f">\n"
                f"> Trained without ever seeing the test window. The EIA benchmark is the "
                f"forecast the US government actually published and grid operators actually "
                f"operated against.\n\n" + table
            )

    table += (
        "\n\n<sub>P10/P50/P90 quantile models are omitted above: they define the "
        "prediction interval rather than competing as point forecasts. Interval "
        "calibration is reported separately.</sub>"
    )

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README is missing the RESULTS markers.", file=sys.stderr)
        return 1

    before = text.split(START)[0]
    after = text.split(END)[1]
    README.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")

    print("README results table updated.")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
