from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
SPACE_README = REPO_ROOT / "deploy" / "README_SPACE.md"
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
    "eia_official": "_EIA official forecast_",
    "seasonal_naive": "Seasonal naive (24h)",
    "weekly_naive": "Weekly naive (168h)",
    "drift_naive": "Drift naive",
}


QUANTILE_MODELS = {"gbm_p10", "gbm_p50", "gbm_p90"}


def render_table(rows: list[dict]) -> str:
    lines = [
        "| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r.get("mape_pct", 999)):
        if row["model"] in QUANTILE_MODELS:
            continue
        name = PRETTY.get(row["model"], row["model"])
        skill = row.get("skill_vs_eia_pct")
        if row["model"] == "eia_official":
            skill_cell = "- (benchmark)"
        elif isinstance(skill, int | float):
            skill_cell = f"**{skill:+.1f}%**" if skill > 0 else f"{skill:+.1f}%"
        else:
            skill_cell = "-"
        lines.append(
            f"| {name} | {row['mape_pct']:.3f} | {row['mae_mwh']:,.0f} | "
            f"{row['rmse_mwh']:,.0f} | {row['r2']:.4f} | "
            f"{row.get('peak_hour_mape_pct', float('nan')):.3f} | "
            f"{row.get('n_obs', 0):,} | {skill_cell} |"
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
        if isinstance(skill, int | float) and skill > 0:
            label = PRETTY.get(head["best_model"], head["best_model"]).replace("*", "")
            table = (
                f"{label} gets {head['best_mape_pct']:.3f}% MAPE where the EIA's own "
                f"published forecast gets {head['eia_benchmark_mape_pct']:.3f}%, which is "
                f"{skill:.1f}% better, measured on {head['test_observations']:,} test hours "
                f"from {head['test_window_start']} onwards across 12 balancing authorities.\n\n"
                + table
            )

    table += (
        "\n\nThe P10, P50 and P90 rows are left out of this table. They draw the prediction "
        "interval rather than competing as point forecasts."
    )

    failed = False
    for path in (README, SPACE_README):
        text = path.read_text(encoding="utf-8")
        if START not in text or END not in text:
            print(f"{path.name} is missing the RESULTS markers.", file=sys.stderr)
            failed = True
            continue
        before = text.split(START)[0]
        after = text.split(END)[1]
        path.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")
        print(f"{path.name} results table updated.")

    if failed:
        return 1
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
