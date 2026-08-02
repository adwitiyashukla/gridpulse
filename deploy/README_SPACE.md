---
title: GridPulse - US Electricity Demand Intelligence
emoji: ⚡
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: true
license: mit
short_description: Day-ahead grid demand forecasting benchmarked vs the EIA
---

# ⚡ GridPulse

Day-ahead electricity demand forecasting for US balancing authorities, benchmarked
against the **EIA's own published day-ahead forecast**.

This Space is the public front end of a full data platform: incremental extraction
from the EIA Open Data API and Open-Meteo, a medallion lakehouse on DuckDB, a
Kimball star schema, a 13-check data quality suite, LightGBM and PyTorch
forecasting models, three-detector anomaly consensus, and a guarded LLM text-to-SQL
agent.

## What you can do here

| Tab | What it does |
|---|---|
| **Forecast** | Generate a live 24-hour demand forecast with P10-P90 intervals for any balancing authority |
| **Explorer** | Historical demand, the V-shaped temperature response, and weekday/weekend load shapes |
| **Model Leaderboard** | Every model scored against the EIA's official forecast on identical out-of-sample hours |
| **Anomalies** | Suspect hours flagged by three independent detectors |
| **Data Quality** | The quality scorecard across six classical quality dimensions |
| **Ask the Grid** | Ask a question in plain English; get guarded SQL and a chart back |

## Configuration

To enable the **Ask the Grid** agent, add a `GROQ_API_KEY` under
**Settings → Variables and secrets**. Free keys are available at
[console.groq.com/keys](https://console.groq.com/keys). Everything else works
without any configuration.

## Source

Full source, pipeline and documentation: **[GitHub repository](https://github.com/adwitiyashukla/gridpulse)**

Data: [US Energy Information Administration, Form EIA-930](https://www.eia.gov/opendata/)
and [Open-Meteo](https://open-meteo.com/).
