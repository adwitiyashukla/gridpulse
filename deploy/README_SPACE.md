---
title: GridPulse
emoji: ⚡
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Day-ahead grid demand forecasting benchmarked vs the EIA
---

# GridPulse

Day-ahead electricity demand forecasting for US balancing authorities,
benchmarked against the EIA's own published day-ahead forecast.

<!-- RESULTS:START -->
LightGBM hybrid (+ EIA forecast as input) gets 3.028% MAPE where the EIA's own published forecast gets 3.455%, which is 12.3% better, measured on 25,911 test hours from 2026-06-23 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 3.028 | 1,328 | 2,350 | 0.9948 | 3.706 | 25,911 | **+12.3%** |
| _EIA official forecast_ | 3.455 | 1,400 | 2,486 | 0.9943 | 2.712 | 25,240 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.805 | 1,661 | 2,781 | 0.9927 | 4.913 | 25,911 | -10.1% |
| **Ensemble** (GBM + LSTM) | 4.884 | 2,102 | 3,363 | 0.9893 | 5.140 | 25,911 | -41.4% |
| Seasonal naive (24h) | 5.158 | 1,844 | 3,052 | 0.9912 | 4.687 | 25,911 | -49.3% |
| **LSTM** encoder | 6.441 | 2,686 | 4,389 | 0.9818 | 5.696 | 25,902 | -86.4% |
| **Transformer** encoder | 7.022 | 3,205 | 4,945 | 0.9769 | 6.264 | 25,902 | -103.3% |
| Weekly naive (168h) | 8.597 | 3,273 | 5,545 | 0.9710 | 10.029 | 25,911 | -148.8% |

The P10, P50 and P90 rows are left out of this table. They draw the prediction interval rather than competing as point forecasts.
<!-- RESULTS:END -->

## What you can do here

| Tab | What it does |
|---|---|
| **Forecast** | Make a live 24-hour forecast with a P10-P90 range for any of the 12 regions, using the current weather forecast |
| **Explorer** | Past demand, the V-shaped link between temperature and demand, and how weekdays differ from weekends |
| **Model Leaderboard** | Every model scored against EIA's own forecast on exactly the same test hours |
| **Anomalies** | Hours that look wrong, found by three detectors that have to agree |
| **Data Quality** | A scorecard covering six categories of data quality |
| **Ask the Grid** | Ask a question in normal English and get SQL and a chart back |

## What is behind it

This Space is just the website. Behind it there is a full data pipeline:

- Downloads data from the EIA and Open-Meteo APIs a bit at a time, remembering
  where it got to so it never re-downloads the same thing
- Stores it in bronze, silver and gold layers, ending in a star schema in DuckDB
- Runs 16 data quality checks covering completeness, validity, consistency,
  timeliness, duplicates and accuracy
- Trains LightGBM with P10/P50/P90 bands, plus an LSTM and a Transformer in
  PyTorch so I could compare them properly
- Finds unusual hours using three different detectors that have to agree: a
  seasonal z-score, an Isolation Forest, and an autoencoder trained on daily
  demand shapes
- Has an LLM that writes SQL, wrapped in guardrails (read-only connection, SELECT
  only, a list of allowed tables, and a row limit)

## Setup

The Ask the Grid tab needs a `GROQ_API_KEY` under
Settings, Variables and secrets. Keys are free at
[console.groq.com/keys](https://console.groq.com/keys). Everything else works with
no setup at all.

## Source

Full pipeline, tests, orchestration and documentation:
[github.com/adwitiyashukla/gridpulse](https://github.com/adwitiyashukla/gridpulse)

Data: [US EIA Form 930](https://www.eia.gov/opendata/) and
[Open-Meteo](https://open-meteo.com/).
