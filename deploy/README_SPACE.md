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
LightGBM hybrid (+ EIA forecast as input) gets 2.991% MAPE where the EIA's own published forecast gets 3.374%, which is 11.3% better, measured on 25,894 test hours from 2026-07-07 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.991 | 1,284 | 2,248 | 0.9950 | 3.084 | 25,894 | **+11.3%** |
| _EIA official forecast_ | 3.374 | 1,332 | 2,353 | 0.9946 | 2.531 | 25,198 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.762 | 1,599 | 2,644 | 0.9930 | 4.461 | 25,894 | -11.5% |
| **Ensemble** (GBM + LSTM) | 4.453 | 1,625 | 2,407 | 0.9942 | 3.573 | 25,894 | -32.0% |
| Seasonal naive (24h) | 5.111 | 1,766 | 2,891 | 0.9917 | 4.516 | 25,894 | -51.5% |
| **LSTM** encoder | 5.699 | 1,811 | 2,520 | 0.9937 | 2.858 | 25,872 | -68.9% |
| **Transformer** encoder | 6.551 | 2,437 | 3,678 | 0.9865 | 4.306 | 25,872 | -94.2% |
| Weekly naive (168h) | 8.977 | 3,180 | 5,151 | 0.9736 | 10.258 | 25,894 | -166.1% |

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
