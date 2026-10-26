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
LightGBM hybrid (+ EIA forecast as input) gets 2.912% MAPE where the EIA's own published forecast gets 3.331%, which is 12.6% better, measured on 25,865 test hours from 2026-07-28 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.912 | 1,209 | 2,170 | 0.9948 | 2.434 | 25,865 | **+12.6%** |
| _EIA official forecast_ | 3.331 | 1,221 | 2,177 | 0.9948 | 2.503 | 25,170 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.608 | 1,466 | 2,448 | 0.9934 | 3.792 | 25,865 | -8.3% |
| **Ensemble** (GBM + LSTM) | 4.132 | 1,434 | 2,232 | 0.9945 | 2.687 | 25,865 | -24.1% |
| Seasonal naive (24h) | 4.984 | 1,636 | 2,725 | 0.9918 | 4.310 | 25,865 | -49.6% |
| **LSTM** encoder | 5.174 | 1,568 | 2,499 | 0.9931 | 2.177 | 25,842 | -55.4% |
| **Transformer** encoder | 6.374 | 2,617 | 4,081 | 0.9816 | 6.332 | 25,842 | -91.4% |
| Weekly naive (168h) | 9.484 | 3,137 | 4,938 | 0.9731 | 9.422 | 25,865 | -184.7% |

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
