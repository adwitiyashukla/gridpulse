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
LightGBM hybrid (+ EIA forecast as input) gets 2.889% MAPE where the EIA's own published forecast gets 3.631%, which is 20.4% better, measured on 25,914 test hours from 2026-05-19 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.889 | 1,151 | 2,016 | 0.9959 | 3.428 | 25,914 | **+20.4%** |
| _EIA official forecast_ | 3.631 | 1,398 | 2,480 | 0.9939 | 2.825 | 25,300 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.654 | 1,442 | 2,382 | 0.9942 | 4.665 | 25,914 | -0.6% |
| **Ensemble** (GBM + LSTM) | 4.166 | 1,484 | 2,260 | 0.9948 | 3.610 | 25,914 | -14.7% |
| **LSTM** encoder | 5.188 | 1,685 | 2,498 | 0.9937 | 2.923 | 25,908 | -42.9% |
| Seasonal naive (24h) | 5.477 | 1,906 | 3,126 | 0.9901 | 4.962 | 25,914 | -50.8% |
| **Transformer** encoder | 5.766 | 2,182 | 3,323 | 0.9888 | 4.436 | 25,908 | -58.8% |
| Weekly naive (168h) | 9.408 | 3,523 | 5,975 | 0.9638 | 12.630 | 25,914 | -159.1% |

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
