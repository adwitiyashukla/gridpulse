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
LightGBM hybrid (+ EIA forecast as input) gets 2.937% MAPE where the EIA's own published forecast gets 3.355%, which is 12.5% better, measured on 25,893 test hours from 2026-07-14 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.937 | 1,248 | 2,211 | 0.9950 | 2.736 | 25,893 | **+12.5%** |
| _EIA official forecast_ | 3.355 | 1,305 | 2,320 | 0.9946 | 2.453 | 25,198 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.707 | 1,545 | 2,566 | 0.9932 | 4.235 | 25,893 | -10.5% |
| **Ensemble** (GBM + LSTM) | 4.182 | 1,592 | 2,610 | 0.9930 | 3.117 | 25,893 | -24.6% |
| Seasonal naive (24h) | 5.055 | 1,730 | 2,851 | 0.9916 | 4.512 | 25,893 | -50.6% |
| **LSTM** encoder | 5.119 | 1,809 | 3,326 | 0.9886 | 2.352 | 25,872 | -52.6% |
| **Transformer** encoder | 5.899 | 2,115 | 3,182 | 0.9896 | 4.090 | 25,872 | -75.8% |
| Weekly naive (168h) | 9.119 | 3,178 | 5,106 | 0.9732 | 9.976 | 25,893 | -171.8% |

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
