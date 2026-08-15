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
LightGBM hybrid (+ EIA forecast as input) gets 2.877% MAPE where the EIA's own published forecast gets 3.655%, which is 21.3% better, measured on 25,918 test hours from 2026-05-17 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.877 | 1,136 | 2,002 | 0.9959 | 3.435 | 25,918 | **+21.3%** |
| **LightGBM** (global, quantile) | 3.644 | 1,430 | 2,372 | 0.9943 | 4.657 | 25,918 | **+0.3%** |
| _EIA official forecast_ | 3.655 | 1,405 | 2,487 | 0.9938 | 2.848 | 25,342 | - (benchmark) |
| **Ensemble** (GBM + LSTM) | 4.490 | 1,694 | 2,688 | 0.9927 | 3.622 | 25,918 | -22.9% |
| Seasonal naive (24h) | 5.572 | 1,942 | 3,206 | 0.9896 | 5.271 | 25,918 | -52.5% |
| **LSTM** encoder | 5.874 | 2,136 | 3,481 | 0.9877 | 2.997 | 25,914 | -60.7% |
| **Transformer** encoder | 8.328 | 3,301 | 4,913 | 0.9755 | 6.073 | 25,914 | -127.8% |
| Weekly naive (168h) | 9.460 | 3,566 | 6,072 | 0.9625 | 13.093 | 25,918 | -158.8% |

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
