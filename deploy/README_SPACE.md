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
LightGBM hybrid (+ EIA forecast as input) gets 2.957% MAPE where the EIA's own published forecast gets 3.344%, which is 11.6% better, measured on 25,887 test hours from 2026-07-21 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.957 | 1,238 | 2,175 | 0.9949 | 2.622 | 25,887 | **+11.6%** |
| _EIA official forecast_ | 3.344 | 1,263 | 2,232 | 0.9947 | 2.545 | 25,192 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.745 | 1,549 | 2,566 | 0.9929 | 4.278 | 25,887 | -12.0% |
| **Ensemble** (GBM + LSTM) | 4.261 | 1,550 | 2,418 | 0.9938 | 3.170 | 25,887 | -27.4% |
| Seasonal naive (24h) | 5.172 | 1,731 | 2,873 | 0.9912 | 4.375 | 25,887 | -54.7% |
| **LSTM** encoder | 5.303 | 1,716 | 2,724 | 0.9921 | 2.651 | 25,866 | -58.6% |
| **Transformer** encoder | 6.396 | 2,269 | 3,339 | 0.9881 | 3.388 | 25,866 | -91.3% |
| Weekly naive (168h) | 9.325 | 3,201 | 5,146 | 0.9717 | 10.026 | 25,887 | -178.9% |

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
