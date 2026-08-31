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
LightGBM hybrid (+ EIA forecast as input) gets 2.907% MAPE where the EIA's own published forecast gets 3.521%, which is 17.4% better, measured on 25,931 test hours from 2026-06-02 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.907 | 1,198 | 2,107 | 0.9957 | 3.239 | 25,931 | **+17.4%** |
| _EIA official forecast_ | 3.521 | 1,406 | 2,506 | 0.9940 | 2.900 | 25,307 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.694 | 1,497 | 2,451 | 0.9941 | 4.066 | 25,931 | -4.9% |
| **Ensemble** (GBM + LSTM) | 4.565 | 1,729 | 2,606 | 0.9934 | 3.214 | 25,931 | -29.6% |
| Seasonal naive (24h) | 5.231 | 1,845 | 3,002 | 0.9912 | 4.500 | 25,931 | -48.6% |
| **Transformer** encoder | 5.745 | 2,225 | 3,322 | 0.9893 | 4.003 | 25,920 | -63.2% |
| **LSTM** encoder | 5.923 | 2,115 | 3,168 | 0.9902 | 2.628 | 25,920 | -68.2% |
| Weekly naive (168h) | 8.833 | 3,349 | 5,679 | 0.9686 | 11.008 | 25,931 | -150.9% |

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
