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
LightGBM hybrid (+ EIA forecast as input) gets 2.869% MAPE where the EIA's own published forecast gets 3.324%, which is 13.7% better, measured on 25,857 test hours from 2026-08-04 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.869 | 1,158 | 2,117 | 0.9949 | 2.247 | 25,857 | **+13.7%** |
| _EIA official forecast_ | 3.324 | 1,188 | 2,131 | 0.9949 | 2.445 | 25,097 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.503 | 1,392 | 2,367 | 0.9936 | 3.999 | 25,857 | -5.4% |
| **Ensemble** (GBM + LSTM) | 4.429 | 1,544 | 2,379 | 0.9935 | 3.498 | 25,857 | -33.2% |
| Seasonal naive (24h) | 4.930 | 1,586 | 2,658 | 0.9919 | 4.264 | 25,857 | -48.3% |
| **LSTM** encoder | 5.828 | 1,822 | 2,786 | 0.9911 | 3.231 | 25,836 | -75.3% |
| **Transformer** encoder | 6.088 | 2,172 | 3,305 | 0.9875 | 5.512 | 25,836 | -83.1% |
| Weekly naive (168h) | 9.279 | 3,013 | 4,813 | 0.9735 | 9.046 | 25,857 | -179.1% |

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
