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
LightGBM hybrid (+ EIA forecast as input) gets 2.954% MAPE where the EIA's own published forecast gets 3.486%, which is 15.3% better, measured on 25,913 test hours from 2026-06-09 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 2.954 | 1,260 | 2,248 | 0.9952 | 3.553 | 25,913 | **+15.3%** |
| _EIA official forecast_ | 3.486 | 1,412 | 2,504 | 0.9941 | 2.873 | 25,210 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.711 | 1,543 | 2,561 | 0.9938 | 4.351 | 25,913 | -6.5% |
| **Ensemble** (GBM + LSTM) | 3.932 | 1,406 | 2,146 | 0.9956 | 3.061 | 25,913 | -12.8% |
| **LSTM** encoder | 4.759 | 1,492 | 2,163 | 0.9956 | 2.420 | 25,902 | -36.5% |
| Seasonal naive (24h) | 5.258 | 1,885 | 3,101 | 0.9909 | 4.845 | 25,913 | -50.8% |
| **Transformer** encoder | 6.639 | 2,526 | 3,668 | 0.9872 | 3.805 | 25,902 | -90.4% |
| Weekly naive (168h) | 8.981 | 3,444 | 5,781 | 0.9682 | 10.975 | 25,913 | -157.6% |

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
