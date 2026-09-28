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
LightGBM hybrid (+ EIA forecast as input) gets 3.007% MAPE where the EIA's own published forecast gets 3.448%, which is 12.8% better, measured on 25,902 test hours from 2026-06-30 onwards across 12 balancing authorities.

| Model | MAPE % | MAE (MW) | RMSE (MW) | R2 | Peak-hour MAPE % | Hours scored | Skill vs EIA |
|---|---|---|---|---|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | 3.007 | 1,320 | 2,349 | 0.9947 | 3.499 | 25,902 | **+12.8%** |
| _EIA official forecast_ | 3.448 | 1,389 | 2,454 | 0.9943 | 2.586 | 25,200 | - (benchmark) |
| **LightGBM** (global, quantile) | 3.850 | 1,673 | 2,798 | 0.9925 | 4.756 | 25,902 | -11.7% |
| **Ensemble** (GBM + LSTM) | 4.537 | 1,778 | 2,753 | 0.9928 | 3.867 | 25,902 | -31.6% |
| Seasonal naive (24h) | 5.074 | 1,785 | 2,923 | 0.9918 | 4.350 | 25,902 | -47.2% |
| **LSTM** encoder | 5.728 | 2,036 | 3,121 | 0.9907 | 3.268 | 25,878 | -66.1% |
| **Transformer** encoder | 6.473 | 2,765 | 4,163 | 0.9834 | 6.090 | 25,878 | -87.7% |
| Weekly naive (168h) | 9.210 | 3,401 | 5,729 | 0.9686 | 11.360 | 25,902 | -167.1% |

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
