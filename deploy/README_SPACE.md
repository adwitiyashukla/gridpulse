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

# ⚡ GridPulse

Day-ahead electricity demand forecasting for US balancing authorities,
benchmarked against the **EIA's own published day-ahead forecast**.

| Model | MAPE | vs EIA |
|---|---|---|
| **LightGBM hybrid** (+ EIA forecast as input) | **2.771%** | **+25.0%** |
| **LightGBM** (global, from first principles) | 3.676% | +0.6% |
| _EIA official forecast_ | 3.696% | benchmark |
| Seasonal naive (24h) | 5.677% | -53.6% |
| Weekly naive (168h) | 9.117% | -146.7% |

Measured over 25,927 out-of-sample hours across 12 balancing authorities, using a
strictly chronological split.

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

The **Ask the Grid** tab needs a `GROQ_API_KEY` under
**Settings, Variables and secrets**. Keys are free at
[console.groq.com/keys](https://console.groq.com/keys). Everything else works with
no setup at all.

## Source

Full pipeline, tests, orchestration and documentation:
**[github.com/adwitiyashukla/gridpulse](https://github.com/adwitiyashukla/gridpulse)**

Data: [US EIA Form 930](https://www.eia.gov/opendata/) and
[Open-Meteo](https://open-meteo.com/).
