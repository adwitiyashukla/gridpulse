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
| **Forecast** | Generate a live 24-hour demand forecast with P10-P90 intervals for any balancing authority, using real-time weather |
| **Explorer** | Historical demand, the V-shaped temperature response, weekday vs weekend load shapes |
| **Model Leaderboard** | Every model scored against the EIA's official forecast on identical out-of-sample hours |
| **Anomalies** | Suspect hours flagged by three independent detectors voting in consensus |
| **Data Quality** | Scorecard across six classical data quality dimensions |
| **Ask the Grid** | Ask a question in plain English, get guarded SQL and a chart back |

## What sits behind it

This Space is the front end of a full data platform:

- Incremental, watermarked extraction from the EIA Open Data API and Open-Meteo
- A medallion lakehouse on DuckDB with a Kimball star schema
- 16 declarative data quality checks across completeness, validity, consistency,
  timeliness, uniqueness and accuracy
- LightGBM with P10/P50/P90 quantile bands, plus PyTorch LSTM and Transformer
  encoders for comparison
- Three-detector anomaly consensus: robust seasonal z-score, Isolation Forest and
  a daily-load-shape autoencoder
- A guarded LLM text-to-SQL agent (read-only connection, SELECT-only, table
  allowlist, enforced row cap)

## Configuration

The **Ask the Grid** tab needs a `GROQ_API_KEY` under
**Settings → Variables and secrets**. Free keys at
[console.groq.com/keys](https://console.groq.com/keys). Everything else works
without any configuration.

## Source

Full pipeline, tests, orchestration and documentation:
**[github.com/adwitiyashukla/gridpulse](https://github.com/adwitiyashukla/gridpulse)**

Data: [US EIA Form 930](https://www.eia.gov/opendata/) and
[Open-Meteo](https://open-meteo.com/).
