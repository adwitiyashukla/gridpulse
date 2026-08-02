# Project status

**Status:** Complete. Pipeline, models, quality suite and public app all working.

For the interesting part - every bug found while building this, its diagnosis and
its fix - see **[ENGINEERING_LOG.md](ENGINEERING_LOG.md)**.

---

## Verified end to end

| Stage | Result |
|---|---|
| Test suite | All passing, no network or API keys required |
| `gridpulse probe` | EIA API v2.1.13, contract validated |
| `gridpulse ingest` | 3,185,904 EIA rows + 802,080 weather rows, 12 balancing authorities |
| `gridpulse build` | 797,677 hourly fact rows, 794,835 carrying the EIA benchmark |
| `gridpulse quality` | 16/16 checks passed, score 100% |
| `gridpulse train` | Best model 2.771% MAPE, 25.0% better than the EIA benchmark |
| `gridpulse anomalies` | 20,986 anomalies flagged across three detectors |
| `gridpulse export` | 13.4 MB app artifact, 9 tables |

## Headline

| Model | MAPE | vs EIA |
|---|---|---|
| LightGBM hybrid (+ EIA forecast as input) | 2.771% | +25.0% |
| LightGBM (global, from first principles) | 3.676% | +0.55% |
| EIA official day-ahead forecast | 3.696% | benchmark |
| Seasonal naive (24h) | 5.677% | -53.6% |
| Weekly naive (168h) | 9.117% | -146.7% |

Measured on 25,927 out-of-sample hours across 12 balancing authorities, using a
strictly chronological split.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps

copy .env.example .env      # add EIA_API_KEY, optionally GROQ_API_KEY

gridpulse probe
gridpulse all
streamlit run app.py
```

Deployment steps for GitHub and Hugging Face are in
**[DEPLOYMENT.md](DEPLOYMENT.md)**.
Interview preparation notes are in **[INTERVIEW_NOTES.md](INTERVIEW_NOTES.md)**.
