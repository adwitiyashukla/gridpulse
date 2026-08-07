# Project status

**Status:** Done. The pipeline, the models, the quality checks and the public
website are all working, and the whole thing refreshes itself once a week.

Live at [Hugging Face](https://huggingface.co/spaces/adwitiyashukla/gridpulse) and
[Streamlit](https://gridpulse-ai.streamlit.app).

The most useful part of this repo is probably
**[ENGINEERING_LOG.md](ENGINEERING_LOG.md)**, where I wrote up every bug I hit,
what caused it and how I fixed it.

---

## What I checked works, end to end

| Step | What happened |
|---|---|
| Test suite | All tests pass, and they need no internet and no API keys |
| `gridpulse probe` | Talks to the EIA API and confirms the responses look how I expect |
| `gridpulse ingest` | About 3.2 million EIA rows and 800,000 weather rows, for 12 regions |
| `gridpulse build` | Roughly 800,000 hourly rows, almost all of them with EIA's forecast attached |
| `gridpulse quality` | 16 out of 16 checks passing |
| `gridpulse train` | The best model beats EIA's own forecast by around 24% |
| `gridpulse anomalies` | About 21,000 unusual hours flagged by three detectors |
| `gridpulse export` | A 13 MB database for the website, with 9 tables |

I have kept the numbers above approximate on purpose. The exact figures change
every Monday when the refresh workflow retrains everything, and the live versions
are always in `artifacts/leaderboard.json` and in the results table in the main
[README](../README.md), which gets rebuilt automatically after each retrain.

## Running it yourself

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps

copy .env.example .env      # add EIA_API_KEY, and GROQ_API_KEY if you want the AI tab

gridpulse probe
gridpulse all
streamlit run app.py
```

`gridpulse all` takes roughly 20 to 40 minutes on a normal laptop, mostly
downloading data and training.

How to deploy it is in **[DEPLOYMENT.md](DEPLOYMENT.md)**.
Questions I expect to be asked in interviews, and my answers, are in
**[INTERVIEW_NOTES.md](INTERVIEW_NOTES.md)**.
