# Deployment guide

```
   local laptop  ──push──►  GitHub  ──Streamlit Cloud──►  public URL
   (full pipeline)          (source of truth)             (optional)
```

---

## 1. Local

```powershell
cd C:\Users\HP\Desktop\ElectricityForecaster
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps

copy .env.example .env      # add EIA_API_KEY, optionally GROQ_API_KEY

gridpulse probe             # validate credentials, about 2 seconds
gridpulse all               # full pipeline
streamlit run app.py        # http://localhost:8501
```

`gridpulse all` produces the two things the app needs:

| Artifact | Purpose |
|---|---|
| `data/gold/gridpulse_app.duckdb` | Slim warehouse slice, about 13 MB |
| `artifacts/` | Trained models, leaderboard, headline metric |

Both are committed on purpose. Everything else under `data/` is regenerable and
excluded by `.gitignore`, including the 129 MB development warehouse.

### Individual stages

```powershell
gridpulse probe       # validate API keys and response contracts
gridpulse ingest      # extract EIA + weather into bronze (incremental)
gridpulse build       # bronze -> silver -> gold star schema
gridpulse quality     # 16 data quality checks
gridpulse train       # train and score every model
gridpulse anomalies   # three-detector anomaly consensus
gridpulse export      # write the slim artifact for the app
```

### Services

```powershell
dagster dev -f orchestration/dagster_app/definitions.py   # asset lineage, :3000
uvicorn gridpulse.api.main:app --reload --port 8000       # REST API + OpenAPI docs
streamlit run app.py                                      # dashboard, :8501
cd dbt/gridpulse; dbt build --profiles-dir .              # analytics marts
pytest -q                                                 # full test suite
```

---

## 2. GitHub

Live at <https://github.com/adwitiyashukla/gridpulse>.

To push further changes:

```powershell
git add -A
git commit -m "your message"
git push
```

### Repository secrets

**Settings → Secrets and variables → Actions**

| Type | Name | Purpose |
|---|---|---|
| Secret | `EIA_API_KEY` | Powers the weekly scheduled refresh workflow |

Without it CI still passes, because the test suite needs no network and no
credentials. Only the scheduled refresh is skipped.

### Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push and pull request | Ruff lint, pytest on Python 3.10/3.11/3.12, dbt parse, Dagster asset graph load |
| `refresh.yml` | Weekly cron, or manual | Re-ingests EIA and weather, rebuilds, validates, retrains, commits updated artifacts |

---

## 3. Public website

**Live at <https://gridpulse-ai.streamlit.app>.**

**Streamlit Community Cloud** is the right host for this app: free, purpose-built
for Streamlit, and it deploys straight from GitHub. Every push to `main`
redeploys automatically.

> Hugging Face Spaces is not an option. Hugging Face removed the Streamlit SDK
> entirely; its API now accepts only `gradio`, `docker` or `static`. Gradio on the
> free tier is locked to ZeroGPU, which is Gradio-only, and Docker requires a PRO
> subscription. There is no free path to a Streamlit Space.

### Steps

1. Sign in at <https://share.streamlit.io> with GitHub and authorise access.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `adwitiyashukla/gridpulse`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings**:
   - **Python version:** set it to **3.11**. The default offered is newer than
     `pyproject.toml` allows (`>=3.10,<3.13`), and several pinned wheels do not
     yet build against it.
   - **Secrets:** paste this to enable the AI agent (optional; the other six tabs
     work without it):
     ```toml
     GROQ_API_KEY = "your_groq_key"
     ```
5. Click **Deploy**. The first build takes 3 to 5 minutes.

Add the resulting URL to the repository's **About** panel so it appears at the
top right of the GitHub page.

### Why it fits the free tier

The root `requirements.txt` holds 11 packages and deliberately excludes Dagster,
dbt, Airflow, PyTorch and MLflow, none of which the app needs. Inference runs on
the committed LightGBM artifact, which loads in milliseconds.

---

## 4. Docker (optional)

```bash
docker compose up --build
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

---

## Troubleshooting

**App shows "No warehouse found".** `data/gold/gridpulse_app.duckdb` was not
committed. Run `gridpulse export`, then
`git add -f data/gold/gridpulse_app.duckdb`.

**App shows "Model artifacts are missing".** Run `gridpulse train`, then commit
the `artifacts/` directory.

**Export exceeds GitHub's 100 MB file limit.** Shrink the window:
`python -c "from gridpulse.warehouse.export import export_for_app; export_for_app(window_days=180)"`

**"Ask the Grid" is disabled.** `GROQ_API_KEY` is missing. Locally it belongs in
`.env`; when deployed it belongs in the host's secrets.

**Streamlit shows stale data after a rebuild.** Streamlit caches queries for 15
minutes. Press **C** in the browser to clear the cache, then **R** to rerun. If
you changed a Python module rather than `app.py`, restart the server entirely:
hot reload does not re-import modules that are already loaded.
