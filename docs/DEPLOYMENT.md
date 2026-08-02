# Deployment guide

Three environments, one source of truth. Local is where the pipeline runs, GitHub
is the canonical repository, and Hugging Face Spaces hosts the public site.

```
   local laptop  ──push──►  GitHub  ──GitHub Action──►  Hugging Face Space
   (full pipeline)          (source of truth)           (public website)
```

---

## 1. Local

```powershell
cd C:\Users\HP\Desktop\ElectricityForecaster
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e . --no-deps

copy .env.example .env      # then add EIA_API_KEY and GROQ_API_KEY

gridpulse probe             # validate credentials, ~2 seconds
gridpulse all               # full pipeline
streamlit run app.py        # http://localhost:8501
```

`gridpulse all` produces the two things the public site needs:

| Artifact | Purpose |
|---|---|
| `data/gold/gridpulse_app.duckdb` | Slim warehouse slice the app reads |
| `artifacts/` | Trained models, leaderboard, headline metric |

Both are committed on purpose. Everything else under `data/` is regenerable and
excluded by `.gitignore`.

---

## 2. GitHub

Create an **empty public repository** named `gridpulse` at
<https://github.com/new> - no README, no .gitignore, no licence.

```powershell
cd C:\Users\HP\Desktop\ElectricityForecaster

git init
git add .
git commit -m "GridPulse: day-ahead grid demand forecasting platform"
git branch -M main
git remote add origin https://github.com/adwitiyashukla/gridpulse.git
git push -u origin main
```

### Repository secrets and variables

**Settings → Secrets and variables → Actions**

| Type | Name | Value |
|---|---|---|
| Secret | `EIA_API_KEY` | Your EIA key - powers the scheduled refresh |
| Secret | `HF_TOKEN` | A **write** token from <https://huggingface.co/settings/tokens> |
| Variable | `HF_USERNAME` | `adwitiyashukla` |
| Variable | `HF_SPACE` | `gridpulse` |

Once set, the CI badge goes green on the first push and the weekly refresh keeps
the deployed site current without any manual work.

---

## 3. Hugging Face Space

Create the Space at <https://huggingface.co/new-space>:

| Field | Value |
|---|---|
| Owner | `adwitiyashukla` |
| Space name | `gridpulse` |
| Licence | MIT |
| SDK | **Streamlit** |
| Hardware | CPU basic (free) |
| Visibility | Public |

Then add the agent key under **Settings → Variables and secrets**:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq key |

The `sync-huggingface.yml` workflow pushes to the Space on every push to `main`,
swapping in `deploy/README_SPACE.md` (which carries the YAML front matter the
Space requires) so the GitHub README stays clean.

### Manual push, if you prefer

```powershell
pip install "huggingface_hub[cli]"
huggingface-cli login
copy deploy\README_SPACE.md README_HF.md
huggingface-cli upload adwitiyashukla/gridpulse . . --repo-type=space
```

---

## 4. Docker (optional)

```bash
docker compose up --build
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

---

## Troubleshooting

**Space build fails on dependencies.** The root `requirements.txt` is deliberately
light - it excludes Dagster, dbt, Airflow and PyTorch, which the public app does
not need. Do not merge `requirements-dev.txt` into it.

**App shows "No warehouse found".** `data/gold/gridpulse_app.duckdb` was not
committed. Run `gridpulse export`, then `git add -f data/gold/gridpulse_app.duckdb`.

**App shows "Model artifacts are missing".** Run `gridpulse train`, then commit the
`artifacts/` directory.

**Export exceeds GitHub's 100 MB file limit.** Shrink the window:
`python -c "from gridpulse.warehouse.export import export_for_app; export_for_app(window_days=180)"`

**"Ask the Grid" is disabled.** `GROQ_API_KEY` is missing. Locally it belongs in
`.env`; on the Space it belongs in Settings → Variables and secrets.
