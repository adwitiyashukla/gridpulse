# Deployment guide

```
                                       ┌── Streamlit Cloud ──► gridpulse-ai.streamlit.app
   local laptop  ──push──►  GitHub  ───┤
   (full pipeline)     (source of truth)└── GitHub Actions ───► HF Docker Space
```

One commit, two public deployments, both updated automatically on push.

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

`gridpulse all` creates the two things the website needs:

| File | What it is |
|---|---|
| `data/gold/gridpulse_app.duckdb` | A cut-down copy of the warehouse, about 13 MB |
| `artifacts/` | The trained models, the leaderboard and the headline number |

I commit both of these on purpose, so the website starts instantly without having
to build anything. Everything else under `data/` can be regenerated, so it is in
`.gitignore`, including the 129 MB full warehouse.

### Running one step at a time

```powershell
gridpulse probe       # check the API keys work and the responses look right
gridpulse ingest      # download EIA + weather data into the bronze layer
gridpulse build       # bronze -> silver -> gold star schema
gridpulse quality     # run the 16 data quality checks
gridpulse train       # train and score every model
gridpulse anomalies   # find odd hours using three detectors
gridpulse export      # write the small database the website uses
```

### The other services

```powershell
dagster dev -f orchestration/dagster_app/definitions.py   # Dagster UI, port 3000
uvicorn gridpulse.api.main:app --reload --port 8000       # the API, with docs
streamlit run app.py                                      # the website, port 8501
cd dbt/gridpulse; dbt build --profiles-dir .              # build the dbt models
pytest -q                                                 # run all the tests
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

| Type | Name | What it is for |
|---|---|---|
| Secret | `EIA_API_KEY` | Lets the weekly refresh download new data |
| Secret | `HF_TOKEN` | Lets the sync workflow push to the Hugging Face Space. Needs write permission |

CI still passes without either of these, because the tests need no internet and no
keys. Only the refresh and the Space sync need them.

### The workflows

| Workflow | When it runs | What it does |
|---|---|---|
| `ci.yml` | Every push and pull request | Runs the linter, runs the tests on Python 3.10, 3.11 and 3.12, checks the dbt project parses and the Dagster assets load |
| `refresh.yml` | Every Monday, or when I run it manually | Downloads new data, rebuilds, checks quality, retrains, rebuilds the README results table, commits, and asks the sync workflow to update the Space |
| `sync-huggingface.yml` | Every push to `main`, or when the refresh asks it to | Copies the repo up to the Hugging Face Space, which then rebuilds the Docker image |

---

## 3. Streamlit Community Cloud

**Live at <https://gridpulse-ai.streamlit.app>.**

Free, made specifically for Streamlit apps, and it deploys straight from GitHub.
Every push to `main` redeploys it automatically.

> Hugging Face used to have a Streamlit option for Spaces and removed it. Now you
> can only pick Gradio, Docker or a static site. So the Space below runs my own
> `Dockerfile` instead. Making a Docker Space on a personal account needs a PRO
> subscription, though the CPU hardware it runs on is free.

### Steps

1. Sign in at <https://share.streamlit.io> with GitHub and authorise access.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `adwitiyashukla/gridpulse`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings**:
   - **Python version:** change it to **3.11**. The version it offers by default
     is newer than `pyproject.toml` allows (`>=3.10,<3.13`), and some of the
     packages I pinned have no build for it yet, so the deploy would fail.
   - **Secrets:** paste this in to turn on the AI tab. It is optional, the other
     six tabs work fine without it:
     ```toml
     GROQ_API_KEY = "your_groq_key"
     ```
5. Click **Deploy**. The first build takes 3 to 5 minutes.

Then put the URL you get into the repo's **About** panel so it shows up at the top
right of the GitHub page.

### Why it fits in the free tier

The main `requirements.txt` has only 11 packages, and I deliberately left out
Dagster, dbt, Airflow, PyTorch and MLflow, because the website does not need any
of them. Making a prediction just means loading the already-trained LightGBM file,
which takes milliseconds.

---

## 4. Hugging Face Docker Space

**Live at <https://huggingface.co/spaces/adwitiyashukla/gridpulse>.**

The Space runs my own `Dockerfile`, so the container running the public website is
exactly the same one `docker compose up` builds on my laptop. Nothing about it is
specific to Hugging Face.

### How the sync works

`.github/workflows/sync-huggingface.yml` runs on every push to `main`:

1. Checks out the repo.
2. Copies it into a `.hf/` folder, leaving out `data/bronze`, `data/silver` and
   `mlruns` since the Space does not need them.
3. Swaps in `deploy/README_SPACE.md` as the README, because a Space needs some
   settings at the top of its README (`sdk: docker`, `app_port: 7860`) that would
   look out of place on the GitHub one.
4. Swaps in `deploy/gitattributes_space.txt` as `.gitattributes`. This one is
   important: without it the big model files come out as placeholders instead of
   real files. Bug 12 in the [engineering log](ENGINEERING_LOG.md) is the full
   story.
5. Checks the three biggest files are real files and not placeholders, and fails
   the build if they are not.
6. Uploads everything using `HfApi.upload_folder`. I use the Python library rather
   than the command line tool because the `huggingface-cli` command was retired.

If `HF_TOKEN` is missing the workflow finishes quietly instead of failing, so
anyone who forks this repo without setting it up does not end up with a red
Actions tab.

### Setting it up the first time

| Where | What to do |
|---|---|
| <https://huggingface.co/settings/tokens> | Create a token with **write** permission |
| GitHub, Settings, Secrets, Actions | Add a secret called `HF_TOKEN` with that token |
| Space, Settings, Variables and secrets | Add `GROQ_API_KEY` as the raw key, not TOML |
| <https://huggingface.co/new-space> | Pick **Docker, Blank**, hardware **CPU basic**, and make it public |

If you rename the Space, you can set the repository variables `HF_USERNAME` and
`HF_SPACE` to point at the new one.

### Why port 7860

Hugging Face expects the container to listen on port 7860 and to run as user ID
1000 with a home folder it can write to, because Streamlit saves its settings and
cache under `$HOME/.streamlit`. My `Dockerfile` already does both, so the Space
needs no extra configuration.

`entrypoint.sh` also turns off Streamlit's CORS and XSRF protection. Hugging Face
shows the app inside an iframe on a different domain, and Streamlit's default XSRF
check blocks that. The app is read-only and nobody logs into it, so there is no
saved state for a cross-site request to mess with.

---

## 5. Docker (optional)

```bash
docker compose up --build
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

---

## If something goes wrong

**The app says "No warehouse found".** The `data/gold/gridpulse_app.duckdb` file
did not get committed. Run `gridpulse export`, then
`git add -f data/gold/gridpulse_app.duckdb`.

**The app says "Model artifacts are missing".** Run `gridpulse train`, then commit
the `artifacts/` folder.

**The export file is bigger than GitHub's 100 MB limit.** Export fewer days:
`python -c "from gridpulse.warehouse.export import export_for_app; export_for_app(window_days=180)"`

**"Ask the Grid" is greyed out.** `GROQ_API_KEY` is missing. On your laptop it
goes in `.env`. When deployed it goes in whatever secrets box that host provides.

**The Hugging Face Space says the DuckDB file is not valid.** The big files came
across as LFS placeholders instead of real files. Check that
`deploy/gitattributes_space.txt` is being copied over as `.gitattributes` in the
sync workflow. Bug 12 in the [engineering log](ENGINEERING_LOG.md) explains this
one properly.

**Streamlit is showing old data after a rebuild.** The app clears its caches
automatically when the data files change, but if you are mid-session you can press
**C** in the browser to clear the cache and **R** to rerun. If you edited a Python
file other than `app.py`, restart the server completely, because Streamlit's hot
reload does not reimport modules it has already loaded.
