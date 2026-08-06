# Single image serving either the Streamlit dashboard or the FastAPI service.
#
#   Hugging Face Space : uses the defaults below (Streamlit on port 7860)
#   Local dashboard    : docker run -p 7860:7860 gridpulse
#   Local API          : docker run -p 8000:8000 -e GRIDPULSE_SERVICE=api -e PORT=8000 gridpulse
#
# Port 7860 is the Hugging Face Spaces convention and is declared as `app_port`
# in the Space README front matter. Defaulting to it here means the Space needs
# no Docker-specific overrides.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRIDPULSE_SERVICE=app \
    PORT=7860 \
    HOME=/home/gridpulse

# libgomp1 is required by LightGBM. curl is used by the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Create the unprivileged user first. Hugging Face Spaces expect the container
# to run as UID 1000 with a writable home directory, because Streamlit writes
# its config and cache under $HOME/.streamlit.
RUN useradd --create-home --uid 1000 gridpulse
WORKDIR /app

# Dependencies before source, so the layer caches across code changes.
# requirements.txt is deliberately the light app-only set: no Dagster, dbt,
# Airflow, PyTorch or MLflow, none of which the served app needs.
COPY --chown=gridpulse:gridpulse requirements.txt ./
RUN pip install -r requirements.txt \
 && pip install fastapi==0.115.6 "uvicorn[standard]==0.34.0"

COPY --chown=gridpulse:gridpulse pyproject.toml ./
COPY --chown=gridpulse:gridpulse src ./src
RUN pip install -e . --no-deps

COPY --chown=gridpulse:gridpulse app.py ./
COPY --chown=gridpulse:gridpulse artifacts ./artifacts
COPY --chown=gridpulse:gridpulse data/gold ./data/gold
COPY --chown=gridpulse:gridpulse deploy/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh && chown -R gridpulse:gridpulse /app /home/gridpulse

USER gridpulse

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/_stcore/health" \
   || curl -fsS "http://localhost:${PORT}/health" \
   || exit 1

ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]
