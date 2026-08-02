# Multi-stage image serving both the FastAPI service and the Streamlit app.
#   API:  docker run -p 8000:8000 gridpulse
#   App:  docker run -p 8501:8501 -e GRIDPULSE_SERVICE=app gridpulse

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across source changes.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt \
 && pip install fastapi==0.115.6 "uvicorn[standard]==0.34.0" statsmodels==0.14.4

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e . --no-deps

COPY app.py ./
COPY dbt ./dbt
COPY orchestration ./orchestration
COPY artifacts ./artifacts
COPY data/gold ./data/gold

# Run unprivileged.
RUN useradd --create-home --uid 1000 gridpulse \
 && chown -R gridpulse:gridpulse /app
USER gridpulse

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || curl -fsS http://localhost:8501/_stcore/health || exit 1

COPY --chown=gridpulse:gridpulse deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]
