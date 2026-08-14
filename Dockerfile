FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRIDPULSE_SERVICE=app \
    PORT=7860 \
    HOME=/home/gridpulse

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 gridpulse
WORKDIR /app

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
