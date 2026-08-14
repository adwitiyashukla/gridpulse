#!/usr/bin/env bash
set -euo pipefail

SERVICE="${GRIDPULSE_SERVICE:-app}"
PORT="${PORT:-7860}"

echo "Starting GridPulse service='${SERVICE}' on port ${PORT}"

case "$SERVICE" in
  app)
    exec streamlit run app.py \
      --server.port "${PORT}" \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection false \
      --server.fileWatcherType none \
      --browser.gatherUsageStats false
    ;;
  api)
    exec uvicorn gridpulse.api.main:app --host 0.0.0.0 --port "${PORT}"
    ;;
  *)
    echo "Unknown GRIDPULSE_SERVICE='${SERVICE}'. Expected 'app' or 'api'." >&2
    exit 1
    ;;
esac
