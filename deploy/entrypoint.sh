#!/usr/bin/env bash
# Container entrypoint. Selects which service to run via GRIDPULSE_SERVICE.
#
#   app (default) : Streamlit dashboard
#   api           : FastAPI service
set -euo pipefail

SERVICE="${GRIDPULSE_SERVICE:-app}"
PORT="${PORT:-7860}"

echo "Starting GridPulse service='${SERVICE}' on port ${PORT}"

case "$SERVICE" in
  app)
    # CORS and XSRF protection are disabled because Hugging Face Spaces serve
    # the app inside an iframe on a different origin, which Streamlit's default
    # XSRF check rejects. The app is read-only and takes no authenticated
    # input, so there is no state for a cross-site request to tamper with.
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
