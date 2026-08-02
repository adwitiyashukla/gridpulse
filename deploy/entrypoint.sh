#!/usr/bin/env bash
set -euo pipefail

SERVICE="${GRIDPULSE_SERVICE:-api}"

case "$SERVICE" in
  api)
    exec uvicorn gridpulse.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  app)
    exec streamlit run app.py \
      --server.port "${PORT:-8501}" \
      --server.address 0.0.0.0 \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  *)
    echo "Unknown GRIDPULSE_SERVICE='$SERVICE'. Expected 'api' or 'app'." >&2
    exit 1
    ;;
esac
