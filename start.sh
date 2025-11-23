#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT=${PORT:-7860}

echo "Activating virtualenv"
source antenv/bin/activate

echo "Starting API on port 8000"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

echo "Starting web frontend on port ${PORT} (foreground)"
python -m uvicorn web.app:app --host 0.0.0.0 --port "${PORT}" --workers 1

echo "Web process exited, shutting down API (pid=${API_PID})"
kill "${API_PID}" || true
wait "${API_PID}" || true
