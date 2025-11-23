#!/usr/bin/env bash
set -euo pipefail

# Lightweight startup script for Azure Web App
# - Starts the API on port 8000 in the background
# - Starts the web frontend on $PORT (Azure-provided) in the foreground
# Note: For production, consider using a process manager (supervisord) or a container image.

PORT=${PORT:-7860}

echo "Starting API on port 8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

echo "Starting web frontend on port ${PORT} (foreground)"
uvicorn web.app:app --host 0.0.0.0 --port "${PORT}" --workers 1

# If web exits, shut down API gracefully
echo "Web process exited, shutting down API (pid=${API_PID})"
kill ${API_PID} || true
wait ${API_PID} || true
