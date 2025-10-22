#!/usr/bin/env bash
set -euo pipefail

# Run the Web UI locally without Docker/Compose.
# Expects the API to be running locally on port 8000 by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-7860}"
export API_INTERNAL_BASE="${API_INTERNAL_BASE:-http://localhost:8000}"

echo "[anag] Starting Web on 127.0.0.1:${PORT} (API: ${API_INTERNAL_BASE})"
exec python3 -m gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b "127.0.0.1:${PORT}" web.app:app

