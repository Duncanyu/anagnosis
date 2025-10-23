#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer SQLite by default for local dev
if [[ "${USE_SQLITE:-1}" == "1" ]]; then
  if [[ -n "${DATABASE_URL:-}" ]]; then
    echo "[anag] USE_SQLITE=1 → ignoring existing DATABASE_URL for local run"
  fi
  unset DATABASE_URL || true
fi

# Ensure CORS allows the local Web UI (7860)
if [[ -z "${ALLOW_ORIGINS:-}" ]]; then
  export ALLOW_ORIGINS="http://localhost:7860,http://127.0.0.1:7860"
else
  case ",${ALLOW_ORIGINS}," in
    *",http://localhost:7860,"*) :;;
    *) ALLOW_ORIGINS="${ALLOW_ORIGINS},http://localhost:7860";;
  esac
  case ",${ALLOW_ORIGINS}," in
    *",http://127.0.0.1:7860,"*) :;;
    *) ALLOW_ORIGINS="${ALLOW_ORIGINS},http://127.0.0.1:7860";;
  esac
  export ALLOW_ORIGINS
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[anag] Using SQLite at ./anagnosis.db (no DATABASE_URL set)"
else
  echo "[anag] Using DATABASE_URL=${DATABASE_URL}"
fi

# Sensible thread/env defaults for small single-instance runs
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"

PORT="${PORT:-8000}"
echo "[anag] Starting API on 127.0.0.1:${PORT}"
exec python3 -m gunicorn -k uvicorn.workers.UvicornWorker -w 1 --timeout 180 --graceful-timeout 180 -b "127.0.0.1:${PORT}" api.main:app
