from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.core.config import load_config, present_keys, save_secret

router = APIRouter(prefix="/api", tags=["settings"])


USERS_DIR = pathlib.Path("artifacts") / "users"


def _user_dir(user_id: str) -> pathlib.Path:
    d = USERS_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prefs_path(user_id: str) -> pathlib.Path:
    return _user_dir(user_id) / "prefs.json"


def _secrets_path(user_id: str) -> pathlib.Path:
    return _user_dir(user_id) / "secrets.json"


def read_user_prefs(user_id: str) -> Dict[str, Any]:
    p = _prefs_path(user_id)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_user_prefs(user_id: str, data: Dict[str, Any]) -> None:
    p = _prefs_path(user_id)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def read_user_secrets(user_id: str) -> Dict[str, Any]:
    s = _secrets_path(user_id)
    try:
        if s.exists():
            return json.loads(s.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_user_secrets(user_id: str, data: Dict[str, Any]) -> None:
    s = _secrets_path(user_id)
    try:
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _bool(val: Any) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _apply_pref_env(prefs: Dict[str, Any]) -> None:
    bool_map = {
        "MEMORY_ENABLED": prefs.get("MEMORY_ENABLED", os.environ.get("MEMORY_ENABLED", "false")),
        "ASK_EXHAUSTIVE": prefs.get("ASK_EXHAUSTIVE", os.environ.get("ASK_EXHAUSTIVE", "false")),
    }
    for key, val in bool_map.items():
        os.environ[key] = "true" if _bool(val) else "false"

    numeric_defaults = {
        "MEMORY_TOKEN_LIMIT": 1200,
        "MEMORY_FILE_LIMIT_MB": 50,
        "ASK_TIME_BUDGET_SEC": 120,
        "ASK_MAX_BATCHES": 6,
        "ASK_BATCH_CHAR_BUDGET": 12000,
        "ASK_CANDIDATES": 300,
    }
    for key, default in numeric_defaults.items():
        value = prefs.get(key) or os.environ.get(key) or default
        os.environ[key] = str(value)

    reranker = prefs.get("ASK_RERANKER") or os.environ.get("ASK_RERANKER", "off")
    os.environ["ASK_RERANKER"] = str(reranker)
    os.environ.setdefault("ASK_AGENTS", "false")


def _settings_defaults_for_user(user_id: str) -> Dict[str, Any]:
    # Global defaults
    try:
        global_cfg = load_config() or {}
    except Exception:
        global_cfg = {}
    # User-specific overrides
    secrets = read_user_secrets(user_id)
    prefs = read_user_prefs(user_id)
    env = os.environ
    def pick(key: str, default: Any = ""):
        return secrets.get(key) or global_cfg.get(key) or default
    return {
        "OPENAI_API_KEY": pick("OPENAI_API_KEY", ""),
        "HF_TOKEN": pick("HF_TOKEN", ""),
        "SERPAPI_KEY": pick("SERPAPI_KEY", ""),
        "BRAVE_API_KEY": pick("BRAVE_API_KEY", ""),
        "OPENAI_CHAT_MODEL": secrets.get("OPENAI_CHAT_MODEL") or env.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "HF_LLM_NAME": secrets.get("HF_LLM_NAME") or env.get("HF_LLM_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "EMBED_BACKEND": secrets.get("EMBED_BACKEND") or "hf",
        "LLM_BACKEND": secrets.get("LLM_BACKEND") or "openai",
        "MEMORY_ENABLED": _bool(prefs.get("MEMORY_ENABLED") or env.get("MEMORY_ENABLED", "false")),
        "MEMORY_TOKEN_LIMIT": int(prefs.get("MEMORY_TOKEN_LIMIT") or env.get("MEMORY_TOKEN_LIMIT", "1200")),
        "MEMORY_FILE_LIMIT_MB": int(prefs.get("MEMORY_FILE_LIMIT_MB") or env.get("MEMORY_FILE_LIMIT_MB", "50")),
        "OPENAI_TPM": int(prefs.get("OPENAI_TPM") or env.get("OPENAI_TPM", "0")),
        "OPENAI_RPM": int(prefs.get("OPENAI_RPM") or env.get("OPENAI_RPM", "0")),
        "ASK_BATCH_CHAR_BUDGET": int(prefs.get("ASK_BATCH_CHAR_BUDGET") or env.get("ASK_BATCH_CHAR_BUDGET", "12000")),
        "ASK_MAX_BATCHES": int(prefs.get("ASK_MAX_BATCHES") or env.get("ASK_MAX_BATCHES", "6")),
        "ASK_TIME_BUDGET_SEC": int(prefs.get("ASK_TIME_BUDGET_SEC") or env.get("ASK_TIME_BUDGET_SEC", "120")),
        "ASK_EXHAUSTIVE": _bool(prefs.get("ASK_EXHAUSTIVE") or env.get("ASK_EXHAUSTIVE", "false")),
        "ASK_RERANKER": (prefs.get("ASK_RERANKER") or env.get("ASK_RERANKER", "off")).lower(),
        "ASK_CANDIDATES": int(prefs.get("ASK_CANDIDATES") or env.get("ASK_CANDIDATES", "300")),
    }


def user_key_status(user_id: str) -> Dict[str, bool]:
    s = read_user_secrets(user_id)
    if not s:
        # fallback to global key presence
        keys = present_keys()
        return {
            "openai": bool(keys.get("OPENAI_API_KEY")),
            "hf": bool(keys.get("HF_TOKEN")),
            "serpapi": bool(keys.get("SERPAPI_KEY")),
            "brave": bool(keys.get("BRAVE_API_KEY")),
        }
    return {
        "openai": bool(s.get("OPENAI_API_KEY")),
        "hf": bool(s.get("HF_TOKEN")),
        "serpapi": bool(s.get("SERPAPI_KEY")),
        "brave": bool(s.get("BRAVE_API_KEY")),
    }


def _save_settings_for_user(user_id: str, payload: Dict[str, Any]) -> str:
    try:
        secrets = read_user_secrets(user_id)
        if payload.get("openai_key", "").strip():
            secrets["OPENAI_API_KEY"] = payload["openai_key"].strip()
        if payload.get("hf_token", "").strip():
            secrets["HF_TOKEN"] = payload["hf_token"].strip()
        if payload.get("serp_key", "").strip():
            secrets["SERPAPI_KEY"] = payload["serp_key"].strip()
        if payload.get("brave_key", "").strip():
            secrets["BRAVE_API_KEY"] = payload["brave_key"].strip()
        secrets["OPENAI_CHAT_MODEL"] = (payload.get("openai_model", "gpt-4o-mini").strip() or "gpt-4o-mini")
        secrets["HF_LLM_NAME"] = (payload.get("hf_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip() or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        secrets["EMBED_BACKEND"] = (payload.get("embed_backend") or "hf").lower()
        secrets["LLM_BACKEND"] = (payload.get("llm_backend") or "openai").lower()
        write_user_secrets(user_id, secrets)

        prefs = read_user_prefs(user_id)
        prefs.update(
            {
                "MEMORY_ENABLED": "true" if payload.get("memory_enabled") else "false",
                "MEMORY_TOKEN_LIMIT": int(payload.get("memory_tokens", 1200)),
                "MEMORY_FILE_LIMIT_MB": int(payload.get("memory_file_mb", 50)),
                "OPENAI_TPM": int(payload.get("openai_tpm", 0)),
                "OPENAI_RPM": int(payload.get("openai_rpm", 0)),
                "ASK_BATCH_CHAR_BUDGET": int(payload.get("ask_char_budget", 12000)),
                "ASK_MAX_BATCHES": int(payload.get("ask_max_batches", 6)),
                "ASK_TIME_BUDGET_SEC": int(payload.get("ask_time_budget", 120)),
                "ASK_EXHAUSTIVE": "true" if payload.get("ask_exhaustive") else "false",
                "ASK_RERANKER": (payload.get("ask_reranker") or "off").lower(),
                "ASK_CANDIDATES": int(payload.get("ask_candidates", 300)),
            }
        )
        write_user_prefs(user_id, prefs)
        return "Settings saved."
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/settings")
async def get_settings(user: User = Depends(require_auth)) -> JSONResponse:
    defaults = _settings_defaults_for_user(str(user.id))
    status = user_key_status(str(user.id))
    return JSONResponse({"defaults": defaults, "keys": status})


@router.post("/settings")
async def post_settings(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    payload = await request.json()
    message = _save_settings_for_user(str(user.id), payload)
    status = _settings_defaults_for_user(str(user.id))
    key_status = user_key_status(str(user.id))
    return JSONResponse({"message": message, "defaults": status, "keys": key_status})

# Helper to apply (and optionally restore) per-user environment for a single job
def apply_env_for_user(user_id: str):
    defaults = _settings_defaults_for_user(user_id)
    # snapshot current env for keys we modify
    touched = {}
    keys = [
        "OPENAI_CHAT_MODEL","HF_LLM_NAME","EMBED_BACKEND","LLM_BACKEND",
        "MEMORY_ENABLED","MEMORY_TOKEN_LIMIT","MEMORY_FILE_LIMIT_MB",
        "OPENAI_TPM","OPENAI_RPM","ASK_BATCH_CHAR_BUDGET","ASK_MAX_BATCHES",
        "ASK_TIME_BUDGET_SEC","ASK_EXHAUSTIVE","ASK_RERANKER","ASK_CANDIDATES",
        "OPENAI_API_KEY","HF_TOKEN","SERPAPI_KEY","BRAVE_API_KEY",
    ]
    for k in keys:
        if k in os.environ:
            touched[k] = os.environ[k]
    # Apply values for this job
    os.environ["OPENAI_CHAT_MODEL"] = str(defaults.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini")
    os.environ["HF_LLM_NAME"] = str(defaults.get("HF_LLM_NAME") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    os.environ["EMBED_BACKEND"] = str(defaults.get("EMBED_BACKEND") or "hf")
    os.environ["LLM_BACKEND"] = str(defaults.get("LLM_BACKEND") or "openai")
    for k in ("MEMORY_ENABLED","ASK_EXHAUSTIVE"):
        os.environ[k] = "true" if bool(defaults.get(k)) else "false"
    for k in (
        "MEMORY_TOKEN_LIMIT","MEMORY_FILE_LIMIT_MB","OPENAI_TPM","OPENAI_RPM",
        "ASK_BATCH_CHAR_BUDGET","ASK_MAX_BATCHES","ASK_TIME_BUDGET_SEC","ASK_CANDIDATES",
    ):
        os.environ[k] = str(defaults.get(k))
    # Provider keys
    for k in ("OPENAI_API_KEY","HF_TOKEN","SERPAPI_KEY","BRAVE_API_KEY"):
        v = defaults.get(k)
        if v:
            os.environ[k] = str(v)
        elif k in os.environ:
            # If not set for this user, leave existing env (global) as is
            pass
    def restore():
        # restore previous env values for touched keys; unset newly set keys
        current = set(keys)
        for k, v in touched.items():
            os.environ[k] = v
        for k in (current - set(touched.keys())):
            try:
                del os.environ[k]
            except KeyError:
                pass
    return restore
