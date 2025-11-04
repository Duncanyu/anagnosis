from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.core.config import load_config, present_keys, save_secret
import httpx

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
        "ASK_CANDIDATES": 150,
        # New VM-friendly defaults
        "RERANK_TOP_N": 80,
        "ASK_MMR_CAP": 40,
        "ASK_MMR_TIMEOUT": 3,
        "SEARCH_TIMEOUT_SEC": 18,
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
        "SERPAPI_KEY": pick("SERPAPI_KEY", ""),
        "BRAVE_API_KEY": pick("BRAVE_API_KEY", ""),
        "WEB_SEARCH_PROVIDER": (secrets.get("WEB_SEARCH_PROVIDER") or env.get("WEB_SEARCH_PROVIDER") or "auto").lower(),
        "OPENAI_CHAT_MODEL": secrets.get("OPENAI_CHAT_MODEL") or env.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "EMBED_BACKEND": secrets.get("EMBED_BACKEND") or "hf",
        "LLM_BACKEND": "openai",
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
        "ASK_CANDIDATES": int(prefs.get("ASK_CANDIDATES") or env.get("ASK_CANDIDATES", "150")),
        # VM-friendly knobs (also respected if present)
        "RERANK_TOP_N": int(prefs.get("RERANK_TOP_N") or env.get("RERANK_TOP_N", "80")),
        "ASK_MMR_CAP": int(prefs.get("ASK_MMR_CAP") or env.get("ASK_MMR_CAP", "40")),
        "ASK_MMR_TIMEOUT": int(prefs.get("ASK_MMR_TIMEOUT") or env.get("ASK_MMR_TIMEOUT", "3")),
        "SEARCH_TIMEOUT_SEC": int(prefs.get("SEARCH_TIMEOUT_SEC") or env.get("SEARCH_TIMEOUT_SEC", "18")),
    }


def _verify_openai_key(key: str) -> Optional[bool]:
    if not key:
        return False
    try:
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5.0,
        )
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403):
            return False
        return None
    except Exception:
        return None


def _verify_hf_token(token: str) -> Optional[bool]:
    if not token:
        return False
    try:
        r = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403):
            return False
        return None
    except Exception:
        return None


def _verify_serpapi_key(key: str) -> Optional[bool]:
    if not key:
        return False
    try:
        r = httpx.get(
            "https://serpapi.com/account",
            params={"api_key": key},
            headers={"Accept": "application/json"},
            timeout=5.0,
        )
        if r.status_code != 200:
            return None if r.status_code >= 500 else False
        try:
            data = r.json()
        except Exception:
            return None
        return False if isinstance(data, dict) and data.get("error") else True
    except Exception:
        return None


def _verify_brave_key(key: str) -> Optional[bool]:
    if not key:
        return False
    try:
        r = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": "anagnosis", "count": 1},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=5.0,
        )
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403):
            return False
        return None
    except Exception:
        return None


def user_key_status(user_id: str, verify: bool = False, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s = read_user_secrets(user_id)
    if not s:
        keys = present_keys()
        s = {
            "OPENAI_API_KEY": keys.get("OPENAI_API_KEY") and "present" or "",
            "HF_TOKEN": keys.get("HF_TOKEN") and "present" or "",
            "SERPAPI_KEY": keys.get("SERPAPI_KEY") and "present" or "",
            "BRAVE_API_KEY": keys.get("BRAVE_API_KEY") and "present" or "",
        }

    if defaults is None:
        defaults = _settings_defaults_for_user(user_id)
    embed_backend = (defaults.get("EMBED_BACKEND") or "hf").lower()
    llm_backend = (defaults.get("LLM_BACKEND") or "openai").lower()
    web_provider = (defaults.get("WEB_SEARCH_PROVIDER") or "auto").lower()

    required = {
        "openai": (llm_backend == "openai") or (embed_backend == "openai"),
        "serpapi": (web_provider == "serpapi"),
        "brave": (web_provider == "brave"),
    }

    present = {
        "openai": bool(s.get("OPENAI_API_KEY")),
        "serpapi": bool(s.get("SERPAPI_KEY")),
        "brave": bool(s.get("BRAVE_API_KEY")),
    }

    if not verify:
        return {k: {"present": present[k], "ok": True if present[k] else False, "required": required[k]} for k in present}

    ok = {
        "openai": _verify_openai_key(str(s.get("OPENAI_API_KEY") or "")),
        "serpapi": _verify_serpapi_key(str(s.get("SERPAPI_KEY") or "")),
        "brave": _verify_brave_key(str(s.get("BRAVE_API_KEY") or "")),
    }
    return {k: {"present": present[k], "ok": ok[k], "required": required[k]} for k in present}


def _save_settings_for_user(user_id: str, payload: Dict[str, Any]) -> str:
    try:
        secrets = read_user_secrets(user_id)
        if payload.get("openai_key", "").strip():
            secrets["OPENAI_API_KEY"] = payload["openai_key"].strip()
        if payload.get("serp_key", "").strip():
            secrets["SERPAPI_KEY"] = payload["serp_key"].strip()
        if payload.get("brave_key", "").strip():
            secrets["BRAVE_API_KEY"] = payload["brave_key"].strip()
        secrets["OPENAI_CHAT_MODEL"] = (payload.get("openai_model", "gpt-4o-mini").strip() or "gpt-4o-mini")
        secrets["EMBED_BACKEND"] = (payload.get("embed_backend") or "hf").lower()
        secrets["LLM_BACKEND"] = "openai"
        if (payload.get("web_provider") or "").strip():
            secrets["WEB_SEARCH_PROVIDER"] = (payload.get("web_provider") or "auto").strip().lower()
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
                "ASK_CANDIDATES": int(payload.get("ask_candidates", 150)),
                # Optional advanced fields (ignored if not sent)
                "RERANK_TOP_N": int(payload.get("rerank_top_n", 80)),
                "ASK_MMR_CAP": int(payload.get("ask_mmr_cap", 40)),
                "ASK_MMR_TIMEOUT": int(payload.get("ask_mmr_timeout", 3)),
                "SEARCH_TIMEOUT_SEC": int(payload.get("search_timeout_sec", 18)),
            }
        )
        write_user_prefs(user_id, prefs)
        return "Settings saved."
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/settings")
async def get_settings(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    defaults = _settings_defaults_for_user(str(user.id))
    verify = False
    try:
        q = request.query_params
        verify = (q.get("verify", "0") in {"1", "true", "yes", "on"})
    except Exception:
        verify = False
    status = user_key_status(str(user.id), verify=verify, defaults=defaults)
    return JSONResponse({"defaults": defaults, "keys": status})


@router.post("/settings")
async def post_settings(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    payload = await request.json()
    message = _save_settings_for_user(str(user.id), payload)
    status = _settings_defaults_for_user(str(user.id))
    key_status = user_key_status(str(user.id))
    return JSONResponse({"message": message, "defaults": status, "keys": key_status})

def apply_env_for_user(user_id: str):
    defaults = _settings_defaults_for_user(user_id)
    touched = {}
    keys = [
        "OPENAI_CHAT_MODEL","EMBED_BACKEND","LLM_BACKEND",
        "MEMORY_ENABLED","MEMORY_TOKEN_LIMIT","MEMORY_FILE_LIMIT_MB",
        "OPENAI_TPM","OPENAI_RPM","ASK_BATCH_CHAR_BUDGET","ASK_MAX_BATCHES",
        "ASK_TIME_BUDGET_SEC","ASK_EXHAUSTIVE","ASK_RERANKER","ASK_CANDIDATES","WEB_SEARCH_PROVIDER",
        "RERANK_TOP_N","ASK_MMR_CAP","ASK_MMR_TIMEOUT","SEARCH_TIMEOUT_SEC",
        "OPENAI_API_KEY","SERPAPI_KEY","BRAVE_API_KEY",
    ]
    for k in keys:
        if k in os.environ:
            touched[k] = os.environ[k]
    os.environ["OPENAI_CHAT_MODEL"] = str(defaults.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini")
    # Fixed HF model selection for fallback
    os.environ["HF_LLM_NAME"] = str(os.environ.get("HF_LLM_NAME") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    os.environ["EMBED_BACKEND"] = str(defaults.get("EMBED_BACKEND") or "hf")
    os.environ["LLM_BACKEND"] = str(defaults.get("LLM_BACKEND") or "openai")
    for k in ("MEMORY_ENABLED","ASK_EXHAUSTIVE"):
        os.environ[k] = "true" if bool(defaults.get(k)) else "false"
    for k in (
        "MEMORY_TOKEN_LIMIT","MEMORY_FILE_LIMIT_MB","OPENAI_TPM","OPENAI_RPM",
        "ASK_BATCH_CHAR_BUDGET","ASK_MAX_BATCHES","ASK_TIME_BUDGET_SEC","ASK_CANDIDATES",
        "RERANK_TOP_N","ASK_MMR_CAP","ASK_MMR_TIMEOUT","SEARCH_TIMEOUT_SEC",
    ):
        os.environ[k] = str(defaults.get(k))
    for k in ("OPENAI_API_KEY","SERPAPI_KEY","BRAVE_API_KEY"):
        v = defaults.get(k)
        if v:
            os.environ[k] = str(v)
        elif k in os.environ:
            pass
    os.environ["WEB_SEARCH_PROVIDER"] = str(defaults.get("WEB_SEARCH_PROVIDER") or "auto")
    def restore():
        current = set(keys)
        for k, v in touched.items():
            os.environ[k] = v
        for k in (current - set(touched.keys())):
            try:
                del os.environ[k]
            except KeyError:
                pass
    return restore
