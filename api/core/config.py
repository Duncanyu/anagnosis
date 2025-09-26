import os, json, pathlib
from typing import Optional, Dict
try:
    import keyring 
except ImportError:
    keyring = None

APP_NAME = "reading-agent"
CONFIG_PATH = pathlib.Path.cwd() / "config.local.json"
SERVICE_NAME = "reading-agent"

WANTED_KEYS = [
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "COHERE_API_KEY",
    "EMBED_BACKEND",
    "LLM_BACKEND",
    "RERANK_BACKEND",
    "QDRANT_URL",
    "SERPAPI_KEY",
    "BRAVE_API_KEY",
]

def _load_json_config() -> Dict[str, str]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _get_keyring(name: str) -> Optional[str]:
    if keyring is None:
        return None
    try:
        return keyring.get_password(SERVICE_NAME, name)
    except Exception:
        return None

def _set_keyring(name: str, value: str) -> bool:
    if keyring is None:
        return False
    try:
        keyring.set_password(SERVICE_NAME, name, value)
        return True
    except Exception:
        return False

def load_config() -> Dict[str, Optional[str]]:
    cfg = {k: None for k in WANTED_KEYS}

    for k in cfg.keys():
        if os.getenv(k):
            cfg[k] = os.getenv(k)

    for k in cfg.keys():
        if not cfg[k]:
            val = _get_keyring(k)
            if val:
                cfg[k] = val

    local = _load_json_config()
    for k in cfg.keys():
        if not cfg[k] and k in local:
            cfg[k] = local[k]

    return cfg

def save_secret(name: str, value: str, prefer="keyring") -> None:
    if prefer == "keyring" and _set_keyring(name, value):
        return
    cfg = _load_json_config()
    cfg[name] = value
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def present_keys() -> Dict[str, bool]:
    cfg = load_config()
    return {k: bool(cfg.get(k)) for k in WANTED_KEYS}
