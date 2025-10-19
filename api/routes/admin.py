from __future__ import annotations

import shutil
import pathlib
from typing import Dict
import os, json
from api.core.config import CONFIG_PATH, WANTED_KEYS, SERVICE_NAME
try:
    import keyring  # optional
except Exception:  # pragma: no cover
    keyring = None

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.auth.middleware import require_dev
from api.db.models import User


router = APIRouter(prefix="/api/admin", tags=["admin"])


ART = pathlib.Path("artifacts")


def _clear_settings_impl() -> int:
    users_dir = ART / "users"
    removed = 0
    if users_dir.exists():
        for sub in users_dir.iterdir():
            if not sub.is_dir():
                continue
            for name in ("prefs.json", "secrets.json"):
                p = sub / name
                if p.exists():
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        pass
    # Remove legacy global prefs if present
    legacy = ART / "ui_prefs.json"
    if legacy.exists():
        try:
            legacy.unlink(); removed += 1
        except Exception:
            pass
    # Also clear global config.local.json keys
    try:
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            changed = False
            for k in WANTED_KEYS:
                if cfg.pop(k, None) is not None:
                    changed = True; removed += 1
            if changed:
                CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass
    # Clear keyring entries if available
    if keyring is not None:
        for k in WANTED_KEYS:
            try:
                keyring.delete_password(SERVICE_NAME, k); removed += 1
            except Exception:
                pass
    # Unset environment variables for this process
    for k in WANTED_KEYS:
        try:
            if k in os.environ:
                os.environ.pop(k, None); removed += 1
        except Exception:
            pass
    return removed


@router.post("/clear_settings")
def clear_settings(user: User = Depends(require_dev)) -> JSONResponse:
    """Remove all per-user settings (prefs + secrets) files under artifacts/users.

    This is a dev convenience endpoint; it does not delete documents or indexes.
    """
    removed = _clear_settings_impl()
    return JSONResponse({"ok": True, "removed_files": removed})


def _clear_memory_impl() -> int:
    """Remove all per-user memory files (memory_*.jsonl) and legacy memory.jsonl.

    This does not touch any document chunks or indexes.
    """
    removed = 0
    # Legacy
    legacy = ART / "memory.jsonl"
    if legacy.exists():
        try:
            legacy.unlink()
            removed += 1
        except Exception:
            pass
    # Per-user memory files
    for p in ART.glob("memory_*.jsonl"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed


@router.post("/clear_memory")
def clear_memory(user: User = Depends(require_dev)) -> JSONResponse:
    removed = _clear_memory_impl()
    return JSONResponse({"ok": True, "removed_files": removed})


@router.post("/clear_all")
def clear_all(user: User = Depends(require_dev)) -> JSONResponse:
    """Clear all per-user settings and memory.

    Convenience endpoint equivalent to calling both clear_settings and clear_memory.
    """
    removed_settings = _clear_settings_impl()
    removed_memory = _clear_memory_impl()
    return JSONResponse({"ok": True, "removed_settings": removed_settings, "removed_memory": removed_memory})
