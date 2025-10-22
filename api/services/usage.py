from __future__ import annotations

import json
import pathlib
import time
from typing import Dict, Any, List

ART = pathlib.Path("artifacts")


def _user_usage_path(user_id: str) -> pathlib.Path:
    d = ART / "users" / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / "usage.json"


def record(user_id: str, action: str) -> None:
    """Record a usage event for a user. Actions: 'ask' | 'ingest'."""
    p = _user_usage_path(user_id)
    now = int(time.time())
    data: Dict[str, Any]
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = {}
    except Exception:
        data = {}
    if action == "ask":
        data["ask_count"] = int(data.get("ask_count", 0)) + 1
        data["last_ask_at"] = now
    elif action == "ingest":
        data["ingest_count"] = int(data.get("ingest_count", 0)) + 1
        data["last_ingest_at"] = now
    else:
        # generic counter
        key = f"{action}_count"
        data[key] = int(data.get(key, 0)) + 1
    try:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def get(user_id: str) -> Dict[str, Any]:
    p = _user_usage_path(user_id)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"ask_count": 0, "ingest_count": 0}


def all_users_usage() -> List[Dict[str, Any]]:
    users_dir = ART / "users"
    out: List[Dict[str, Any]] = []
    if not users_dir.exists():
        return out
    for sub in users_dir.iterdir():
        if not sub.is_dir():
            continue
        uid = sub.name
        p = sub / "usage.json"
        data = {"ask_count": 0, "ingest_count": 0}
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        out.append({"user_id": uid, **data})
    return out

