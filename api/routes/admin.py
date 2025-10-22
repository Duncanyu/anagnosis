from __future__ import annotations

import shutil
import pathlib
from typing import Dict
import os, json
from api.core.config import CONFIG_PATH, WANTED_KEYS, SERVICE_NAME
from api.services import index as index_service
from api.services import usage as usage_service
from api.services import memory as mem
from api.services import summaries_store
from api.db.database import get_db
import pathlib
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
from api.db.database import get_db
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


@router.post("/rebuild_index")
def rebuild_index(user: User = Depends(require_dev)) -> JSONResponse:
    """Rebuild the FAISS index from the current chunks file.

    Useful when index artifacts are suspected to be out of sync. Does not change
    any underlying chunk rows; simply rewrites the index and normalizes rids.
    """
    rows = index_service.list_chunks(user_id=None)
    before = len(rows)
    index_service.rebuild_index_from_rows(rows)
    after = len(index_service.list_chunks(user_id=None))
    return JSONResponse({"ok": True, "rows_before": before, "rows_after": after})


@router.post("/purge_all_libraries")
def purge_all_libraries(user: User = Depends(require_dev)) -> JSONResponse:
    """Delete all document chunks for all users across all namespaces.

    Leaves user settings and memory intact.
    """
    before = len(index_service.list_chunks(user_id=None))
    removed = index_service.clear_all_documents()
    after = len(index_service.list_chunks(user_id=None))
    return JSONResponse({"ok": True, "rows_before": before, "rows_removed": removed, "rows_after": after})


@router.get("/status")
def status(_: User = Depends(require_dev), db: Session = Depends(get_db)) -> JSONResponse:
    """Return system status: namespaces, row counts, users, and artifact stats."""
    # Namespaces summary
    namespaces = []
    total_rows = 0
    for idx_path, meta_path, ch_path in index_service._namespace_files():  # type: ignore[attr-defined]
        count = 0
        try:
            if meta_path.exists():
                m = json.loads(meta_path.read_text(encoding='utf-8'))
                count = int(m.get('count') or 0)
            elif ch_path.exists():
                with ch_path.open('r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
        except Exception:
            count = 0
        namespaces.append({
            "chunks": str(ch_path),
            "index": str(idx_path),
            "meta": str(meta_path),
            "rows": count,
        })
        total_rows += count
    # User count
    from api.db.models import User as DBUser
    users = db.query(DBUser).count()
    # Memory files
    mem_files = 0
    try:
        for p in (ART := pathlib.Path('artifacts')).glob('memory_*.jsonl'):
            if p.is_file():
                mem_files += 1
    except Exception:
        mem_files = 0
    return JSONResponse({
        "namespaces": namespaces,
        "total_rows": total_rows,
        "users": users,
        "memory_files": mem_files,
    })




@router.get("/users")
def list_users(_: User = Depends(require_dev), db: Session = Depends(get_db)) -> JSONResponse:
    """List users (id and email) for administrative diagnostics."""
    from api.db.models import User as DBUser
    users = db.query(DBUser).all()
    data = [{"id": str(u.id), "email": u.email, "created_at": getattr(u, "created_at", None)} for u in users]
    return JSONResponse({"users": data})


@router.get("/usage")
def usage(_: User = Depends(require_dev), db: Session = Depends(get_db)) -> JSONResponse:
    """Return per-user usage counters (ask/ingest), joined with emails."""
    from api.db.models import User as DBUser
    users = {str(u.id): u.email for u in db.query(DBUser).all()}
    rows = usage_service.all_users_usage()
    for r in rows:
        r["email"] = users.get(str(r.get("user_id")))
    return JSONResponse({"usage": rows})


def _normalize_doc_name(name: str) -> str:
    try:
        return pathlib.Path(str(name)).name
    except Exception:
        return str(name or "").strip()


@router.get("/scan_doc")
def scan_doc(name: str, user: User = Depends(require_dev)) -> JSONResponse:
    """Scan all namespaces, summaries, and memory for a doc name for current user.

    Returns counts per namespace file and hits in summaries/memory.
    """
    doc = _normalize_doc_name(name)
    uid = str(user.id)
    by_ns = []
    total_rows = 0
    for idx_path, meta_path, ch_path in index_service._namespace_files():  # type: ignore[attr-defined]
        rows = []
        try:
            if ch_path.exists():
                with ch_path.open('r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            rows = []
        hits = [r for r in rows if str(r.get('user_id') or '') == uid and _normalize_doc_name(r.get('doc_name') or '') == doc]
        total_rows += len(hits)
        by_ns.append({
            "chunks_file": str(ch_path),
            "index_file": str(idx_path),
            "meta_file": str(meta_path),
            "matches": len(hits),
        })
    # Memory + summaries hits (best-effort substring check)
    mem_hits = 0
    try:
        p = mem._mem_path(user_id=uid)  # type: ignore[attr-defined]
        if p.exists():
            low = p.read_text(encoding='utf-8').lower()
            if doc.lower() in low:
                mem_hits = low.count(doc.lower())
    except Exception:
        pass
    sum_hits = 0
    try:
        p2 = summaries_store.PATH  # type: ignore[attr-defined]
        if p2.exists():
            low = p2.read_text(encoding='utf-8').lower()
            if doc.lower() in low:
                sum_hits = low.count(doc.lower())
    except Exception:
        pass
    return JSONResponse({
        "doc": doc,
        "user_id": uid,
        "namespaces": by_ns,
        "total_rows": total_rows,
        "memory_hits": mem_hits,
        "summary_hits": sum_hits,
    })


@router.post("/purge_doc")
def purge_doc(name: str, user: User = Depends(require_dev)) -> JSONResponse:
    """Force purge a document for the current user across all namespaces + memory + summaries."""
    doc = _normalize_doc_name(name)
    uid = str(user.id)
    removed = index_service.remove_documents_for_user([doc], user_id=uid)
    try:
        mem_removed = mem.prune_by_doc_names([doc], user_id=uid)
    except Exception:
        mem_removed = 0
    try:
        sum_removed = summaries_store.prune_by_doc_names([doc])
    except Exception:
        sum_removed = 0
    return JSONResponse({
        "doc": doc,
        "rows_removed": removed,
        "memory_lines_removed": mem_removed,
        "summary_lines_removed": sum_removed,
    })
