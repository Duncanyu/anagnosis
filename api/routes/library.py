from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.services import index as index_service

router = APIRouter(prefix="/api", tags=["library"])


def _normalize_doc_name(name: str) -> str:
    try:
        return pathlib.Path(str(name)).name
    except Exception:
        return str(name or "").strip()


def _collect_user_docs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = _normalize_doc_name(row.get("doc_name") or "Unknown.pdf")
        entry = stats.setdefault(
            name,
            {
                "name": name,
                "display_name": pathlib.Path(name).stem,
                "chunks": 0,
                "_pages": set(),
            },
        )
        entry["chunks"] += 1
        for key_page in ("page", "page_start", "page_number"):
            if row.get(key_page) is not None:
                try:
                    entry["_pages"].add(int(row[key_page]))
                except Exception:
                    pass
                break
    out: List[Dict[str, Any]] = []
    for info in stats.values():
        info["pages"] = len(info.pop("_pages", set())) or None
        out.append(info)
    out.sort(key=lambda r: r.get("display_name", ""))
    return out


@router.get("/library")
async def api_library(user: User = Depends(require_auth)) -> JSONResponse:
    rows = index_service.list_chunks(user_id=str(user.id))
    documents = _collect_user_docs(rows)
    return JSONResponse({"documents": documents})


def _filter_keep_rows_for_user(rem_names: Sequence[str], user_id: str) -> List[Dict[str, Any]]:
    names = { _normalize_doc_name(n) for n in (rem_names or []) if n }
    all_rows = index_service.list_chunks(user_id=None)
    keep: List[Dict[str, Any]] = []
    for r in all_rows:
        uid = str(r.get("user_id") or "")
        doc = _normalize_doc_name(r.get("doc_name") or "")
        if uid == str(user_id) and (not doc or doc in names):
            # drop this row
            continue
        keep.append(r)
    return keep


@router.delete("/library/{doc_name}")
async def api_library_delete(doc_name: str, user: User = Depends(require_auth)) -> JSONResponse:
    keep_rows = _filter_keep_rows_for_user([doc_name], user_id=str(user.id))
    before = len(index_service.list_chunks(user_id=None))
    index_service.rebuild_index_from_rows(keep_rows)
    after = len(index_service.list_chunks(user_id=None))
    removed = max(0, before - after)
    if removed <= 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return JSONResponse({"removed": removed})


@router.post("/library/delete")
async def api_library_delete_batch(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    names = payload.get("names") or []
    if not isinstance(names, list) or not names:
        raise HTTPException(status_code=400, detail="names list required")
    keep_rows = _filter_keep_rows_for_user([str(n) for n in names], user_id=str(user.id))
    before = len(index_service.list_chunks(user_id=None))
    index_service.rebuild_index_from_rows(keep_rows)
    after = len(index_service.list_chunks(user_id=None))
    removed = max(0, before - after)
    if removed <= 0:
        raise HTTPException(status_code=404, detail="No documents removed")
    return JSONResponse({"removed": removed})


@router.post("/library/clear")
async def api_library_clear(user: User = Depends(require_auth)) -> JSONResponse:
    all_rows = index_service.list_chunks(user_id=None)
    keep_rows = [r for r in all_rows if str(r.get("user_id") or "") != str(user.id)]
    before = len(all_rows)
    index_service.rebuild_index_from_rows(keep_rows)
    after = len(index_service.list_chunks(user_id=None))
    removed = max(0, before - after)
    return JSONResponse({"removed": removed})

