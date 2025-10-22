from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Sequence
import json
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.services import index as index_service
from api.services import memory as mem
from api.services import summaries_store
import pathlib
import fitz
from fastapi.responses import Response

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


def _extract_doc_names(summary_text: str) -> List[str]:
    names: List[str] = []
    if not summary_text:
        return names
    try:
        for m in re.finditer(r"^##\s+(.+)$", summary_text, re.MULTILINE):
            nm = (m.group(1) or "").strip()
            if nm:
                names.append(_normalize_doc_name(nm))
    except Exception:
        pass
    return names


def _doc_ingest_times() -> Dict[str, int]:
    """Return latest known ingest timestamp per doc from summaries log."""
    times: Dict[str, int] = {}
    path = pathlib.Path("artifacts") / "doc_summaries.jsonl"
    if not path.exists():
        return times
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = int(rec.get("ts") or 0)
                summary = rec.get("summary") or ""
                for raw in _extract_doc_names(summary):
                    name = _normalize_doc_name(raw)
                    prev = int(times.get(name) or 0)
                    if ts > prev:
                        times[name] = ts
    except Exception:
        return times
    return times


@router.get("/library")
async def api_library(user: User = Depends(require_auth)) -> JSONResponse:
    # Show documents across all embedding namespaces so users always see their
    # ingested files regardless of current backend selection
    rows = index_service.list_chunks_all_namespaces(user_id=str(user.id))
    documents = _collect_user_docs(rows)
    # Attach latest known ingest timestamps for sorting by "Date added"
    times = _doc_ingest_times()
    for d in documents:
        try:
            d["ingested_at"] = int(times.get(d.get("name") or "") or 0) or None
        except Exception:
            d["ingested_at"] = None
    return JSONResponse({"documents": documents})


@router.get("/library/doc/{doc_name}")
async def api_library_doc_details(doc_name: str, user: User = Depends(require_auth)) -> JSONResponse:
    name = _normalize_doc_name(doc_name)
    # Look across all namespaces so details work regardless of current backend
    rows = index_service.list_chunks_all_namespaces(user_id=str(user.id))
    rows = [r for r in rows if _normalize_doc_name(r.get("doc_name") or "") == name]
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found.")
    pages = set()
    headings = {}
    math_count = 0
    eq_count = 0
    formula_count = 0
    for r in rows:
        for k in ("page", "page_start", "page_number"):
            if r.get(k) is not None:
                try:
                    pages.add(int(r.get(k)))
                except Exception:
                    pass
                break
        hp = (r.get("heading_path") or "").strip()
        if hp:
            headings[hp] = headings.get(hp, 0) + 1
        if r.get("has_math"):
            math_count += 1
        if r.get("is_equation"):
            eq_count += 1
        if r.get("is_formula"):
            formula_count += 1
    # Top headings by frequency
    top_headings = sorted(headings.items(), key=lambda kv: -kv[1])[:8]
    samples = []
    for r in rows[:6]:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        samples.append({
            "page": r.get("page") or r.get("page_start") or None,
            "snippet": txt[:400],
        })
    info = {
        "name": name,
        "display_name": pathlib.Path(name).stem,
        "pages": sorted(pages),
        "page_count": len(pages) or None,
        "chunk_count": len(rows),
        "math_chunks": math_count,
        "equations": eq_count,
        "formula_chunks": formula_count,
        "top_headings": [{"heading": h, "count": c} for (h, c) in top_headings],
        "samples": samples,
    }
    return JSONResponse(info)


@router.get("/library/page_image")
async def api_page_image(doc: str, page: int, needle: str | None = None, user: User = Depends(require_auth)):
    """Render a single PDF page as PNG for inline viewing.

    Looks under artifacts/docs/<user_id>/<doc>.
    """
    try:
        safe = pathlib.Path(str(doc)).name
        pdf_path = pathlib.Path("artifacts") / "docs" / str(user.id) / safe
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Source PDF not available.")
        if pdf_path.suffix.lower() != '.pdf':
            raise HTTPException(status_code=415, detail="Preview available for PDFs only.")
        pnum = max(1, int(page))
        with fitz.open(pdf_path) as d:
            if pnum < 1 or pnum > len(d):
                raise HTTPException(status_code=400, detail="Invalid page number.")
            pg = d.load_page(pnum - 1)
            pix = pg.get_pixmap(dpi=180, alpha=False)
            img_bytes = pix.tobytes("png")
            # Highlights disabled: always return raw rendered page
        return Response(content=img_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
    removed = index_service.remove_documents_for_user([doc_name], user_id=str(user.id))
    if removed <= 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        mem.prune_by_doc_names([doc_name], user_id=str(user.id))
    except Exception:
        pass
    try:
        summaries_store.prune_by_doc_names([doc_name])
    except Exception:
        pass
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
    removed = index_service.remove_documents_for_user([str(n) for n in names], user_id=str(user.id))
    if removed <= 0:
        raise HTTPException(status_code=404, detail="No documents removed")
    try:
        mem.prune_by_doc_names(names, user_id=str(user.id))
    except Exception:
        pass
    try:
        summaries_store.prune_by_doc_names(names)
    except Exception:
        pass
    return JSONResponse({"removed": removed})


@router.post("/library/clear")
async def api_library_clear(user: User = Depends(require_auth)) -> JSONResponse:
    removed = index_service.clear_documents_for_user(user_id=str(user.id))
    try:
        mem.clear(user_id=str(user.id))
    except Exception:
        pass
    return JSONResponse({"removed": removed})
