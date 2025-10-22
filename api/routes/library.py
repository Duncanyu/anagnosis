from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Sequence

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


@router.get("/library")
async def api_library(user: User = Depends(require_auth)) -> JSONResponse:
    # Show documents across all embedding namespaces so users always see their
    # ingested files regardless of current backend selection
    rows = index_service.list_chunks_all_namespaces(user_id=str(user.id))
    documents = _collect_user_docs(rows)
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
        pnum = max(1, int(page))
        with fitz.open(pdf_path) as d:
            if pnum < 1 or pnum > len(d):
                raise HTTPException(status_code=400, detail="Invalid page number.")
            pg = d.load_page(pnum - 1)
            pix = pg.get_pixmap(dpi=180, alpha=False)
            img_bytes = pix.tobytes("png")
            # Optional highlight overlay using needle
            if needle:
                try:
                    import io
                    from PIL import Image, ImageDraw
                    rects = []
                    # Try multiple windows of the needle to improve match reliability
                    text = (needle or "").strip()
                    windows = [text]
                    if len(text) > 60:
                        step = max(10, len(text)//4)
                        for i in range(0, len(text)-30, step):
                            windows.append(text[i:i+30])
                    quads = None
                    for w in windows:
                        w = " ".join(w.split())
                        try:
                            qs = pg.search_for(w, hit_max=32, flags=fitz.TEXT_IGNORECASE)
                        except Exception:
                            qs = pg.search_for(w)
                        if qs:
                            quads = qs; break
                    for r in quads or []:
                        # r is Rect; scale to current DPI
                        rects.append(r)
                    if rects:
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                        draw = ImageDraw.Draw(img, 'RGBA')
                        # Fit rect coordinates: PyMuPDF coordinates are in points at base matrix. We used dpi=180 pixmap -> scale factor = dpi/72
                        scale = 180/72.0
                        for r in rects:
                            x0,y0,x1,y1 = r.x0*scale, r.y0*scale, r.x1*scale, r.y1*scale
                            pad = 2
                            draw.rectangle([x0-pad,y0-pad,x1+pad,y1+pad], outline=(88,101,242,255), width=3, fill=(88,101,242,64))
                        out = io.BytesIO()
                        img.save(out, format='PNG')
                        img_bytes = out.getvalue()
                except Exception:
                    pass
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
