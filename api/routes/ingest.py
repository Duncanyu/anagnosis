from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import threading
import uuid
from typing import Any, Dict, List, Sequence
import time

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.services import pipeline
from api.services.pipeline import PipelineCancelled
from api.routes.settings import apply_env_for_user
from api.core.rate_limit import rate_limiter
from api.services import usage as usage_service

try:
    import markdown as mdlib
except Exception:
    mdlib = None

router = APIRouter(prefix="/api", tags=["ingest"])


def _md_to_html(md_text: str) -> str:
    if mdlib:
        return mdlib.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    return f"<pre>{(md_text or '').replace('<','&lt;').replace('>','&gt;')}</pre>"


def _format_ingest_details(details: Sequence[Dict[str, Any]]) -> str:
    if not details:
        return "No documents ingested yet."
    lines = [
        "| File | Pages | OCR Pages | Chunks | Suspect Pages |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in details:
        name = pathlib.Path(row.get("file", "")).name or "?"
        num_pages = row.get("num_pages", "?")
        num_chunks = row.get("num_chunks", "?")
        ocr_pages = row.get("ocr_page_numbers") or []
        suspect_pages = row.get("suspect_pages") or []
        ocr_preview = ", ".join(str(x) for x in ocr_pages[:6]) or "—"
        if len(ocr_pages) > 6:
            ocr_preview += " …"
        suspect_preview = ", ".join(str(x) for x in suspect_pages[:6]) or "—"
        if len(suspect_pages) > 6:
            suspect_preview += " …"
        lines.append(
            f"| {name} | {num_pages} | {ocr_preview} | {num_chunks} | {suspect_preview} |"
        )
    return "\n".join(lines)


INGEST_JOBS: Dict[str, Dict[str, Any]] = {}
INGEST_LOCK = threading.Lock()


def _append_log(job_id: str, message: str) -> None:
    with INGEST_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is not None:
            job.setdefault("logs", []).append(message)


def _start_ingest_job(
    paths: Sequence[pathlib.Path],
    temp_dir: pathlib.Path,
    file_names: Sequence[str],
    user_id: str,
) -> str:
    job_id = uuid.uuid4().hex
    job_info = {
        "status": "running",
        "logs": ["Files queued:"] + [f"- {n}" for n in file_names] + ["Starting ingestion..."],
        "progress": 0,
        "documents": [],
        "summary_html": "",
        "details_html": "",
        "error": None,
        "user_id": str(user_id),
    }
    with INGEST_LOCK:
        INGEST_JOBS[job_id] = job_info

    doc_count = max(1, len(paths))
    doc_span = 100.0 / doc_count
    name_lookup: Dict[str, int] = {}
    for idx, name in enumerate(file_names):
        name_lookup.setdefault(name, idx)

    doc_state = {
        "index": 0,
        "name": file_names[0] if file_names else "Document",
        "pages_total": 0,
        "pages_done": 0,
    }

    def update_overall(fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        base = doc_span * doc_state["index"]
        overall = min(100.0, base + doc_span * fraction)
        with INGEST_LOCK:
            job = INGEST_JOBS.get(job_id)
            if job is not None:
                job["progress"] = max(job.get("progress", 0), int(overall))

    def worker() -> None:
        try:
            restore_env = apply_env_for_user(str(user_id))
            # Validate keys for selected backends before doing any work
            be_llm = (os.environ.get("LLM_BACKEND", "openai") or "").lower()
            be_embed = (os.environ.get("EMBED_BACKEND", "hf") or "").lower()
            need_openai = (be_llm == "openai") or (be_embed == "openai")
            if need_openai and not os.environ.get("OPENAI_API_KEY"):
                _append_log(job_id, "Configuration error: Missing OpenAI API key")
                with INGEST_LOCK:
                    job = INGEST_JOBS.get(job_id)
                    if job is not None:
                        job["status"] = "error"
                        job["error"] = "Missing OpenAI API key"
                return
            # Hard timeout (Phase 4): default 10 minutes
            try:
                tout = float(os.environ.get("INGEST_TIMEOUT_SEC", "600"))
            except Exception:
                tout = 600.0
            deadline = time.time() + max(30.0, tout)

            def should_cancel() -> bool:
                return time.time() >= deadline
            def log(msg: str) -> None:
                nonlocal doc_state
                if msg.startswith("Loading "):
                    name = msg.split("Loading ", 1)[1].strip("…")
                    idx = name_lookup.get(name)
                    if idx is None:
                        idx = min(doc_state["index"] + 1, doc_count - 1)
                        name_lookup[name] = idx
                    doc_state = {
                        "index": idx,
                        "name": name,
                        "pages_total": 0,
                        "pages_done": 0,
                    }
                    with INGEST_LOCK:
                        job = INGEST_JOBS.get(job_id)
                        if job is not None:
                            docs = job.setdefault("documents", [])
                            while len(docs) <= idx:
                                docs.append(
                                    {
                                        "name": name,
                                        "pages": 0,
                                        "pages_total": 0,
                                        "pages_done": 0,
                                    }
                                )
                            docs[idx]["name"] = name
                            docs[idx]["pages"] = 0
                            docs[idx]["pages_total"] = 0
                            docs[idx]["pages_done"] = 0
                    update_overall(0.0)
                    _append_log(job_id, msg)
                    return
                if msg.startswith("Pages: "):
                    try:
                        doc_state["pages_total"] = int(msg.split("Pages:", 1)[1].strip())
                        with INGEST_LOCK:
                            job = INGEST_JOBS.get(job_id)
                            if job is not None:
                                docs = job.setdefault("documents", [])
                                while len(docs) <= doc_state["index"]:
                                    docs.append(
                                        {
                                            "name": doc_state["name"],
                                            "pages": 0,
                                            "pages_total": 0,
                                            "pages_done": 0,
                                        }
                                    )
                                docs[doc_state["index"]]["pages_total"] = doc_state["pages_total"]
                    except Exception:
                        pass
                elif msg.startswith("p "):
                    tail = msg[2:]
                    if "/" in tail:
                        left, right = tail.split("/", 1)
                        try:
                            done = int(left.split()[0])
                            total = int(right.split()[0])
                            doc_state["pages_total"] = max(
                                doc_state.get("pages_total", total), total
                            )
                            doc_state["pages_done"] = done
                            if doc_state["pages_total"] > 0:
                                ratio = doc_state["pages_done"] / doc_state["pages_total"]
                                update_overall(0.5 * ratio)
                            with INGEST_LOCK:
                                job = INGEST_JOBS.get(job_id)
                                if job is not None:
                                    docs = job.setdefault("documents", [])
                                    while len(docs) <= doc_state["index"]:
                                        docs.append(
                                            {
                                                "name": doc_state["name"],
                                                "pages": 0,
                                                "pages_total": 0,
                                                "pages_done": 0,
                                            }
                                        )
                                    docs[doc_state["index"]]["name"] = doc_state["name"]
                                    docs[doc_state["index"]]["pages"] = doc_state["pages_done"]
                                    docs[doc_state["index"]]["pages_done"] = doc_state["pages_done"]
                                    docs[doc_state["index"]]["pages_total"] = doc_state["pages_total"]
                        except Exception:
                            pass
                elif msg.startswith("Parsing done"):
                    update_overall(0.5)
                _append_log(job_id, msg)

            def update_pct(pct: int) -> None:
                if pct <= 15:
                    update_overall(0.5)
                    return
                fraction = 0.5 + 0.5 * max(0.0, min(1.0, (pct - 15) / 85))
                update_overall(fraction)

            result = pipeline.ingest_documents(
                paths,
                progress=log,
                progress_pct=update_pct,
                should_cancel=should_cancel,
                user_id=str(job_info["user_id"]),
            )
            summary_md = result.get("doc_summary") or "No documents ingested."
            details_md = _format_ingest_details(result.get("details", []))
            with INGEST_LOCK:
                job = INGEST_JOBS.get(job_id)
                if job is not None:
                    job["summary_html"] = _md_to_html(summary_md)
                    job["details_html"] = _md_to_html(details_md)
                    job["status"] = "done"
                    job["progress"] = 100
                    job.setdefault("logs", []).append("Ingestion complete.")
        except PipelineCancelled:
            with INGEST_LOCK:
                job = INGEST_JOBS.get(job_id)
                if job is not None:
                    job["status"] = "cancelled"
                    job["error"] = "Ingestion timed out"
                    job.setdefault("logs", []).append("Ingestion timed out.")
        except Exception as exc:
            with INGEST_LOCK:
                job = INGEST_JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    job["progress"] = job.get("progress", 0)
                    job.setdefault("logs", []).append(f"Error: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            try:
                restore_env()
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _ingest_job_payload(job_id: str, user_id: str) -> Dict[str, Any]:
    with INGEST_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is None or str(job.get("user_id")) != str(user_id):
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "logs": list(job.get("logs", [])),
            "progress": job.get("progress", 0),
            "documents": list(job.get("documents", [])),
            "summary_html": job.get("summary_html"),
            "details_html": job.get("details_html"),
            "error": job.get("error"),
        }


@router.post("/ingest")
async def api_ingest(
    files: List[UploadFile] = File(...),
    user: User = Depends(require_auth),
) -> JSONResponse:
    if not rate_limiter.allow(str(user.id), "ingest"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for ingest. Try again shortly.")
    if not files:
        raise HTTPException(status_code=400, detail="Upload one or more documents with text.")
    tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="anag_upload_"))
    paths: List[pathlib.Path] = []
    names: List[str] = []
    try:
        for upload in files:
            safe_name = pathlib.Path(upload.filename).name
            dest = tmp_path / safe_name
            data = await upload.read()
            dest.write_bytes(data)
            # Persist a copy for the inline viewer
            try:
                persist_dir = pathlib.Path("artifacts") / "docs" / str(user.id)
                persist_dir.mkdir(parents=True, exist_ok=True)
                (persist_dir / safe_name).write_bytes(data)
            except Exception:
                pass
            paths.append(dest)
            names.append(safe_name)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    job_id = _start_ingest_job(paths, tmp_path, names, user_id=str(user.id))
    try:
        usage_service.record(str(user.id), "ingest")
    except Exception:
        pass
    payload = _ingest_job_payload(job_id, user_id=str(user.id))
    return JSONResponse(payload)


@router.get("/ingest/status/{job_id}")
async def api_ingest_status(job_id: str, user: User = Depends(require_auth)) -> JSONResponse:
    payload = _ingest_job_payload(job_id, user_id=str(user.id))
    return JSONResponse(payload)
