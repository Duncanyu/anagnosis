from __future__ import annotations

import os
import re
import threading
import uuid
from typing import Any, Dict, List
import base64

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.services import pipeline
from api.services import memory as mem
from api.routes.settings import apply_env_for_user
from api.core.rate_limit import rate_limiter
from api.services import usage as usage_service
from api.services.vision import describe_image_openai

try:
    import markdown as mdlib
except Exception:
    mdlib = None

router = APIRouter(prefix="/api", tags=["ask"])


def _md_to_html(md_text: str) -> str:
    if mdlib:
        return mdlib.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    return f"<pre>{(md_text or '').replace('<','&lt;').replace('>','&gt;')}</pre>"


def _vision_direct_answer(question: str, images: List[Dict[str, Any]], *, log_fn=None) -> str | None:
    """Send the user's question + attached images directly to OpenAI Vision and return the answer text.

    Returns None if no API key is configured or on error.
    """
    try:
        from api.core.config import load_config
        cfg = load_config() or {}
        key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        # Build a multimodal content array: instructions + all images
        instruction = (
            "Analyze the image(s) and answer the user's question. If there is readable text, transcribe key text first (verbatim). "
            "Then provide a thorough, well‑structured answer with short section headings and 6–10 concise bullets where helpful. "
            "Unless the user asks otherwise, aim for 180–320 words."
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": f"{instruction}\n\nQuestion: {question}"}]
        img_count = 0
        for img in (images or []):
            data = img.get("data") or img.get("data_url")
            if not isinstance(data, str) or not data.strip():
                continue
            url = data if data.strip().startswith("data:") else None
            if not url:
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})
            img_count += 1
        if log_fn:
            try:
                sizes = []
                for img in (images or []):
                    d = img.get("data") or img.get("data_url") or ""
                    if isinstance(d, str):
                        sizes.append(len(d))
                log_fn(f"Vision: building request with {img_count} image(s); sizes={sizes[:4]}")
            except Exception:
                pass
        if len(content) <= 1:
            return None
        try:
            msg = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
            )
            out = (msg.choices[0].message.content or "").strip() or None
            if log_fn:
                try: log_fn(f"Vision: response chars={len(out or '')}")
                except Exception: pass
            return out
        except Exception as e:
            if log_fn:
                try: log_fn(f"Vision error: {e}")
                except Exception: pass
            return None
    except Exception:
        return None


def _format_quotes(quotes: List[Dict[str, Any]]) -> str:
    if not quotes:
        return ""
    blocks: List[str] = ["", "### Evidence snippets"]
    for q in quotes:
        quote = q.get("quote") or q.get("text") or ""
        source = q.get("source") or q.get("citation") or ""
        if not quote:
            continue
        block = f"> {quote}\n>\n> — {source}" if source else f"> {quote}"
        blocks.append(block)
    return "\n".join(blocks)


def _format_agent_diag(result: Dict[str, Any], agents_enabled: bool) -> str:
    meta = result.get("agent_meta") if isinstance(result, dict) else None
    report = result.get("agent_report") if isinstance(result, dict) else None
    if meta and meta.get("enabled"):
        verdict = meta.get("verdict", "ran")
        if meta.get("changed"):
            verdict = "modified"
        summary_bits: List[str] = []
        kept = meta.get("kept_sentences")
        total = meta.get("total_sentences")
        if kept is not None and total:
            summary_bits.append(f"{kept}/{total} sentences kept")
        status_counts = meta.get("status_counts") or {}
        if status_counts.get("supported"):
            summary_bits.append(f"{status_counts['supported']} supported")
        if status_counts.get("weak"):
            summary_bits.append(f"{status_counts['weak']} need review")
        if meta.get("time_sec") is not None:
            summary_bits.append(f"{meta['time_sec']:.2f}s")
        summary = ", ".join(summary_bits) or "completed"
        diag = f"\n\n_Agents ({verdict}): {summary}._"
        if isinstance(report, str) and report.strip():
            diag += "\n\n<details><summary>Agent report</summary>\n\n" + report + "\n\n</details>"
        web_rep = meta.get("web_results_md")
        if isinstance(web_rep, str) and web_rep.strip():
            diag += "\n\n<details><summary>Web evidence</summary>\n\n" + web_rep + "\n\n</details>"
        return diag
    if agents_enabled:
        return "\n\n_Agents: ran._"
    return ""


def _format_answer(question: str, result: Dict[str, Any], agents_enabled: bool) -> str:
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    citations = result.get("citations") if isinstance(result, dict) else []
    quotes = result.get("quotes") if isinstance(result, dict) else []
    # Move citations to the trace chip; do not inline in the answer body
    cite_block = ""
    # Move evidence snippets into the trace chip; do not inline in the main answer
    quote_block = ""
    agent_block = _format_agent_diag(result if isinstance(result, dict) else {}, agents_enabled)
    body = (answer or "").strip()
    # Enforce readable Markdown when model returns a plain blob
    try:
        blob = (answer or "").strip()
        has_md = bool(re.search(r"(^#|\n\s*[-*]|\n\s*\d+\.|```|\*\*|\|)", blob, flags=re.M))
        if not has_md and len(blob) > 120:
            sents = re.split(r"(?<=[\.!?])\s+", blob)
            sents = [s.strip() for s in sents if s.strip()]
            if sents:
                head = sents[0]
                bullets = sents[1:6]
                rebuilt = f"**{head}**" + ("\n\n" + "\n".join(f"- {b}" for b in bullets) if bullets else "")
                body = rebuilt
    except Exception:
        pass

    # Do not echo the user's question; keep model-provided headings as-is.
    return body + cite_block + quote_block + agent_block


ASK_JOBS: Dict[str, Dict[str, Any]] = {}
ASK_LOCK = threading.Lock()


def _append_ask_log(job_id: str, message: str) -> None:
    with ASK_LOCK:
        job = ASK_JOBS.get(job_id)
        if job is not None:
            job.setdefault("logs", []).append(message)

def _set_cancel(job_id: str) -> None:
    with ASK_LOCK:
        job = ASK_JOBS.get(job_id)
        if job is not None:
            job["cancel"] = True
            # Reflect cancellation immediately in status so the UI stops polling
            job["status"] = "cancelled"
            job.setdefault("logs", []).append("Cancelled by user")


def _ask_job_payload(job_id: str, user_id: str) -> Dict[str, Any]:
    with ASK_LOCK:
        job = ASK_JOBS.get(job_id)
        if job is None or str(job.get("user_id")) != str(user_id):
            raise HTTPException(status_code=404, detail="Ask job not found.")
        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "logs": list(job.get("logs", [])),
            "answer": job.get("answer"),
            "answer_markdown": job.get("answer_markdown"),
            "error": job.get("error"),
            "trace": job.get("trace"),
        }


def _decode_data_url(data: str) -> bytes:
    """Decode a data URL or base64 string to raw bytes.

    Accepts either a full data URL (data:<mime>;base64,...) or a bare base64 string.
    """
    if not isinstance(data, str):
        return b""
    s = data.strip()
    try:
        if s.startswith("data:"):
            head, b64 = s.split(",", 1)
            return base64.b64decode(b64)
        # If it looks like base64, decode directly
        return base64.b64decode(s)
    except Exception:
        return b""


def _attachments_to_text(images: List[Dict[str, Any]], *, log_fn=None) -> str:
    out_blocks: List[str] = []
    if not images:
        return ""
    if log_fn:
        try:
            log_fn(f"Processing {len(images)} image(s)…")
        except Exception:
            pass
    # Use OpenAI Vision only; do not run local OCR.
    openai_present = bool(os.environ.get("OPENAI_API_KEY"))
    for i, img in enumerate(images, 1):
        name = str(img.get("name") or f"image_{i}")
        data = img.get("data") or img.get("data_url") or ""
        raw = _decode_data_url(data)
        if not raw:
            # Also accept 'bytes' field (hex) or ignore
            continue
        # Guard against very large images
        try:
            if len(raw) > 5 * 1024 * 1024:
                if log_fn:
                    try: log_fn(f"Skipping {name}: image too large (>5MB)")
                    except Exception: pass
                continue
        except Exception:
            pass
        # Vision-only path
        vision_text = None
        if openai_present:
            try:
                vision_text = describe_image_openai(raw)
            except Exception:
                vision_text = None
        else:
            if log_fn:
                try:
                    log_fn("Vision unavailable (no OPENAI_API_KEY); skipping image(s).")
                except Exception:
                    pass
        parts = [f"Image: {name}"]
        if vision_text:
            parts.append("Vision:\n" + str(vision_text).strip())
        if not vision_text:
            parts.append("(Vision unavailable.)")
        block = "\n\n".join(parts)
        if block.strip():
            out_blocks.append(block)
    if not out_blocks:
        return ""
    header = "Context from attached images (OCR + vision when available):"
    body = ("\n\n".join(out_blocks)).strip()
    return f"{header}\n\n{body}"


def _attachments_to_chunks(images: List[Dict[str, Any]], *, question: str | None = None, log_fn=None) -> List[Dict[str, Any]]:
    """Convert image payloads to pseudo-document chunks using OpenAI Vision.

    Each image is converted into a chunk with doc_name 'Attachment: <name>' so the
    summarizer can treat it as a normal source document.
    """
    chunks: List[Dict[str, Any]] = []
    if not images:
        return chunks
    openai_present = bool(os.environ.get("OPENAI_API_KEY"))
    for i, img in enumerate(images, 1):
        name = str(img.get("name") or f"image_{i}")
        data = img.get("data") or img.get("data_url") or ""
        raw = _decode_data_url(data)
        if not raw:
            continue
        # size guard
        try:
            if len(raw) > 8 * 1024 * 1024:
                if log_fn:
                    try: log_fn(f"Skipping {name}: image too large (>8MB)")
                    except Exception: pass
                continue
        except Exception:
            pass
        text = None
        if openai_present:
            try:
                # Pass the user's question to guide the vision model
                p = None
                if question and str(question).strip():
                    q = str(question).strip()
                    p = (
                        "Answer the user's question using this image. "
                        "If relevant, transcribe any readable text verbatim first, then provide a concise answer.\n\n"
                        f"Question: {q}"
                    )
                text = describe_image_openai(raw, prompt=p)
            except Exception:
                text = None
        if not (text and str(text).strip()):
            if log_fn:
                try: log_fn(f"Attachment {i}: Vision unavailable or empty for {name}")
                except Exception: pass
            continue
        chunk = {
            "text": str(text).strip(),
            "doc_name": f"Attachment: {name}",
            "page_start": 1,
            "page_end": 1,
            "section_tag": "attachment",
            "_score": 0.95,
        }
        chunks.append(chunk)
    return chunks


@router.post("/ask/start")
async def api_ask_start(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    if not rate_limiter.allow(str(user.id), "ask"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for ask. Try again shortly.")
    payload = await request.json()
    job_id = uuid.uuid4().hex
    with ASK_LOCK:
        ASK_JOBS[job_id] = {
            "status": "running",
            "logs": ["Working…"],
            "answer": None,
            "answer_markdown": None,
            "error": None,
            "user_id": str(user.id),
        }

    def worker() -> None:
        try:
            restore_env = apply_env_for_user(str(user.id))
            # Validate keys for selected backends before doing any work
            be_llm = (os.environ.get("LLM_BACKEND", "openai") or "").lower()
            be_embed = (os.environ.get("EMBED_BACKEND", "hf") or "").lower()
            need_openai = (be_llm == "openai") or (be_embed == "openai")
            errs = []
            if need_openai and not os.environ.get("OPENAI_API_KEY"):
                errs.append("Missing OpenAI API key")
            question = (payload.get("question") or "").strip()
            if not question:
                raise HTTPException(status_code=400, detail="Question required.")

            history = payload.get("history") or []
            formula_mode = bool(payload.get("formula_mode"))
            agents_enabled = bool(payload.get("agents_enabled"))
            strict_docs = bool(payload.get("strict_docs"))
            web_enabled = bool(payload.get("web_enabled"))
            top_k = int(payload.get("top_k") or 10)
            only_doc = (payload.get("only_doc") or "").strip() or None

            # Per-request overrides from chat controls (don't require visiting Settings)
            try:
                req_rerank = str(payload.get("reranker") or "").strip().lower()
                if req_rerank:
                    os.environ["ASK_RERANKER"] = (req_rerank if req_rerank not in {"none", ""} else "off")
                req_mb = payload.get("max_batches")
                if req_mb is not None:
                    os.environ["ASK_MAX_BATCHES"] = str(int(req_mb))
                req_tb = payload.get("time_budget")
                if req_tb is not None:
                    os.environ["ASK_TIME_BUDGET_SEC"] = str(int(req_tb))
                req_cand = payload.get("ask_candidates")
                if req_cand is not None:
                    os.environ["ASK_CANDIDATES"] = str(int(req_cand))
            except Exception:
                pass

            # Additional provider checks once flags are known
            if web_enabled:
                provider = (os.environ.get("WEB_SEARCH_PROVIDER") or "duckduckgo").lower()
                if provider == "serpapi" and not os.environ.get("SERPAPI_KEY"):
                    errs.append("Missing SerpAPI key")
                if provider == "brave" and not os.environ.get("BRAVE_API_KEY"):
                    errs.append("Missing Brave API key")
            if errs:
                with ASK_LOCK:
                    job = ASK_JOBS.get(job_id)
                    if job is not None:
                        job["status"] = "error"
                        job["error"] = "; ".join(errs)
                        job.setdefault("logs", []).append("Configuration error: " + job["error"])
                return

            def log(msg: str) -> None:
                _append_ask_log(job_id, msg)

            # If images exist, try a direct Vision answer first (image + question)
            try:
                images = payload.get("images") if isinstance(payload.get("images"), list) else []
            except Exception:
                images = []
            if images:
                try:
                    log("Vision: answering directly from attached image(s)…")
                except Exception:
                    pass
                direct = _vision_direct_answer(question, images, log_fn=log)
                if direct:
                    formatted_md = (direct or "").strip()
                    answer_html = _md_to_html(formatted_md)
                    with ASK_LOCK:
                        job = ASK_JOBS.get(job_id)
                        if job is not None:
                            job["status"] = "done"
                            job["answer"] = answer_html
                            job["answer_markdown"] = formatted_md
                    return
            # Otherwise build attachment chunks and proceed with pipeline
            attach_chunks = _attachments_to_chunks(images, question=question, log_fn=log)
            history_payload = list(history)
            # Provide a cancel check for cooperative cancellation (best-effort)
            def should_cancel():
                with ASK_LOCK:
                    j = ASK_JOBS.get(job_id) or {}
                    return bool(j.get("cancel"))

            result = pipeline.answer_question(
                question,
                k=top_k,
                history=history_payload,
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                progress=log,
                should_cancel=should_cancel,
                only_doc=only_doc,
                user_id=str(user.id),
                exhaustive=bool(payload.get("exhaustive")),
                extra_chunks=attach_chunks,
            )

            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)

            # Persist conversational memory per-user (best-effort)
            try:
                if question and isinstance(result, dict):
                    raw_answer = (result.get("answer") or "").strip()
                    if raw_answer:
                        mem.append_turn(question, raw_answer, user_id=str(user.id))
                        mem.prune_file(max_mb=int(os.getenv("MEMORY_FILE_LIMIT_MB", "50")), user_id=str(user.id))
            except Exception:
                pass

            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    if job.get("cancel"):
                        job["status"] = "cancelled"
                        job.setdefault("logs", []).append("Cancelled")
                    else:
                        job["status"] = "done"
                        job["answer"] = answer_html
                        job["answer_markdown"] = formatted_md
                        try:
                            if isinstance(result, dict):
                                tr = result.get("trace") or {}
                                try:
                                    cits = result.get("citations")
                                    if isinstance(cits, list):
                                        tr["citations"] = cits
                                except Exception:
                                    pass
                                if tr:
                                    job["trace"] = tr
                        except Exception:
                            pass
        except Exception as exc:
            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    if job.get("cancel"):
                        job["status"] = "cancelled"
                        job.setdefault("error", "Cancelled")
                    else:
                        job["status"] = "error"
                        job["error"] = str(exc)
        finally:
            try:
                restore_env()
            except Exception:
                pass
        
    threading.Thread(target=worker, daemon=True).start()
    try:
        usage_service.record(str(user.id), "ask")
    except Exception:
        pass
    return JSONResponse(_ask_job_payload(job_id, user_id=str(user.id)))


@router.get("/ask/status/{job_id}")
async def api_ask_status(job_id: str, user: User = Depends(require_auth)) -> JSONResponse:
    return JSONResponse(_ask_job_payload(job_id, user_id=str(user.id)))


@router.post("/ask")
async def api_ask(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    if not rate_limiter.allow(str(user.id), "ask"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for ask. Try again shortly.")
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question required.")

    history = payload.get("history") or []
    formula_mode = bool(payload.get("formula_mode"))
    agents_enabled = bool(payload.get("agents_enabled"))
    strict_docs = bool(payload.get("strict_docs"))
    web_enabled = bool(payload.get("web_enabled"))
    top_k = int(payload.get("top_k") or 10)
    only_doc = (payload.get("only_doc") or "").strip() or None

    # Per-request overrides from chat controls (don't require visiting Settings)
    try:
        req_rerank = str(payload.get("reranker") or "").strip().lower()
        if req_rerank:
            os.environ["ASK_RERANKER"] = (req_rerank if req_rerank not in {"none", ""} else "off")
        req_mb = payload.get("max_batches")
        if req_mb is not None:
            os.environ["ASK_MAX_BATCHES"] = str(int(req_mb))
        req_tb = payload.get("time_budget")
        if req_tb is not None:
            os.environ["ASK_TIME_BUDGET_SEC"] = str(int(req_tb))
        req_cand = payload.get("ask_candidates")
        if req_cand is not None:
            os.environ["ASK_CANDIDATES"] = str(int(req_cand))
    except Exception:
        pass

    # Validate keys for selected backends
    be_llm = (os.environ.get("LLM_BACKEND", "openai") or "").lower()
    be_embed = (os.environ.get("EMBED_BACKEND", "hf") or "").lower()
    errs = []
    need_openai = (be_llm == "openai") or (be_embed == "openai")
    if need_openai and not os.environ.get("OPENAI_API_KEY"):
        errs.append("Missing OpenAI API key")
    if web_enabled:
        provider = (os.environ.get("WEB_SEARCH_PROVIDER") or "duckduckgo").lower()
        if provider == "serpapi" and not os.environ.get("SERPAPI_KEY"):
            errs.append("Missing SerpAPI key")
        if provider == "brave" and not os.environ.get("BRAVE_API_KEY"):
            errs.append("Missing Brave API key")
    if errs:
        raise HTTPException(status_code=400, detail="; ".join(errs))

    restore_env = apply_env_for_user(str(user.id))
    logs: List[str] = []
    def log(msg: str) -> None:
        logs.append(msg)

    try:
        # Prefer a direct Vision answer when images are present
        try:
            images = payload.get("images") if isinstance(payload.get("images"), list) else []
        except Exception:
            images = []
        if images:
            ans = _vision_direct_answer(question, images, log_fn=log)
            if ans:
                result = {"answer": ans, "citations": [], "quotes": []}
            else:
                attach_chunks = _attachments_to_chunks(images, question=question, log_fn=log)
                hist_payload = list(history)
                result = pipeline.answer_question(
                    question,
                    k=top_k,
                    history=hist_payload,
                    formula_mode=formula_mode,
                    agents_enabled=agents_enabled,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    progress=log,
                    exhaustive=bool(payload.get("exhaustive")),
                    extra_chunks=attach_chunks,
                )
        else:
            hist_payload = list(history)
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=hist_payload,
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                progress=log,
                exhaustive=bool(payload.get("exhaustive")),
            )
    finally:
        try:
            restore_env()
        except Exception:
            pass
    formatted_md = _format_answer(question, result, agents_enabled)
    answer_html = _md_to_html(formatted_md)
    try:
        usage_service.record(str(user.id), "ask")
    except Exception:
        pass
    try:
        tr = None
        if isinstance(result, dict):
            tr = result.get("trace") or {}
            cits = result.get("citations")
            if isinstance(cits, list):
                tr["citations"] = cits
    except Exception:
        tr = result.get("trace") if isinstance(result, dict) else None
    return JSONResponse({
        "answer": answer_html,
        "answer_markdown": formatted_md,
        "log": "\n".join(logs) if logs else "Ready.",
        "trace": tr,
    })
@router.post("/ask/cancel/{job_id}")
async def api_ask_cancel(job_id: str, user: User = Depends(require_auth)) -> JSONResponse:
    _set_cancel(job_id)
    return JSONResponse({"ok": True})
