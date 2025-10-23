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
from api.services import index as index_service
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
    cite_block = ""
    quote_block = ""
    agent_block = _format_agent_diag(result if isinstance(result, dict) else {}, agents_enabled)
    body = (answer or "").strip()
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
            "actions": job.get("actions") if isinstance(job.get("actions"), list) else [],
            "next_steps": job.get("next_steps") if isinstance(job.get("next_steps"), list) else [],
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
                            job["actions"] = []
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
                        # Persist any suggested actions for UI
                        try:
                            job["actions"] = result.get("actions") if isinstance(result, dict) and isinstance(result.get("actions"), list) else []
                        except Exception:
                            job["actions"] = []
                        # Persist next_steps for UI
                        try:
                            from api.services.pipeline import extract_next_steps
                            ns = []
                            try:
                                ns = result.get('next_steps') if isinstance(result.get('next_steps'), list) else []
                            except Exception:
                                ns = []
                            if not ns:
                                try:
                                    ns = extract_next_steps(result.get('answer') or '')
                                except Exception:
                                    ns = []
                            job['next_steps'] = ns
                        except Exception:
                            job['next_steps'] = []
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
                        try:
                            print(f"[DEBUG] ask job_done actions={job.get('actions')} next_steps={job.get('next_steps')}")
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
    actions = []
    next_steps = []
    try:
        if isinstance(result, dict):
            actions = result.get("actions") if isinstance(result.get("actions"), list) else []
            # Try to get next_steps from the result first (LLM-generated)
            next_steps = result.get("next_steps") if isinstance(result.get("next_steps"), list) else []
            # If no next_steps in result, try to extract from answer text as fallback
            if not next_steps:
                from api.services.pipeline import extract_next_steps
                next_steps = extract_next_steps(result.get("answer") or "")
    except Exception:
        actions = []
        next_steps = []

    # DEBUG: print actions/next_steps to server stdout to help debug frontend integration
    try:
        print(f"[DEBUG] /api/ask -> actions={actions} next_steps={next_steps}")
    except Exception:
        pass

    return JSONResponse({
        "answer": answer_html,
        "answer_markdown": formatted_md,
        "log": "\n".join(logs) if logs else "Ready.",
        "trace": tr,
        "actions": actions,
        "next_steps": next_steps,
    })
@router.post("/ask/cancel/{job_id}")
async def api_ask_cancel(job_id: str, user: User = Depends(require_auth)) -> JSONResponse:
    _set_cancel(job_id)
    return JSONResponse({"ok": True})


@router.post("/ask/action")
async def api_ask_action(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    """Execute a suggested action (from the assistant's `actions` array).

    Payload should include the same fields as `/api/ask` plus an `action_id`.
    Actions that change sources will re-run the pipeline and return an updated answer.
    """
    if not rate_limiter.allow(str(user.id), "ask"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for ask. Try again shortly.")
    payload = await request.json()
    action_id = (payload.get("action_id") or "").strip()
    if not action_id:
        raise HTTPException(status_code=400, detail="action_id required")

    # Basic request params (same as /ask)
    question = (payload.get("question") or "").strip()
    history = payload.get("history") or []
    formula_mode = bool(payload.get("formula_mode"))
    agents_enabled = bool(payload.get("agents_enabled"))
    strict_docs = bool(payload.get("strict_docs"))
    web_enabled = bool(payload.get("web_enabled"))
    top_k = int(payload.get("top_k") or 10)
    only_doc = (payload.get("only_doc") or "").strip() or None
    images = payload.get("images") if isinstance(payload.get("images"), list) else []

    # Demo mode hardening: optionally enforce stable, offline-friendly behavior
    demo_mode = str(os.environ.get("DEMO_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
    if demo_mode:
        # In demo recordings we want deterministic, fast, offline-friendly runs
        web_enabled = False
        strict_docs = True
        agents_enabled = False
        # Keep results concise to reduce load
        try:
            top_k = max(5, min(15, int(top_k)))
        except Exception:
            top_k = 10

    # Helper function to get a proper search question when the current question is just an action label
    def get_search_question(current_q: str, action_labels: list = None) -> str:
        """
        If current question is just an action label, try to find the last real question from history.
        Falls back to a generic query if no good question is found.
        """
        if action_labels is None:
            action_labels = [
                'Generate Quiz', 'Suggest Follow-ups', 'Generate Formula Sheet', 'Summarize Documents',
                'Create Mind Map', 'Compare Sources', 'Simplify Explanation', 'List Citations',
                'Expand Answer', 'Translate Answer', 'Show Reasoning', 'Identify Knowledge Gaps',
                'Recommend Documents', 'generate_quiz', 'followup_questions', 'generate_formula_sheet',
                'summarize_docs', 'mindmap_outline', 'compare_sources', 'simplify_explanation',
                'cite_sources_only', 'expand_detail', 'translate_answer', 'debug_reasoning_trace',
                'detect_gaps', 'recommend_new_docs'
            ]
        
        if current_q not in action_labels and current_q:
            return current_q
        
        # Try to find last actual user question from history
        for turn in reversed(history):
            if isinstance(turn, dict):
                q = (turn.get('q') or '').strip()
                if q and q not in action_labels:
                    return q
        
        # If still no good question, use a generic query
        return "overview of the main topics and key information"

    restore_env = apply_env_for_user(str(user.id))
    try:
        be_llm = (os.environ.get("LLM_BACKEND", "openai") or "").lower()
        be_embed = (os.environ.get("EMBED_BACKEND", "hf") or "").lower()
        need_openai = (be_llm == "openai") or (be_embed == "openai")
        if need_openai and not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OpenAI API key required to execute actions.")

        attach_chunks = []
        try:
            if images:
                attach_chunks = _attachments_to_chunks(images, question=question, log_fn=None)
        except Exception:
            attach_chunks = []

        if action_id == "enable_web":
            web_enabled = True
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=False,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), "ask_action")
            except Exception:
                pass
            return JSONResponse({"answer": answer_html, "answer_markdown": formatted_md, "result": result})

        if action_id == "broaden_docs":
            only_doc = None
            strict_docs = False
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), "ask_action")
            except Exception:
                pass
            return JSONResponse({"answer": answer_html, "answer_markdown": formatted_md, "result": result})

        if action_id == "upload_docs":
            # UI-driven: ask the frontend to open an upload dialog. No pipeline re-run here.
            return JSONResponse({"ui": {"open_upload": True}, "message": "Please attach documents to include in the search."})

        if action_id == "clarify_question":
            # Use OpenAI to craft a concise clarifying question
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for clarify action.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                prompt = (
                    "You are a helpful assistant. Produce a single concise clarifying question (<=20 words) the assistant should ask the user to disambiguate their intent. "
                    "Also provide a one-sentence reason why this clarification helps. Return JSON with keys 'question' and 'reason'.\n\n"
                    f"Original question: {question}\n"
                )
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                out = (msg.choices[0].message.content or "").strip()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Clarify action failed: {exc}")
            # Try to parse JSON-like output, else return as plain text
            try:
                import json
                parsed = json.loads(out)
                qtxt = parsed.get("question") or parsed.get("clarifying_question") or out
                reason = parsed.get("reason") or ""
            except Exception:
                # Fallback: split into first line and rest
                lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
                qtxt = lines[0] if lines else out
                reason = (lines[1] if len(lines) > 1 else "")
            return JSONResponse({"clarifying_question": qtxt, "reason": reason})

        if action_id == "expand_detail":
            # Re-run pipeline with an explicit expansion instruction
            # Stronger guidance: add NEW material, avoid repetition, surface additional evidence and clear follow-ups
            expand_q = (
                "Expand the previous answer with fresh, non-overlapping content. Do NOT repeat earlier sentences. "
                "Add 3–5 new pieces of evidence with inline citations (author/title or URL if available). "
                "Begin with a brief 'What's new' bullet list summarizing the added insights. "
                "Keep structure clear, tighten wording, and integrate any missing caveats, counterpoints, or edge cases. "
                "Conclude with 4–6 very specific next actions the user can take. Then answer: \n\n" + question
            )
            result = pipeline.answer_question(
                expand_q,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(expand_q, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), "ask_action")
            except Exception:
                pass
            return JSONResponse({"answer": answer_html, "answer_markdown": formatted_md, "result": result})

        # New action handlers: accept optional action_params from the payload
        action_params = payload.get('action_params') if isinstance(payload.get('action_params'), dict) else {}

        if action_id == 'select_doc' or action_id == 'only_doc':
            # Select a single document to focus the search. Expect params: { "doc": "filename.pdf" }
            sel = (action_params.get('doc') or payload.get('only_doc') or '').strip() or None
            only_doc = sel
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'set_top_k' or action_id == 'set_k':
            # Change the number of retrieval/top-k. Expect params: { "k": 8 }
            try:
                newk = int(action_params.get('k') if action_params.get('k') is not None else payload.get('top_k') or top_k)
                top_k = max(1, min(200, newk))
            except Exception:
                top_k = top_k
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'toggle_reranker' or action_id == 'set_reranker':
            # Toggle or set a reranker. Expect params: { "reranker": "minilm" } or { "on": true }
            try:
                rr = action_params.get('reranker') if action_params.get('reranker') is not None else payload.get('reranker')
                if rr is None:
                    # toggle: if current env is off, set to minilm
                    cur = str(os.environ.get('ASK_RERANKER') or 'off')
                    rr = 'minilm' if cur in {'off', '', 'none'} else 'off'
                rr = str(rr) if rr is not None else 'off'
                os.environ['ASK_RERANKER'] = (rr if rr not in {'none', ''} else 'off')
            except Exception:
                pass
            # re-run with updated reranker
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'generate_formula_sheet' or action_id == 'formula_sheet':
            # Run pipeline in formula mode to generate concise formulas. No change to web/only_doc.
            # Get a proper search question
            search_question = get_search_question(question)
            
            result = pipeline.answer_question(
                search_question,
                k=top_k,
                history=list(history),
                formula_mode=True,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'strict_docs_only' or action_id == 'rag_only':
            # Switch to RAG-only mode (strict docs, no web)
            strict_docs = True
            web_enabled = False
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'exhaustive_search' or action_id == 'exhaustive':
            # Enable exhaustive search mode
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
                exhaustive=True,
            )
            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'summarize_docs':
            # Generate concise summaries of relevant documents
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for summarize action.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get a proper search question - if the user just clicked "Summarize Documents"
                # without a prior question, we need to extract context from history or use a generic query
                search_question = get_search_question(question)
                
                # First get relevant documents via pipeline
                result = pipeline.answer_question(
                    search_question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                # Extract citations/chunks from result
                citations = result.get('citations', []) if isinstance(result, dict) else []
                if not citations:
                    formatted_md = "**Document Summaries**\n\nNo documents found for this query."
                else:
                    # Build prompt to summarize each document - handle both dict and string citations
                    docs_parts = []
                    for c in citations[:5]:
                        if isinstance(c, dict):
                            source = c.get('source', 'Unknown')
                            text = c.get('text', '')
                        elif isinstance(c, str):
                            source = 'Document'
                            text = c
                        else:
                            continue
                        if text:
                            docs_parts.append(f"**{source}**:\n{text}")
                    
                    if not docs_parts:
                        formatted_md = "**Document Summaries**\n\nNo document content available."
                    else:
                        docs_text = "\n\n---\n\n".join(docs_parts)
                        prompt = f"Summarize each of the following document excerpts in 2-3 concise bullet points:\n\n{docs_text}"
                        
                        msg = client.chat.completions.create(
                            model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                        )
                        summary = (msg.choices[0].message.content or "").strip()
                        formatted_md = f"**Document Summaries**\n\n{summary}"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Summarize action failed: {exc}")

        if action_id == 'generate_quiz':
            # Create quiz questions to test understanding
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for quiz generation.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Desired number of questions (default 5, clamp 1–30)
                try:
                    req_count = int((action_params.get('count') if isinstance(action_params, dict) else None) or 5)
                except Exception:
                    req_count = 5
                question_count = max(1, min(30, req_count))
                
                previous_answer = ""
                if history and len(history) > 0:
                    last_turn = history[-1]
                    previous_answer = last_turn.get('a') or last_turn.get('a_markdown') or ''
                
                # Get a proper search question
                search_question = get_search_question(question)
                
                result = pipeline.answer_question(
                    search_question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                # Determine primary document to focus the quiz on
                primary_doc = None
                trace = result.get('trace') if isinstance(result, dict) else {}
                try:
                    # Prefer selected_context docs (stronger signal)
                    selected_ctx = trace.get('selected_context') if isinstance(trace, dict) else []
                    freq = {}
                    for it in (selected_ctx or []):
                        if not isinstance(it, dict):
                            continue
                        if str(it.get('kind') or 'doc') != 'doc':
                            continue
                        dn = (it.get('doc') or '').strip()
                        if dn:
                            freq[dn] = freq.get(dn, 0) + 1
                    if not primary_doc and freq:
                        primary_doc = sorted(freq.items(), key=lambda kv: -kv[1])[0][0]
                except Exception:
                    primary_doc = None
                if not primary_doc:
                    # Fall back to citations tag parsing like [Doc.pdf p.N]
                    try:
                        cites = result.get('citations', []) if isinstance(result, dict) else []
                        doc_counts = {}
                        import re as _re
                        for tag in cites:
                            if not isinstance(tag, str):
                                continue
                            m = _re.match(r"^\[([^\]]+?)\s+p\.", tag.strip())
                            if m:
                                dn = m.group(1).strip()
                                if dn:
                                    doc_counts[dn] = doc_counts.get(dn, 0) + 1
                        if doc_counts:
                            primary_doc = sorted(doc_counts.items(), key=lambda kv: -kv[1])[0][0]
                    except Exception:
                        primary_doc = None
                if not primary_doc and only_doc:
                    primary_doc = only_doc
                
                # Build rich context: include as many chunks as possible from the primary doc
                context_snippets = []
                if primary_doc:
                    try:
                        # Pull all chunks for this user and document
                        all_user_chunks = index_service.list_chunks(user_id=str(user.id))
                        import pathlib as _pl
                        base = _pl.Path(str(primary_doc)).name.lower()
                        doc_chunks = [c for c in (all_user_chunks or []) if _pl.Path(str(c.get('doc_name') or '')).name.lower() == base]
                        # Order by page_start then rid for stable coverage
                        doc_chunks.sort(key=lambda c: (
                            int(c.get('page_start') or c.get('page') or 0) if isinstance(c.get('page_start') or c.get('page'), int) else 0,
                            int(c.get('rid') or 0)
                        ))
                        # Construct a large-but-bounded context
                        char_budget = 16000
                        used = 0
                        for c in doc_chunks:
                            if used >= char_budget:
                                break
                            txt = (c.get('text') or '').strip()
                            if not txt:
                                continue
                            pg = c.get('page_start') or c.get('page') or '?'
                            sn = txt[:600]
                            block = f"[p.{pg}] {sn}"
                            context_snippets.append(block)
                            used += len(block)
                    except Exception:
                        context_snippets = []
                
                # If no primary doc context, fallback to selected_context snippets
                if not context_snippets:
                    try:
                        sel = (trace.get('selected_context') if isinstance(trace, dict) else []) or []
                        for it in sel[:8]:
                            if not isinstance(it, dict):
                                continue
                            if str(it.get('kind') or 'doc') != 'doc':
                                continue
                            doc = it.get('doc') or 'Unknown'
                            sn = it.get('snippet') or ''
                            if sn:
                                context_snippets.append(f"From {doc}: {sn[:400]}")
                    except Exception:
                        pass
                
                # Build conversation context summary
                conversation_context = []
                previous_quiz_topics = []
                if len(history) > 1:
                    for turn in history[-3:]:  # Last 3 turns for context
                        q = turn.get('q', '')
                        if q:
                            conversation_context.append(f"Q: {q}")
                
                    # Extract topics from previous quizzes in history
                    try:
                        for turn in history:
                            a_md = turn.get('a_markdown', '') or turn.get('a', '')
                            # Check if this turn contains a quiz (look for "Quiz:" marker or multiple questions)
                            if 'Quiz:' in a_md or ('Question' in a_md and 'Answer:' in a_md):
                                # Extract question topics from quiz markdown
                                lines = a_md.split('\n')
                                for line in lines:
                                    # Match lines like "1. What is...?" or lines with numbered questions
                                    if line.strip() and (line[0].isdigit() or line.strip().startswith('-')):
                                        # Extract just the question text (first ~80 chars)
                                        cleaned = line.strip().lstrip('0123456789.-) ').split('\n')[0][:80]
                                        if len(cleaned) > 10 and '?' in cleaned:
                                            previous_quiz_topics.append(cleaned)
                    except Exception:
                        pass
                
                # Build comprehensive prompt
                prompt_parts = [
                    f"Generate EXACTLY {question_count} multiple-choice quiz questions based on the following information.",
                    "Focus on the main answer content, but also consider the conversation context and document sources.",
                    ]
                
                if previous_quiz_topics:
                        prompt_parts.extend([
                            "",
                            "IMPORTANT: The user has already been quizzed on these topics. Generate NEW questions covering DIFFERENT aspects:",
                            "\n".join([f"- {t}" for t in previous_quiz_topics[-20:]]),  # Last 20 quiz questions
                            "",
                        ])
                
                prompt_parts.extend([
                        f"Return ONLY a JSON array of length EXACTLY {question_count} where each question is an object with:",
                        "- question (string): the quiz question",
                        "- options (array of 4 strings): the answer choices",
                        "- correct (integer 0-3): index of the correct answer",
                        "- explanation (string): why the correct answer is right",
                        "",
                        "IMPORTANT: Distribute correct answers randomly across all four positions (A/B/C/D). Do NOT favor any position.",
                        "Aim for roughly equal distribution: ~25% each for indices 0, 1, 2, and 3.",
                        "",
                        "PREVIOUS ANSWER (main content to quiz on):",
                        previous_answer[:2000],  # Limit length
                        ""
                ])
                
                if conversation_context:
                    prompt_parts.extend([
                        "CONVERSATION CONTEXT:",
                        "\n".join(conversation_context),
                        ""
                    ])
                
                if context_snippets:
                    if primary_doc:
                        prompt_parts.extend([
                            f"RELEVANT DOCUMENT: {primary_doc}",
                        ])
                    prompt_parts.extend([
                        "DOCUMENT EXCERPTS (ordered, include page refs):",
                        "\n".join(context_snippets),
                        ""
                    ])
                
                prompt_parts.append("Generate the JSON array now (no markdown, just the array). Do not include any prose before or after the JSON array:")
                prompt = "\n".join(prompt_parts)
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                raw_output = (msg.choices[0].message.content or "").strip()
                
                try:
                    import json
                    import re
                    if '```' in raw_output:
                        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_output, re.DOTALL)
                        if json_match:
                            raw_output = json_match.group(1)
                    quiz_parsed = json.loads(raw_output)
                    if isinstance(quiz_parsed, dict):
                        quiz_data = [quiz_parsed]
                    elif isinstance(quiz_parsed, list):
                        quiz_data = quiz_parsed
                    else:
                        quiz_data = []
                except Exception:
                    # Fallback: return error message
                    quiz_data = []

                # Enforce exact question count: trim extras or top-up via one additional request
                try:
                    if isinstance(quiz_data, list):
                        # Basic sanitation: ensure each item has expected keys
                        def _valid(q):
                            try:
                                if not isinstance(q, dict):
                                    return False
                                if not q.get('question'):
                                    return False
                                opts = q.get('options') or []
                                if not isinstance(opts, list) or len(opts) < 4:
                                    return False
                                ci = int(q.get('correct', 0))
                                return 0 <= ci <= 3
                            except Exception:
                                return False
                        quiz_data = [q for q in quiz_data if _valid(q)]
                        if len(quiz_data) > question_count:
                            quiz_data = quiz_data[:question_count]
                        elif len(quiz_data) < question_count:
                            # Attempt up to 3 follow-up generations to reach the exact count
                            attempts = 0
                            while len(quiz_data) < question_count and attempts < 3:
                                attempts += 1
                                missing = question_count - len(quiz_data)
                                try:
                                    existing_titles = [str((q.get('question') or '')).strip()[:200] for q in quiz_data]
                                    add_parts = [
                                        f"Generate EXACTLY {missing} additional multiple-choice quiz questions based on the same information.",
                                        "Avoid duplicates of the existing questions listed below.",
                                        f"Return ONLY a JSON array of length EXACTLY {missing} with the same schema as before.",
                                        "",
                                        "IMPORTANT: Distribute correct answers randomly across all four positions (A/B/C/D). Do NOT favor any position.",
                                        "",
                                        "EXISTING QUESTIONS (avoid repeating these):",
                                        "\n".join([f"- {t}" for t in existing_titles if t]),
                                        "",
                                    ]
                                    # Reuse context
                                    add_parts.append("PREVIOUS ANSWER (main content to quiz on):")
                                    add_parts.append(previous_answer[:2000])
                                    add_parts.append("")
                                    if conversation_context:
                                        add_parts.extend(["CONVERSATION CONTEXT:", "\n".join(conversation_context), ""]) 
                                    if context_snippets:
                                        if primary_doc:
                                            add_parts.extend([f"RELEVANT DOCUMENT: {primary_doc}"])
                                        add_parts.extend(["DOCUMENT EXCERPTS:", "\n".join(context_snippets), ""]) 
                                    add_parts.append("Generate the JSON array now (no markdown, just the array). Do not include any prose before or after the JSON array:")
                                    add_prompt = "\n".join(add_parts)

                                    add_msg = client.chat.completions.create(
                                        model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                                        messages=[{"role": "user", "content": add_prompt}],
                                        temperature=0.4,
                                    )
                                    add_raw = (add_msg.choices[0].message.content or "").strip()
                                    try:
                                        if '```' in add_raw:
                                            import re as _re2
                                            m2 = _re2.search(r'```(?:json)?\s*(\[.*?\])\s*```', add_raw, _re2.DOTALL)
                                            if m2:
                                                add_raw = m2.group(1)
                                        add_parsed = json.loads(add_raw)
                                        if isinstance(add_parsed, dict):
                                            add_list = [add_parsed]
                                        elif isinstance(add_parsed, list):
                                            add_list = add_parsed
                                        else:
                                            add_list = []
                                    except Exception:
                                        add_list = []
                                    def _key(q):
                                        try:
                                            return (str(q.get('question') or '').strip() or '')[:200].lower()
                                        except Exception:
                                            return ''
                                    seen = { _key(q) for q in quiz_data }
                                    for q in add_list:
                                        if _valid(q) and _key(q) not in seen:
                                            quiz_data.append(q)
                                            seen.add(_key(q))
                                        if len(quiz_data) >= question_count:
                                            break
                                except Exception:
                                    # Continue attempts if any parsing or generation issue
                                    continue
                        # Final trim in case of overflow
                        if len(quiz_data) > question_count:
                            quiz_data = quiz_data[:question_count]
                        
                        # Shuffle options to randomize correct answer positions
                        import random
                        for q in quiz_data:
                            if not isinstance(q, dict):
                                continue
                            opts = q.get('options')
                            if not isinstance(opts, list) or len(opts) < 4:
                                continue
                            try:
                                correct_idx = int(q.get('correct', 0))
                                if 0 <= correct_idx < len(opts):
                                    correct_answer = opts[correct_idx]
                                    # Shuffle options
                                    random.shuffle(opts)
                                    # Update correct index to new position
                                    q['correct'] = opts.index(correct_answer)
                                    q['options'] = opts
                            except Exception:
                                pass
                except Exception:
                    pass
                
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({
                    'quiz_questions': quiz_data,
                    'quiz_meta': {
                        'count': question_count,
                        'primary_doc': primary_doc,
                    },
                    'answer': '',
                    'answer_markdown': '',
                    'result': {'quiz_questions': quiz_data}
                })
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Quiz generation failed: {exc}")

        if action_id == 'compare_sources':
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for source comparison.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get a proper search question
                search_question = get_search_question(question)
                
                result = pipeline.answer_question(
                    search_question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                citations = result.get('citations', []) if isinstance(result, dict) else []
                if len(citations) < 2:
                    formatted_md = "**Source Comparison**\n\nNeed at least 2 sources to compare. Try broadening your search."
                else:
                    sources_text = "\n\n---\n\n".join([f"**Source {i+1} ({c.get('source', 'Unknown')})**:\n{c.get('text', '')}" for i, c in enumerate(citations[:4])])
                    prompt = (
                        f"Compare and contrast the perspectives from these sources on: {search_question}\n\n"
                        f"Identify:\n- Common themes\n- Disagreements or contradictions\n- Unique insights from each source\n\n"
                        f"{sources_text}"
                    )
                    
                    msg = client.chat.completions.create(
                        model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    comparison = (msg.choices[0].message.content or "").strip()
                    formatted_md = f"**Source Comparison**\n\n{comparison}"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Source comparison failed: {exc}")

        if action_id == 'simplify_explanation':
            # Rephrase in simpler terms without jargon
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for simplification.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get original answer
                result = pipeline.answer_question(
                    question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=agents_enabled,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                answer_text = result.get('answer', '') if isinstance(result, dict) else ''
                prompt = (
                    f"Simplify this explanation for a general audience. Use plain language, avoid jargon, "
                    f"and explain technical terms when necessary. Maintain accuracy while making it accessible:\n\n{answer_text}"
                )
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                simplified = (msg.choices[0].message.content or "").strip()
                formatted_md = f"**Simplified Explanation**\n\n{simplified}"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Simplification failed: {exc}")

        if action_id == 'cite_sources_only':
            # Extract and list only the sources and citations
            # Get a proper search question
            search_question = get_search_question(question)
            
            result = pipeline.answer_question(
                search_question,
                k=top_k,
                history=list(history),
                formula_mode=False,
                agents_enabled=False,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            
            citations = result.get('citations', []) if isinstance(result, dict) else []
            if not citations:
                formatted_md = "**Citations**\n\nNo sources found for this query."
            else:
                # Build citation list
                cite_lines = []
                for i, c in enumerate(citations, 1):
                    source = c.get('source', 'Unknown')
                    text_preview = (c.get('text', '')[:150] + '...') if len(c.get('text', '')) > 150 else c.get('text', '')
                    cite_lines.append(f"{i}. **{source}**\n   > {text_preview}")
                formatted_md = "**Citations**\n\n" + "\n\n".join(cite_lines)
            
            answer_html = _md_to_html(formatted_md)
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})

        if action_id == 'mindmap_outline':
            # Generate hierarchical outline or mind map structure
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for mind map generation.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get context
                result = pipeline.answer_question(
                    question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                answer_text = result.get('answer', '') if isinstance(result, dict) else ''
                prompt = (
                    f"Create a hierarchical mind map outline for this topic. Use markdown format with nested bullets. "
                    f"Include 3-5 main branches, each with 2-4 sub-topics:\n\n{answer_text}"
                )
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                outline = (msg.choices[0].message.content or "").strip()
                formatted_md = f"**Mind Map Outline**\n\n{outline}"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Mind map generation failed: {exc}")

        if action_id == 'followup_questions':
            # Generate relevant follow-up questions
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for follow-up generation.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get context
                result = pipeline.answer_question(
                    question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                answer_text = result.get('answer', '') if isinstance(result, dict) else ''
                prompt = (
                    f"Based on this answer to '{question}', generate 6-8 relevant follow-up questions that would help deepen understanding. "
                    f"Return ONLY a JSON array of strings, each string being one question. No explanations, no markdown, just the JSON array:\n\n{answer_text}"
                )
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                raw_output = (msg.choices[0].message.content or "").strip()
                
                # Parse JSON array of questions
                try:
                    import json
                    # Try to extract JSON array if wrapped in markdown code blocks
                    if '```' in raw_output:
                        import re
                        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_output, re.DOTALL)
                        if json_match:
                            raw_output = json_match.group(1)
                    followup_list = json.loads(raw_output)
                    if not isinstance(followup_list, list):
                        followup_list = []
                except Exception:
                    # Fallback: split by newlines and clean up
                    lines = [ln.strip() for ln in raw_output.split('\n') if ln.strip()]
                    followup_list = []
                    for ln in lines:
                        # Remove numbering like "1.", "2)", etc.
                        cleaned = re.sub(r'^\d+[\.\)]\s*', '', ln).strip()
                        # Remove leading dashes or bullets
                        cleaned = re.sub(r'^[-•*]\s*', '', cleaned).strip()
                        if cleaned and len(cleaned) > 10:
                            followup_list.append(cleaned)
                
                # Return structured response with followup_questions array
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({
                    'followup_questions': followup_list,
                    'answer': '',  # No need for HTML answer
                    'answer_markdown': '',
                    'result': {'followup_questions': followup_list}
                })
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Follow-up generation failed: {exc}")

        if action_id == 'translate_answer':
            # Translate answer to another language
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for translation.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get target language from action_params, default to Spanish
                target_lang = action_params.get('language', 'Spanish')
                
                # Get original answer
                result = pipeline.answer_question(
                    question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=agents_enabled,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                answer_text = result.get('answer', '') if isinstance(result, dict) else ''
                prompt = f"Translate this text to {target_lang}, preserving formatting and technical terms:\n\n{answer_text}"
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                translated = (msg.choices[0].message.content or "").strip()
                formatted_md = f"**Translation ({target_lang})**\n\n{translated}"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Translation failed: {exc}")

        if action_id == 'debug_reasoning_trace':
            # Display detailed reasoning steps and process
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=list(history),
                formula_mode=False,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                strict_docs=strict_docs,
                only_doc=only_doc,
                user_id=str(user.id),
                extra_chunks=attach_chunks,
            )
            
            # Extract trace information
            trace = result.get('trace', {}) if isinstance(result, dict) else {}
            answer_text = result.get('answer', '') if isinstance(result, dict) else ''
            
            # Build detailed reasoning trace
            trace_parts = ["**Reasoning Trace**\n"]
            
            if trace.get('retrieval_strategy'):
                trace_parts.append(f"- **Strategy**: {trace['retrieval_strategy']}")
            if trace.get('chunks_retrieved'):
                trace_parts.append(f"- **Chunks Retrieved**: {trace['chunks_retrieved']}")
            if trace.get('reranker_used'):
                trace_parts.append(f"- **Reranker**: {trace['reranker_used']}")
            if trace.get('web_results'):
                trace_parts.append(f"- **Web Results**: {trace['web_results']}")
            if trace.get('time_sec'):
                trace_parts.append(f"- **Processing Time**: {trace['time_sec']:.2f}s")
            
            citations = result.get('citations', []) if isinstance(result, dict) else []
            if citations:
                trace_parts.append(f"\n**Sources Used** ({len(citations)} total):")
                for i, c in enumerate(citations[:5], 1):
                    trace_parts.append(f"{i}. {c.get('source', 'Unknown')}")
            
            trace_parts.append(f"\n**Final Answer**:\n\n{answer_text}")
            
            formatted_md = "\n".join(trace_parts)
            answer_html = _md_to_html(formatted_md)
            
            try:
                usage_service.record(str(user.id), 'ask_action')
            except Exception:
                pass
            return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': result})

        if action_id == 'detect_gaps':
            # Analyze missing or incomplete information
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for gap detection.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get a proper search question
                search_question = get_search_question(question)
                
                # Get answer and analyze
                result = pipeline.answer_question(
                    search_question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                answer_text = result.get('answer', '') if isinstance(result, dict) else ''
                prompt = (
                    f"Analyze this answer for knowledge gaps. Identify:\n"
                    f"1. Missing information that would make the answer more complete\n"
                    f"2. Areas that need more detail or evidence\n"
                    f"3. Assumptions that should be verified\n"
                    f"4. Related topics not covered\n\n"
                    f"Question: {search_question}\n\nAnswer: {answer_text}"
                )
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                gaps = (msg.choices[0].message.content or "").strip()
                formatted_md = f"**Knowledge Gap Analysis**\n\n{gaps}"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Gap detection failed: {exc}")

        if action_id == 'recommend_new_docs':
            # Suggest additional documents to upload
            try:
                from api.core.config import load_config as _load_cfg
                cfg = _load_cfg() or {}
                key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise HTTPException(status_code=400, detail="OpenAI API key required for recommendations.")
                from openai import OpenAI
                client = OpenAI(api_key=key)
                
                # Get a proper search question
                search_question = get_search_question(question)
                
                # Get context to understand what's missing
                result = pipeline.answer_question(
                    search_question,
                    k=top_k,
                    history=list(history),
                    formula_mode=False,
                    agents_enabled=False,
                    web_enabled=web_enabled,
                    strict_docs=strict_docs,
                    only_doc=only_doc,
                    user_id=str(user.id),
                    extra_chunks=attach_chunks,
                )
                
                answer_text = result.get('answer', '') if isinstance(result, dict) else ''
                citations = result.get('citations', []) if isinstance(result, dict) else []
                available_sources = list(set([c.get('source', '') for c in citations if c.get('source')]))
                
                prompt = (
                    f"Based on this question and the available sources, recommend 5-8 specific types of documents "
                    f"that would help provide a more complete answer. Be specific about document types, topics, or sources.\n\n"
                    f"Question: {search_question}\n\n"
                    f"Available sources: {', '.join(available_sources) if available_sources else 'None'}\n\n"
                    f"Current answer quality: {answer_text[:300]}"
                )
                
                msg = client.chat.completions.create(
                    model=(cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )
                recommendations = (msg.choices[0].message.content or "").strip()
                formatted_md = f"**Recommended Documents**\n\n{recommendations}\n\n_Tip: Upload relevant documents to improve answer quality._"
                
                answer_html = _md_to_html(formatted_md)
                try:
                    usage_service.record(str(user.id), 'ask_action')
                except Exception:
                    pass
                return JSONResponse({'answer': answer_html, 'answer_markdown': formatted_md, 'result': {'answer': formatted_md}})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Document recommendation failed: {exc}")

        raise HTTPException(status_code=400, detail=f"Unknown action_id: {action_id}")
    finally:
        try:
            restore_env()
        except Exception:
            pass
