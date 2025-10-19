from __future__ import annotations

import os
import re
import threading
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.services import pipeline
from api.services import memory as mem
from api.routes.settings import apply_env_for_user

try:
    import markdown as mdlib
except Exception:
    mdlib = None

router = APIRouter(prefix="/api", tags=["ask"])


def _md_to_html(md_text: str) -> str:
    if mdlib:
        return mdlib.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    return f"<pre>{(md_text or '').replace('<','&lt;').replace('>','&gt;')}</pre>"


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
    cite_block = f"\n\n**Citations:** {', '.join(citations)}" if citations else ""
    quote_block = _format_quotes(quotes)
    agent_block = _format_agent_diag(result if isinstance(result, dict) else {}, agents_enabled)
    body = (answer or "").strip()
    if question and body and not body.lstrip().startswith("#"):
        body = f"### {question}\n\n" + body
    return body + cite_block + quote_block + agent_block


ASK_JOBS: Dict[str, Dict[str, Any]] = {}
ASK_LOCK = threading.Lock()


def _append_ask_log(job_id: str, message: str) -> None:
    with ASK_LOCK:
        job = ASK_JOBS.get(job_id)
        if job is not None:
            job.setdefault("logs", []).append(message)


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
        }


@router.post("/ask/start")
async def api_ask_start(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
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
            web_enabled = bool(payload.get("web_enabled"))
            top_k = int(payload.get("top_k") or 10)
            only_doc = (payload.get("only_doc") or "").strip() or None

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

            history_payload = history
            result = pipeline.answer_question(
                question,
                k=top_k,
                history=history_payload,
                formula_mode=formula_mode,
                agents_enabled=agents_enabled,
                web_enabled=web_enabled,
                progress=log,
                only_doc=only_doc,
                user_id=str(user.id),
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
                    job["status"] = "done"
                    job["answer"] = answer_html
                    job["answer_markdown"] = formatted_md
        except Exception as exc:
            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
        finally:
            try:
                restore_env()
            except Exception:
                pass
        
    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse(_ask_job_payload(job_id, user_id=str(user.id)))


@router.get("/ask/status/{job_id}")
async def api_ask_status(job_id: str, user: User = Depends(require_auth)) -> JSONResponse:
    return JSONResponse(_ask_job_payload(job_id, user_id=str(user.id)))


@router.post("/ask")
async def api_ask(request: Request, user: User = Depends(require_auth)) -> JSONResponse:
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question required.")

    history = payload.get("history") or []
    formula_mode = bool(payload.get("formula_mode"))
    agents_enabled = bool(payload.get("agents_enabled"))
    web_enabled = bool(payload.get("web_enabled"))
    top_k = int(payload.get("top_k") or 10)
    only_doc = (payload.get("only_doc") or "").strip() or None

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

    result = pipeline.answer_question(
        question,
        k=top_k,
        history=history,
        formula_mode=formula_mode,
        agents_enabled=agents_enabled,
        web_enabled=web_enabled,
        only_doc=only_doc,
        user_id=str(user.id),
    )
    formatted_md = _format_answer(question, result, agents_enabled)
    answer_html = _md_to_html(formatted_md)
    return JSONResponse({
        "answer": answer_html,
        "answer_markdown": formatted_md,
        "log": "Ready.",
    })
