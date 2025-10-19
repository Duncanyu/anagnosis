"""FastAPI web interface for the Anagnosis pipeline."""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Sequence

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import markdown as mdlib
except Exception:  # pragma: no cover
    mdlib = None

from api.core.config import load_config, present_keys, save_secret
from api.services import pipeline, index as index_service
from api.services import memory as mem
from api.services.pipeline import PipelineCancelled
from api.core.config import load_config

PREFS_PATH = pathlib.Path("artifacts") / "ui_prefs.json"
DOC_SUMMARY_PATH = pathlib.Path("artifacts") / "doc_summaries.jsonl"
LIBRARY_INDEX_PATH = pathlib.Path("artifacts") / "library_index.json"

app = FastAPI(title="Anagnosis Web Service")
templates = Jinja2Templates(directory=str(ROOT / "web" / "templates"))
# Serve static assets (CSS/JS) and the project icon located at ROOT/assets
app.mount("/static", StaticFiles(directory=str(ROOT / "web" / "static")), name="static")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")


# ---------------------------------------------------------------------------
# Preference helpers


def read_prefs() -> Dict[str, Any]:
    try:
        if PREFS_PATH.exists():
            return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_prefs(data: Dict[str, Any]) -> None:
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _bool(val: Any) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _apply_pref_env(prefs: Dict[str, Any]) -> None:
    bool_map = {
        "MEMORY_ENABLED": prefs.get("MEMORY_ENABLED", os.environ.get("MEMORY_ENABLED", "false")),
        "ASK_EXHAUSTIVE": prefs.get("ASK_EXHAUSTIVE", os.environ.get("ASK_EXHAUSTIVE", "false")),
    }
    for key, val in bool_map.items():
        os.environ[key] = "true" if _bool(val) else "false"

    numeric_defaults = {
        "MEMORY_TOKEN_LIMIT": 1200,
        "MEMORY_FILE_LIMIT_MB": 50,
        "ASK_TIME_BUDGET_SEC": 120,
        "ASK_MAX_BATCHES": 6,
        "ASK_BATCH_CHAR_BUDGET": 12000,
        "ASK_CANDIDATES": 300,
    }
    for key, default in numeric_defaults.items():
        value = prefs.get(key) or os.environ.get(key) or default
        os.environ[key] = str(value)

    reranker = prefs.get("ASK_RERANKER") or os.environ.get("ASK_RERANKER", "off")
    os.environ["ASK_RERANKER"] = str(reranker)
    os.environ.setdefault("ASK_AGENTS", "false")


def _settings_defaults() -> Dict[str, Any]:
    cfg = {}
    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    prefs = read_prefs()
    env = os.environ
    return {
        "OPENAI_API_KEY": cfg.get("OPENAI_API_KEY", ""),
        "HF_TOKEN": cfg.get("HF_TOKEN", ""),
        "SERPAPI_KEY": cfg.get("SERPAPI_KEY", ""),
        "BRAVE_API_KEY": cfg.get("BRAVE_API_KEY") or cfg.get("BRAVE_SEARCH_KEY", ""),
        "OPENAI_CHAT_MODEL": cfg.get("OPENAI_CHAT_MODEL") or env.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "HF_LLM_NAME": cfg.get("HF_LLM_NAME") or env.get("HF_LLM_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "EMBED_BACKEND": cfg.get("EMBED_BACKEND", "hf"),
        "LLM_BACKEND": cfg.get("LLM_BACKEND", "openai"),
        "MEMORY_ENABLED": _bool(prefs.get("MEMORY_ENABLED") or env.get("MEMORY_ENABLED", "false")),
        "MEMORY_TOKEN_LIMIT": int(prefs.get("MEMORY_TOKEN_LIMIT") or env.get("MEMORY_TOKEN_LIMIT", "1200")),
        "MEMORY_FILE_LIMIT_MB": int(prefs.get("MEMORY_FILE_LIMIT_MB") or env.get("MEMORY_FILE_LIMIT_MB", "50")),
        "OPENAI_TPM": int(prefs.get("OPENAI_TPM") or env.get("OPENAI_TPM", "0")),
        "OPENAI_RPM": int(prefs.get("OPENAI_RPM") or env.get("OPENAI_RPM", "0")),
        "ASK_BATCH_CHAR_BUDGET": int(prefs.get("ASK_BATCH_CHAR_BUDGET") or env.get("ASK_BATCH_CHAR_BUDGET", "12000")),
        "ASK_MAX_BATCHES": int(prefs.get("ASK_MAX_BATCHES") or env.get("ASK_MAX_BATCHES", "6")),
        "ASK_TIME_BUDGET_SEC": int(prefs.get("ASK_TIME_BUDGET_SEC") or env.get("ASK_TIME_BUDGET_SEC", "120")),
        "ASK_EXHAUSTIVE": _bool(prefs.get("ASK_EXHAUSTIVE") or env.get("ASK_EXHAUSTIVE", "false")),
        "ASK_RERANKER": (prefs.get("ASK_RERANKER") or env.get("ASK_RERANKER", "off")).lower(),
        "ASK_CANDIDATES": int(prefs.get("ASK_CANDIDATES") or env.get("ASK_CANDIDATES", "300")),
    }


def _save_settings(payload: Dict[str, Any]) -> str:
    try:
        if payload.get("openai_key", "").strip():
            save_secret("OPENAI_API_KEY", payload["openai_key"].strip(), prefer="keyring")
        if payload.get("hf_token", "").strip():
            save_secret("HF_TOKEN", payload["hf_token"].strip(), prefer="keyring")
        if payload.get("serp_key", "").strip():
            save_secret("SERPAPI_KEY", payload["serp_key"].strip(), prefer="file")
        if payload.get("brave_key", "").strip():
            save_secret("BRAVE_API_KEY", payload["brave_key"].strip(), prefer="file")
        save_secret("OPENAI_CHAT_MODEL", payload.get("openai_model", "gpt-4o-mini").strip() or "gpt-4o-mini", prefer="file")
        save_secret("HF_LLM_NAME", payload.get("hf_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip() or "TinyLlama/TinyLlama-1.1B-Chat-v1.0", prefer="file")
        save_secret("EMBED_BACKEND", (payload.get("embed_backend") or "hf").lower(), prefer="file")
        save_secret("LLM_BACKEND", (payload.get("llm_backend") or "openai").lower(), prefer="file")

        prefs = read_prefs()
        prefs.update(
            {
                "MEMORY_ENABLED": "true" if payload.get("memory_enabled") else "false",
                "MEMORY_TOKEN_LIMIT": int(payload.get("memory_tokens", 1200)),
                "MEMORY_FILE_LIMIT_MB": int(payload.get("memory_file_mb", 50)),
                "OPENAI_TPM": int(payload.get("openai_tpm", 0)),
                "OPENAI_RPM": int(payload.get("openai_rpm", 0)),
                "ASK_BATCH_CHAR_BUDGET": int(payload.get("ask_char_budget", 12000)),
                "ASK_MAX_BATCHES": int(payload.get("ask_max_batches", 6)),
                "ASK_TIME_BUDGET_SEC": int(payload.get("ask_time_budget", 120)),
                "ASK_EXHAUSTIVE": "true" if payload.get("ask_exhaustive") else "false",
                "ASK_RERANKER": (payload.get("ask_reranker") or "off").lower(),
                "ASK_CANDIDATES": int(payload.get("ask_candidates", 300)),
            }
        )
        write_prefs(prefs)
        _apply_pref_env(prefs)
        os.environ["OPENAI_CHAT_MODEL"] = payload.get("openai_model", "gpt-4o-mini").strip() or "gpt-4o-mini"
        os.environ["HF_LLM_NAME"] = payload.get("hf_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip() or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        return "Settings saved."
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


PREF_STORE = read_prefs()
_apply_pref_env(PREF_STORE)


# ---------------------------------------------------------------------------
# Formatting helpers


def _md_to_html(md_text: str) -> str:
    if mdlib:
        return mdlib.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    return f"<pre>{html.escape(md_text)}</pre>"


def _format_log_lines(lines: Sequence[str], title: str) -> str:
    if not lines:
        return f"{title}: idle."
    return "\n".join(lines)


def _format_ingest_details(details: Sequence[Dict[str, Any]]) -> str:
    if not details:
        return "No documents ingested yet."
    lines = ["| File | Pages | OCR Pages | Chunks | Suspect Pages |", "| --- | --- | --- | --- | --- |"]
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
        lines.append(f"| {name} | {num_pages} | {ocr_preview} | {num_chunks} | {suspect_preview} |")
    return "\n".join(lines)


def _format_quotes(quotes: Sequence[Dict[str, Any]]) -> str:
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


# ---------------------------------------------------------------------------
# Ingestion job management


INGEST_JOBS: Dict[str, Dict[str, Any]] = {}
INGEST_LOCK = threading.Lock()


def _append_log(job_id: str, message: str) -> None:
    with INGEST_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is not None:
            job.setdefault("logs", []).append(message)


def _start_ingest_job(paths: Sequence[pathlib.Path], temp_dir: pathlib.Path, file_names: Sequence[str]) -> str:
    job_id = uuid.uuid4().hex
    job_info = {
        "status": "running",
        "logs": ["Files queued:"] + [f"- {name}" for name in file_names] + ["Starting ingestion..."],
        "progress": 0,
        "documents": [],
        "summary_html": "",
        "details_html": "",
        "error": None,
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
            def log(msg: str) -> None:
                nonlocal doc_state
                if msg.startswith("Loading "):
                    name = msg.split("Loading ", 1)[1].strip("…")
                    idx = name_lookup.get(name)
                    if idx is None:
                        idx = min(doc_state["index"] + 1, doc_count - 1)
                        name_lookup[name] = idx
                    doc_state = {"index": idx, "name": name, "pages_total": 0, "pages_done": 0}
                    with INGEST_LOCK:
                        job = INGEST_JOBS.get(job_id)
                        if job is not None:
                            docs = job.setdefault("documents", [])
                            while len(docs) <= idx:
                                docs.append({"name": name, "pages": 0, "pages_total": 0, "pages_done": 0})
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
                                    docs.append({"name": doc_state["name"], "pages": 0, "pages_total": 0, "pages_done": 0})
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
                            doc_state["pages_total"] = max(doc_state.get("pages_total", total), total)
                            doc_state["pages_done"] = done
                            if doc_state["pages_total"] > 0:
                                ratio = doc_state["pages_done"] / doc_state["pages_total"]
                                update_overall(0.5 * ratio)
                            with INGEST_LOCK:
                                job = INGEST_JOBS.get(job_id)
                                if job is not None:
                                    docs = job.setdefault("documents", [])
                                    while len(docs) <= doc_state["index"]:
                                        docs.append({"name": doc_state["name"], "pages": 0, "pages_total": 0, "pages_done": 0})
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

            result = pipeline.ingest_documents(paths, progress=log, progress_pct=update_pct)
            summary_md = result.get("doc_summary") or "No documents ingested."
            details_md = _format_ingest_details(result.get("details", []))
            try:
                _update_library_catalog(result.get("details", []), summary_md)
            except Exception:
                pass
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
                    job["error"] = "Cancelled"
                    job["progress"] = job.get("progress", 0)
                    job.setdefault("logs", []).append("Ingestion cancelled.")
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

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _ingest_job_payload(job_id: str) -> Dict[str, Any]:
    with INGEST_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is None:
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


# ---------------------------------------------------------------------------
# Title generation (LLM-backed with heuristic fallback)


def _generate_chat_title(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "New chat"
    # Prefer OpenAI if configured
    try:
        cfg = load_config() or {}
        key = cfg.get("OPENAI_API_KEY")
        model = cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        if key:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            sys = (
                "Return ONLY a short, descriptive chat title in Title Case. "
                "3–6 words. No surrounding quotes, no trailing period."
            )
            prompt = text[:1200]
            msg = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            )
            title = (msg.choices[0].message.content or "").strip().strip("\"'")
            title = re.sub(r"[\r\n].*", "", title)
            return title[:72] or "New chat"
    except Exception:
        pass

    # Heuristic fallback: first informative slice
    t = re.sub(r"[#*_`>\-]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:72] or "New chat").strip()


# Note: Chat title generation is now provided by the API server


# ---------------------------------------------------------------------------
# Pipeline wrappers

def _answer_question(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question required.")

    history = payload.get("history") or []
    # Always enable in-session memory; limits are controlled solely by settings
    formula_mode = bool(payload.get("formula_mode"))
    agents_enabled = bool(payload.get("agents_enabled"))
    web_enabled = bool(payload.get("web_enabled"))
    top_k = int(payload.get("top_k") or 10)
    only_doc = (payload.get("only_doc") or "").strip() or None

    logs: List[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    history_payload = history
    try:
        result = pipeline.answer_question(
            question,
            k=top_k,
            history=history_payload,
            formula_mode=formula_mode,
            agents_enabled=agents_enabled,
            web_enabled=web_enabled,
            progress=log,
            only_doc=only_doc,
        )
    except PipelineCancelled:
        raise HTTPException(status_code=409, detail="Answering cancelled.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    formatted_md = _format_answer(question, result, agents_enabled)
    answer_html = _md_to_html(formatted_md)

    # Persist conversational memory (best-effort)
    try:
        if question and isinstance(result, dict):
            raw_answer = (result.get("answer") or "").strip()
            if raw_answer:
                mem.append_turn(question, raw_answer)
                lim_mb = int(os.environ.get("MEMORY_FILE_LIMIT_MB", "50"))
                mem.prune_file(lim_mb)
    except Exception:
        pass
    return {
        "answer": answer_html,
        "answer_markdown": formatted_md,
        "log": _format_log_lines(logs, "Ask log"),
    }


# ---------------------------------------------------------------------------
# Library helpers


def _summary_doc_name(summary_text: str) -> str | None:
    if not summary_text:
        return None
    match = re.search(r"^##\s+(.+)$", summary_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    match = re.search(r"#\s+Summary\s+of\s+(.+)$", summary_text.strip().splitlines()[0]) if summary_text.strip() else None
    if match:
        return match.group(1).strip()
    return None


def _load_library_summaries() -> Dict[str, Dict[str, Any]]:
    """Load latest per-document summaries from the JSONL log.

    The summary file stores a combined Markdown blob per ingestion run, where
    each document's section is prefixed by a top-level heading `## <doc_name>`.
    We split that combined blob so the mapping we return is per ingested PDF
    (keyed by the normalized file name), not by internal section headers like
    "High-level overview".
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not DOC_SUMMARY_PATH.exists():
        return results
    try:
        with DOC_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                blob = rec.get("summary", "")
                ts = rec.get("ts")
                parts = _extract_doc_summaries(blob)
                for raw_name, text in parts.items():
                    name = _normalize_doc_name(raw_name)
                    key = _catalog_key(name)
                    prev = results.get(key)
                    # Keep the latest summary by timestamp
                    if prev is None or (ts or 0) >= (prev.get("ts") or 0):
                        results[key] = {"summary": text, "ts": ts}
    except Exception:
        pass
    return results


def _normalize_doc_name(name: str) -> str:
    return pathlib.Path(name or "").name


def _catalog_key(name: str) -> str:
    return _normalize_doc_name(name).lower()


def _load_library_catalog() -> Dict[str, Dict[str, Any]]:
    if not LIBRARY_INDEX_PATH.exists():
        return {}
    try:
        data = json.loads(LIBRARY_INDEX_PATH.read_text(encoding="utf-8") or "[]")
        if isinstance(data, dict):
            data = list(data.values())
        catalog: Dict[str, Dict[str, Any]] = {}
        for entry in data or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("doc_name")
            if not name:
                continue
            entry = dict(entry)
            entry.setdefault("display_name", pathlib.Path(name).stem)
            entry.setdefault("chunks", 0)
            entry.setdefault("pages", None)
            entry.setdefault("summary", "")
            entry.setdefault("ingested_at", None)
            catalog[_catalog_key(name)] = entry
        return catalog
    except Exception:
        return {}


def _save_library_catalog(catalog: Dict[str, Dict[str, Any]]) -> None:
    try:
        LIBRARY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(catalog.values(), key=lambda r: (-(r.get("ingested_at") or 0), r.get("name", "")))
        LIBRARY_INDEX_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_doc_summaries(summary_text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    current: str | None = None
    buffer: List[str] = []
    for line in (summary_text or "").splitlines():
        if line.startswith("## "):
            if current is not None:
                mapping[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if current is not None:
        mapping[current] = "\n".join(buffer).strip()
    return mapping


def _scan_chunk_stats() -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    try:
        _, _, chunk_path, _, _, _ = index_service._active_paths()
    except Exception:
        chunk_path = None
    if not chunk_path or not chunk_path.exists():
        return stats
    try:
        with chunk_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = _normalize_doc_name(row.get("doc_name") or "Unknown.pdf")
                key = _catalog_key(name)
                entry = stats.setdefault(
                    key,
                    {
                        "name": name,
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
    except Exception:
        pass
    for info in stats.values():
        info["pages"] = len(info.pop("_pages", set())) or None
    return stats


def _update_library_catalog(details: Sequence[Dict[str, Any]], summary_text: str) -> None:
    if not details:
        return
    catalog = _load_library_catalog()
    summaries_raw = _extract_doc_summaries(summary_text)
    summaries = { _normalize_doc_name(name): text for name, text in summaries_raw.items() }
    now = time.time()
    for detail in details:
        name = _normalize_doc_name(detail.get("file"))
        if not name:
            continue
        key = _catalog_key(name)
        entry = catalog.get(key, {})
        entry["name"] = name
        entry["display_name"] = pathlib.Path(name).stem
        if detail.get("num_pages") is not None:
            entry["pages"] = detail.get("num_pages")
        entry["chunks"] = detail.get("num_chunks") if detail.get("num_chunks") is not None else entry.get("chunks", 0)
        entry["summary"] = summaries.get(name, entry.get("summary", ""))
        entry["ingested_at"] = now
        catalog[key] = entry
    _save_library_catalog(catalog)


def _remove_from_catalog(names: Sequence[str]) -> None:
    if not names:
        return
    catalog = _load_library_catalog()
    changed = False
    for name in names:
        key = _catalog_key(name)
        if key in catalog:
            catalog.pop(key, None)
            changed = True
    if changed:
        _save_library_catalog(catalog)


def _library_inventory() -> List[Dict[str, Any]]:
    catalog = _load_library_catalog()
    # Merge summaries onto existing catalog entries only; do not create new
    # records from headings like "High-level overview".
    summaries = _load_library_summaries()
    for key, info in summaries.items():
        entry = catalog.get(key)
        if entry is None:
            continue
        if not entry.get("summary"):
            entry["summary"] = info.get("summary", "")
        entry.setdefault("ingested_at", info.get("ts"))

    stats = _scan_chunk_stats()
    # Merge stats into catalog
    for key, info in stats.items():
        entry = catalog.get(key)
        if entry is None:
            catalog[key] = {
                "name": info["name"],
                "display_name": pathlib.Path(info["name"]).stem,
                "chunks": info.get("chunks", 0),
                "pages": info.get("pages"),
                "summary": "",
                "ingested_at": None,
            }
        else:
            entry["chunks"] = info.get("chunks", entry.get("chunks", 0))
            if not entry.get("pages"):
                entry["pages"] = info.get("pages")
    records = list(catalog.values())
    records.sort(key=lambda r: (-(r.get("ingested_at") or 0), r.get("display_name", "")))
    return records


def _prune_doc_summaries(remove: Sequence[str]) -> None:
    if not remove or not DOC_SUMMARY_PATH.exists():
        return
    names = {n.lower() for n in remove}
    try:
        lines = DOC_SUMMARY_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    keep: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        doc = _summary_doc_name(data.get("summary", ""))
        if doc and doc.lower() in names:
            continue
        keep.append(json.dumps(data, ensure_ascii=False))
    try:
        DOC_SUMMARY_PATH.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    except Exception:
        pass


def _remove_documents(doc_names: Sequence[str]) -> int:
    try:
        _, _, chunk_path, _, _, _ = index_service._active_paths()
    except Exception:
        chunk_path = None
    names = {_catalog_key(n) for n in doc_names if n}
    if not names:
        return 0
    # Track whether any entry exists in catalog; used as soft success if chunks file missing
    catalog = _load_library_catalog()
    existed = any(k in catalog for k in names)
    rows: List[Dict[str, Any]] = []
    removed = 0
    if chunk_path and chunk_path.exists():
        with chunk_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = _catalog_key(row.get("doc_name"))
                if name in names:
                    removed += 1
                    continue
                rows.append(row)
        if removed > 0:
            index_service.clear_index()
    if rows:
        # Fast path: rebuild index from kept rows without re-embedding
        try:
            index_service.rebuild_index_from_rows(rows)
        except Exception:
            # Fallback to re-embedding if reconstruct path fails
            sanitized: List[Dict[str, Any]] = []
            for row in rows:
                row = dict(row)
                row.pop("rid", None)
                sanitized.append(row)
            index_service.add_chunks(sanitized)
    _prune_doc_summaries(doc_names)
    _remove_from_catalog([_normalize_doc_name(name) for name in doc_names])
    return removed if removed > 0 else (1 if existed else 0)


# ---------------------------------------------------------------------------
# Ask job management (for live logging)

ASK_JOBS: Dict[str, Dict[str, Any]] = {}
ASK_LOCK = threading.Lock()


def _append_ask_log(job_id: str, message: str) -> None:
    with ASK_LOCK:
        job = ASK_JOBS.get(job_id)
        if job is not None:
            job.setdefault("logs", []).append(message)


def _ask_job_payload(job_id: str) -> Dict[str, Any]:
    with ASK_LOCK:
        job = ASK_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ask job not found.")
        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "logs": list(job.get("logs", [])),
            "answer": job.get("answer"),
            "answer_markdown": job.get("answer_markdown"),
            "error": job.get("error"),
        }


def _start_ask_job(payload: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    with ASK_LOCK:
        ASK_JOBS[job_id] = {
            "status": "running",
            "logs": ["Working…"],
            "answer": None,
            "answer_markdown": None,
            "error": None,
        }

    def worker() -> None:
        try:
            # Mirror environment and parameters from _answer_question
            question = (payload.get("question") or "").strip()
            if not question:
                raise HTTPException(status_code=400, detail="Question required.")

            history = payload.get("history") or []
            formula_mode = bool(payload.get("formula_mode"))
            agents_enabled = bool(payload.get("agents_enabled"))
            web_enabled = bool(payload.get("web_enabled"))
            top_k = int(payload.get("top_k") or 10)
            only_doc = (payload.get("only_doc") or "").strip() or None

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
            )

            formatted_md = _format_answer(question, result, agents_enabled)
            answer_html = _md_to_html(formatted_md)
            # Persist conversational memory (best-effort)
            try:
                if question and isinstance(result, dict):
                    raw_answer = (result.get("answer") or "").strip()
                    if raw_answer:
                        mem.append_turn(question, raw_answer)
                        lim_mb = int(os.environ.get("MEMORY_FILE_LIMIT_MB", "50"))
                        mem.prune_file(lim_mb)
            except Exception:
                pass
            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    job["answer"] = answer_html
                    job["answer_markdown"] = formatted_md
                    job["status"] = "done"
                    # Ensure at least one line to indicate completion
                    job.setdefault("logs", []).append("Done.")
        except PipelineCancelled:
            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    job["status"] = "cancelled"
                    job["error"] = "Cancelled"
        except HTTPException as exc:  # re-signal validation errors
            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc.detail)
        except Exception as exc:
            with ASK_LOCK:
                job = ASK_JOBS.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# Auth helpers

async def get_current_user(request: Request):
    """Check if user is authenticated via cookie."""
    try:
        token = request.cookies.get("access_token")
        if not token:
            return None
        # Verify token with the API server
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "http://localhost:8000/api/auth/check",
                cookies={"access_token": token},
                timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("authenticated"):
                    return data.get("user")
        return None
    except Exception as e:
        print(f"Auth check error: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Serve the login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request) -> HTMLResponse:
    """Serve the signup page."""
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Main application page - requires authentication."""
    user = await get_current_user(request)
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/login", status_code=302)
    
    defaults = _settings_defaults()
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request,
            "defaults": defaults,
            "user": user,
        },
    )


# Note: Settings endpoints are now served by the API server


# Note: Ingestion endpoints are now served by the API server


# Note: Ask endpoints are now served by the API server


# Note: Library endpoints are now served by the API server


# Note: Ask job endpoints are now served by the API server


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("web.app:app", host="0.0.0.0", port=port, reload=False)
