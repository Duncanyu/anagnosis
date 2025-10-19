"""Shared ingestion and question-answering pipeline utilities."""
from __future__ import annotations

import os
import pathlib
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

from api.services.parse import parse_any_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks, search
from api.services.summarize import (
    summarize_document,
    summarize_all_formulas,
    summarize_batched,
    summarizer_info,
    is_formula_query,
)
from api.services.embed import embedding_info
from api.services import agent
from api.services import memory as mem
from api.services import websearch
from api.core.config import load_config


ProgressFn = Optional[Callable[[str], None]]
PctFn = Optional[Callable[[int], None]]
CancelFn = Optional[Callable[[], bool]]


class PipelineCancelled(RuntimeError):
    """Raised when a pipeline operation is cancelled."""


def _emit(callback: ProgressFn, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass


def _emit_pct(callback: PctFn, value: int) -> None:
    if not callback:
        return
    try:
        callback(value)
    except Exception:
        pass


def _check_cancel(should_cancel: CancelFn) -> None:
    if should_cancel and should_cancel():
        raise PipelineCancelled("Cancelled")


def ingest_documents(
    paths: Sequence[pathlib.Path],
    progress: ProgressFn = None,
    progress_pct: PctFn = None,
    should_cancel: CancelFn = None,
) -> Dict[str, Any]:
    """Ingest documents into the vector index and produce summaries."""

    if not paths:
        return {"details": [], "num_docs": 0, "num_chunks": 0, "doc_summary": ""}

    summaries: List[str] = []
    details: List[Dict[str, Any]] = []
    total_chunks = 0

    info = embedding_info()
    _emit(progress, f"Embedding: {info['backend']} ({info['model']})")
    sumi = summarizer_info()
    _emit(progress, f"Summarizer: {sumi['backend']}")

    for raw in paths:
        _check_cancel(should_cancel)
        path = pathlib.Path(raw)
        _emit(progress, f"Loading {path.name}…")
        _emit_pct(progress_pct, 2)
        data = path.read_bytes()

        _emit(progress, "Parsing document…")
        parsed = parse_any_bytes(path.name, data, progress_cb=lambda msg: _emit(progress, msg))
        _check_cancel(should_cancel)
        _emit_pct(progress_pct, 15)

        ocr_list = parsed.get("ocr_page_numbers") or []
        suspect_list = parsed.get("suspect_pages") or []
        salv_list = parsed.get("salvaged_pages") or []

        if ocr_list:
            head = ", ".join(str(x) for x in ocr_list[:20])
            tail = " ..." if len(ocr_list) > 20 else ""
            _emit(progress, f"OCR pages: [{head}]{tail}")
        else:
            _emit(progress, "OCR pages: []")
        if suspect_list:
            head = ", ".join(str(x) for x in suspect_list[:20])
            tail = " ..." if len(suspect_list) > 20 else ""
            _emit(progress, f"Suspect pages: [{head}]{tail}")
        else:
            _emit(progress, "Suspect pages: []")
        if salv_list:
            head = ", ".join(str(x) for x in salv_list[:20])
            tail = " ..." if len(salv_list) > 20 else ""
            _emit(progress, f"Salvaged pages: [{head}]{tail}")
        else:
            _emit(progress, "Salvaged pages: []")

        for page in parsed["pages"]:
            page["doc_name"] = path.name

        _check_cancel(should_cancel)
        _emit(progress, "Chunking…")
        chunks = chunk_pages(parsed["pages"])
        _check_cancel(should_cancel)
        _emit_pct(progress_pct, 35)

        def embed_cb(done: int, total: int) -> None:
            _check_cancel(should_cancel)
            base = 35
            span = 55
            pct = base + int(span * (done / max(1, total)))
            pct = min(95, max(36, pct))
            _emit_pct(progress_pct, pct)

        _emit(progress, "Embedding and indexing…")
        add_chunks(chunks, progress_cb=embed_cb)
        _check_cancel(should_cancel)
        _emit_pct(progress_pct, 96)

        total_chunks += len(chunks)

        _emit(progress, "Summarizing…")
        _check_cancel(should_cancel)
        docsum = summarize_document(chunks)
        _emit_pct(progress_pct, 100)

        summaries.append(f"## {path.name}\n\n{docsum['summary']}")
        details.append(
            {
                "file": str(path),
                "num_pages": parsed["num_pages"],
                "ocr_pages": parsed["ocr_pages"],
                "ocr_page_numbers": ocr_list,
                "suspect_pages": suspect_list,
                "num_chunks": len(chunks),
            }
        )

    combined = "\n\n".join(summaries) if summaries else "No documents ingested."
    return {
        "details": details,
        "num_docs": len(paths),
        "num_chunks": total_chunks,
        "doc_summary": combined,
    }


def _token_overlap(q_text: str, chunk_text: str) -> float:
    qtoks = [t for t in re.findall(r"[A-Za-z0-9_]+", (q_text or "").lower()) if len(t) > 2]
    if not qtoks:
        return 0.0
    ctoks = [t for t in re.findall(r"[A-Za-z0-9_]+", (chunk_text or "").lower()) if len(t) > 2]
    if not ctoks:
        return 0.0
    cset = Counter(ctoks)
    shared = sum(1 for t in qtoks if cset.get(t, 0) > 0)
    return shared / max(1, len(qtoks))


def answer_question(
    question: str,
    *,
    k: int = 5,
    history: Optional[Sequence[Dict[str, str]]] = None,
    formula_mode: bool = False,
    agents_enabled: bool = False,
    web_enabled: bool = False,
    progress: ProgressFn = None,
    progress_pct: PctFn = None,
    should_cancel: CancelFn = None,
    only_doc: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the question-answering pipeline synchronously."""

    _check_cancel(should_cancel)

    base_prefix = (
        "Write normal text and structure in Markdown. "
        "Typeset ALL mathematical expressions in LaTeX: use $...$ for inline and $$...$$ for display. "
        "Do NOT wrap plain prose in \\text{...}. "
        "Use standard LaTeX commands (\\frac, \\sqrt, ^, _). "
        "Preserve citations as plain text.\n"
    )
    if formula_mode:
        fmt_prefix = (
            base_prefix
            + "\nWhen listing formulas, format EACH item on ONE line exactly as: "
            + "<label/meaning> — $$ <LaTeX formula> $$ [FileName.pdf p.N] — <1–2 sentence explanation>. "
            + "Put the citation AFTER the formula (not inside the math). "
            + "Do not use bullets. Do not place citations or explanation inside $...$ or $$...$$. Keep explanations concise and factual.\n\n"
        )
    else:
        fmt_prefix = base_prefix + "\n"
    fmt_q = fmt_prefix + question

    history = list(history or [])
    q_lower = (question or "").strip().lower()
    os.environ["ASK_FORMULA_MODE"] = "true" if formula_mode else "false"
    if history:
        last = history[-1]
        if re.search(r"what\s+(?:was|is)\s+(?:my|the)\s+(?:last|previous)\s+question", q_lower):
            prev_q = last.get("q") or "(unknown)"
            text = f"Your previous question was:\n\n> {prev_q}"
            return {"answer": text, "citations": [], "quotes": []}
        if re.search(r"what\s+(?:was|is)\s+(?:your|the)\s+(?:last|previous)\s+answer", q_lower) or re.search(
            r"repeat\s+your\s+(?:last|previous)\s+answer", q_lower
        ):
            prev_a = last.get("a") or "(no recent answer recorded)"
            text = "Here is my previous answer:\n\n" + prev_a
            return {"answer": text, "citations": [], "quotes": []}

    tb_base = int(os.environ.get("ASK_TIME_BUDGET_SEC", "120"))
    tb = int(os.environ.get("ASK_TIME_BUDGET_SEC_FORMULA", "240")) if formula_mode else tb_base
    web_chunks: List[Dict[str, Any]] = []
    web_hits: List[Any] = []
    max_web_overlap = 0.0
    provider_cfg: Dict[str, Any] = {}
    try:
        provider_cfg = load_config() or {}
    except Exception:
        provider_cfg = {}
    provider_label = (
        provider_cfg.get("WEB_SEARCH_PROVIDER")
        or os.environ.get("WEB_SEARCH_PROVIDER")
        or "duckduckgo"
    )

    if web_enabled:
        _emit(progress, f"Web search ({provider_label})…")
        try:
            web_results = websearch.search_web(question, max_results=6)
        except Exception:
            web_results = []
        if web_results:
            _emit(progress, f"Web results: {len(web_results)}")
        else:
            _emit(progress, "Web search returned no results.")
        from urllib.parse import urlparse

        for res in web_results:
            snippet = (res.get("snippet") or "").strip()
            title = (res.get("title") or "").strip()
            if not snippet and not title:
                continue
            text = (title + "\n" + snippet).strip()
            url = res.get("url") or ""
            host = urlparse(url).netloc or (res.get("source") or "web")
            chunk = {
                "text": text,
                "doc_name": f"Web:{host}",
                "page_start": 1,
                "page_end": 1,
                "section_tag": "web",
                "web_url": url,
                "is_web": True,
                "_score": 0.85,
            }
            max_web_overlap = max(max_web_overlap, _token_overlap(question, text))
            web_chunks.append(chunk)
            web_hits.append((0.85, chunk))

    doc_reference = bool(re.search(r"(textbook|chapter|section|lecture|notes|\.(pdf|docx))", q_lower))

    rel_score, rel_meta = agent.estimate_relevance(question)
    relevance_threshold = float(os.getenv("ASK_RAG_MIN_RELEVANCE", "0.20"))
    _emit(progress, f"Relevance score: {rel_score if rel_score is not None else 'n/a'}")

    # Default: always use local RAG when web search is OFF. This avoids
    # over-pruning questions like "What is calculus" that are general but
    # still benefit from local textbooks.
    use_rag = not web_enabled
    if web_enabled:
        if not web_chunks:
            use_rag = True
        elif doc_reference:
            use_rag = True
        else:
            overlap_threshold = float(os.getenv("ASK_WEB_MIN_OVERLAP", "0.45"))
            use_rag = max_web_overlap < overlap_threshold
    # Do not disable RAG purely based on the heuristic relevance score when
    # web search is off; keep RAG on to maximize useful local hits.
    if web_enabled and rel_score is not None and rel_score < relevance_threshold and not doc_reference:
        use_rag = False

    hits: List[Any] = []
    rag_chunks: List[Dict[str, Any]] = []
    mem_chunks: List[Dict[str, Any]] = []
    rag_scores: List[float] = []

    if use_rag:
        base_pool = int(os.environ.get("ASK_CANDIDATES", "300"))
        pool = int(os.environ.get("ASK_CANDIDATES_FORMULA", "3000")) if formula_mode else base_pool
        pc_holder = {"value": 12}

        def s_cb(msg: str) -> None:
            _emit(progress, msg)
            pc_holder["value"] = min(55, pc_holder["value"] + 3)
            _emit_pct(progress_pct, pc_holder["value"])

        st_base = max(10, min(tb_base // 2, 30))
        st = int(os.environ.get("SEARCH_TIMEOUT_SEC", str(st_base)))
        if formula_mode:
            st = int(os.environ.get("SEARCH_TIMEOUT_SEC_FORMULA", str(max(20, min(tb // 2, 60)))))

        _check_cancel(should_cancel)
        _emit(progress, "Searching index…")
        hits = search(question, k=pool, progress_cb=s_cb, timeout_sec=st, pool=pool, only_doc=only_doc)
        if only_doc and not hits:
            _emit(progress, "No hits in selected document; expanding to all documents…")
            hits = search(question, k=pool, progress_cb=s_cb, timeout_sec=st, pool=pool)
        _check_cancel(should_cancel)
        _emit(progress, f"Hits: {len(hits)}")
        rag_scores = [float(h[0]) for h in hits]
        for sc, ch in hits[:k]:
            x = dict(ch)
            x["_score"] = float(sc)
            rag_chunks.append(x)
        if not hits and not web_chunks:
            _emit_pct(progress_pct, 100)
            return {"answer": "**No local results. Try web search.**", "citations": [], "quotes": []}
    else:
        _emit(progress, "Using web results only…")
        if not web_chunks:
            _emit_pct(progress_pct, 100)
            return {"answer": "**No web results found for this question.**", "citations": [], "quotes": []}

    rag_threshold = float(os.getenv("ASK_WEB_MIN_RAG", "0.35"))

    if web_enabled and web_chunks:
        if use_rag and rag_scores and rag_scores[0] >= rag_threshold:
            top_chunks = list(web_chunks) + rag_chunks
        elif use_rag and rag_chunks:
            top_chunks = list(web_chunks) + rag_chunks
        else:
            top_chunks = list(web_chunks)
    else:
        top_chunks = rag_chunks

    # In-session memory (recent turns from current chat) — keep a longer window
    if history:
        for idx, turn in enumerate(history[-16:], 1):
            q_prev = (turn.get("q") or "").strip()
            a_prev = (turn.get("a") or "").strip()
            if not q_prev and not a_prev:
                continue
            parts = []
            if q_prev:
                parts.append(f"Prev question: {q_prev}")
            if a_prev:
                parts.append(f"Prev answer: {a_prev}")
            text_mem = "\n".join(parts)
            if not text_mem:
                continue
            mem_chunks.append(
                {
                    "text": text_mem,
                    "doc_name": "Conversation memory",
                    "page_start": idx,
                    "page_end": idx,
                    "section_tag": "memory",
                    "is_memory": True,
                    "_score": 0.55,
                }
            )

    # Cross-conversation memory from memory.jsonl (always enabled)
    try:
        limit = int(os.getenv("MEMORY_TOKEN_LIMIT", "1200"))
        recent = mem.load_recent(limit_tokens=limit) or []
        # Deduplicate against current in-session history
        hist_set = {
            ((t.get("q") or "").strip(), (t.get("a") or "").strip())
            for t in (history or [])
        }
        cross = []
        for t in recent:
            pair = ((t.get("q") or "").strip(), (t.get("a") or "").strip())
            if not pair[0] and not pair[1]:
                continue
            if pair in hist_set:
                continue
            cross.append(t)
        # Keep a modest number to avoid flooding context
        for j, turn in enumerate(cross[-8:], 1):
            q_prev = (turn.get("q") or "").strip()
            a_prev = (turn.get("a") or "").strip()
            parts = []
            if q_prev:
                parts.append(f"Prev question: {q_prev}")
            if a_prev:
                parts.append(f"Prev answer: {a_prev}")
            text_mem = "\n".join(parts)
            if not text_mem:
                continue
            mem_chunks.append(
                {
                    "text": text_mem,
                    "doc_name": "Long‑term memory",
                    "page_start": j,
                    "page_end": j,
                    "section_tag": "memory",
                    "is_memory": True,
                    "_score": 0.50,
                }
            )
    except Exception:
        # Memory is best-effort; ignore failures
        pass
    if mem_chunks:
        top_chunks.extend(mem_chunks)

    if web_enabled and web_chunks and not use_rag:
        combined_hits = list(web_hits) if web_hits else [(0.0, c) for c in web_chunks]
    else:
        base_hits = list(hits)
        web_extras = web_hits if web_hits else [(0.85, c) for c in web_chunks]
        combined_hits = base_hits + web_extras
    if mem_chunks:
        combined_hits.extend([(0.55, ch) for ch in mem_chunks])

    if formula_mode:
        _check_cancel(should_cancel)
        _emit(progress, "Formula mode: extracting all formulas in scope…")
        total = len(top_chunks)
        if total < (len(hits) if use_rag else len(web_chunks)):
            _emit(progress, f"Warning: only {total} chunks returned; consider increasing index size or candidate pool.")

        def p_cb(msg: str) -> None:
            _check_cancel(should_cancel)
            _emit(progress, msg)
            m = re.search(r"(\d+)\s*/\s*(\d+)", msg)
            if m:
                done = int(m.group(1))
                pct = 20 + int(70 * (done / max(1, total)))
                pct = max(20, min(95, pct))
                _emit_pct(progress_pct, pct)

        out = summarize_all_formulas(fmt_q, top_chunks, progress_cb=p_cb)
        if agents_enabled:
            _check_cancel(should_cancel)
            try:
                _emit(progress, "Agents: verifying…")
                agent_budget = int(os.environ.get("AGENT_VERIFY_BUDGET", "30"))
                _prev = out if isinstance(out, dict) else {"answer": str(out)}
                agent_hits = [
                    h[1] if isinstance(h, tuple) and len(h) > 1 else h
                    for h in (combined_hits or [])
                ]
                _res = agent.verify_answer(question, out, agent_hits, time_budget_sec=agent_budget)
                changed = False
                try:
                    changed = (_res.get("answer", "") != _prev.get("answer", "")) if isinstance(_res, dict) else False
                except Exception:
                    changed = False
                if isinstance(_res, dict):
                    meta = _res.get("agent_meta", {})
                    meta.update({"enabled": True, "changed": bool(changed)})
                    _res["agent_meta"] = meta
                out = _res
                _emit(progress, "Agents: verdict — " + ("modified" if changed else "validated"))
            except Exception:
                _emit(progress, "Agent verification skipped (error).")
            _check_cancel(should_cancel)
    else:
        _check_cancel(should_cancel)
        _emit(progress, "Summarizing with context…")
        _emit_pct(progress_pct, 60)
        mb = int(os.environ.get("ASK_MAX_BATCHES", "6"))
        exhaustive = os.environ.get("ASK_EXHAUSTIVE", "false").lower() in {"1", "true", "yes", "on"}
        chs = list(top_chunks)
        _check_cancel(should_cancel)
        out = summarize_batched(
            fmt_q,
            chs,
            history=history,
            progress_cb=lambda s: _emit(progress, s),
            max_batches=mb,
            time_budget_sec=tb,
            exhaustive=exhaustive,
        )
        if agents_enabled:
            _check_cancel(should_cancel)
            try:
                _emit(progress, "Agents: verifying…")
                agent_budget = int(os.environ.get("AGENT_VERIFY_BUDGET", "30"))
                _prev = out if isinstance(out, dict) else {"answer": str(out)}
                agent_hits = [
                    h[1] if isinstance(h, tuple) and len(h) > 1 else h
                    for h in (combined_hits or [])
                ]
                _res = agent.verify_answer(question, out, agent_hits, time_budget_sec=agent_budget)
                changed = False
                try:
                    changed = (_res.get("answer", "") != _prev.get("answer", "")) if isinstance(_res, dict) else False
                except Exception:
                    changed = False
                if isinstance(_res, dict):
                    meta = _res.get("agent_meta", {})
                    meta.update({"enabled": True, "changed": bool(changed)})
                    _res["agent_meta"] = meta
                out = _res
                _emit(progress, "Agents: verdict — " + ("modified" if changed else "validated"))
            except Exception:
                _emit(progress, "Agent verification skipped (error).")
            _check_cancel(should_cancel)

    try:
        if isinstance(out, dict) and "citations" in out:
            out["citations"] = sorted(set(out.get("citations", [])))
    except Exception:
        pass

    _emit_pct(progress_pct, 100)

    if isinstance(out, dict) and web_enabled and web_chunks:
        try:
            rep = out.get("agent_report") or ""
            if rep and "Web evidence" not in rep:
                links: List[str] = []
                for ch in web_chunks[:3]:
                    url = ch.get("web_url")
                    if url:
                        links.append(f"- {url}")
                if links:
                    extra = "\n\n**Web sources**\n" + "\n".join(links)
                    out["answer"] = (out.get("answer") or "") + extra
        except Exception:
            pass

    if isinstance(out, dict):
        return out
    return {"answer": str(out), "citations": [], "quotes": []}
