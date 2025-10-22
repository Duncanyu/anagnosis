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
    *,
    user_id: Optional[str] = None,
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
        add_chunks(chunks, progress_cb=embed_cb, user_id=user_id)
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
    strict_docs: bool = False,
    progress: ProgressFn = None,
    progress_pct: PctFn = None,
    should_cancel: CancelFn = None,
    only_doc: Optional[str] = None,
    user_id: Optional[str] = None,
    exhaustive: Optional[bool] = None,
    extra_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the question-answering pipeline synchronously."""

    _check_cancel(should_cancel)

    # Keep the original user phrasing for final rendering
    _orig_user_q = question

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

    # Simple retrieval augmentation: if the last turn has a question but no answer
    # (e.g., user cancelled), include it to provide context for pronouns like "his/it".
    try:
        if history:
            _last = history[-1]
            prev_q = (_last.get("q") or "").strip()
            prev_a = (_last.get("a") or "").strip()
            if prev_q and not prev_a:
                question = f"{prev_q}\nFollow-up: {_orig_user_q}"
    except Exception:
        pass

    tb_base = int(os.environ.get("ASK_TIME_BUDGET_SEC", "120"))
    tb = int(os.environ.get("ASK_TIME_BUDGET_SEC_FORMULA", "240")) if formula_mode else tb_base
    web_chunks: List[Dict[str, Any]] = []
    web_hits: List[Any] = []
    max_web_overlap = 0.0
    provider_label = "duckduckgo"
    try:
        from api.services.websearch import get_provider_name
        provider_label = get_provider_name() or "duckduckgo"
    except Exception:
        pass

    if web_enabled:
        # Web search and strict docs are mutually exclusive; allow both sources by disabling strict
        if strict_docs:
            strict_docs = False
            _emit(progress, "Strict documents disabled due to Web search.")
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
    _emit(progress, f"Relevance score: {rel_score if rel_score is not None else 'n/a'}")
    # Decide if we should also include local RAG when web is enabled.
    # Use the same LLM judge logic as strict mode, but as a gate (lenient in that any >0/Relevant passes).
    lenient_th = float(os.getenv("ASK_WEB_LENIENT_MIN", "0.05"))
    include_rag = True if not web_enabled else False
    hits: List[Any] = []
    if web_enabled:
        include_rag = bool(doc_reference or only_doc)
        # Heuristic hint before LLM
        if not include_rag and rel_score is not None:
            include_rag = (rel_score >= lenient_th)
        # Run a light local search to feed the LLM gate
        try:
            gate_pool = int(os.getenv("ASK_WEB_GATE_CANDIDATES", "80"))
            st_gate = int(os.getenv("ASK_WEB_GATE_TIMEOUT", "12"))
            _emit(progress, "Local gate search…")
            hits = search(
                question,
                k=gate_pool,
                progress_cb=lambda s: _emit(progress, s),
                timeout_sec=st_gate,
                pool=gate_pool,
                only_doc=only_doc,
                user_id=user_id,
            )
            gate_max = int(os.getenv("ASK_STRICT_LLM_MAX_HITS", "24"))
            gate_chunks = []
            for sc, ch in hits[:max(1, min(gate_max, len(hits)))] if hits else []:
                x = dict(ch); x["_score"] = float(sc); gate_chunks.append(x)
            if gate_chunks:
                llm_score, meta = agent.llm_relevance_gate(question, gate_chunks, doc_focus=only_doc)
                decided = None
                llm_label = meta.get('label') if isinstance(meta, dict) else None
                if isinstance(llm_label, str):
                    decided = (llm_label.strip().lower() == 'relevant')
                elif llm_score is not None:
                    decided = (float(llm_score) > 0.0)
                if progress:
                    try:
                        detail = ""
                        if isinstance(meta, dict) and meta.get('ctx_snippets') is not None:
                            detail = f" • ctx={meta.get('ctx_snippets')} snippets/{meta.get('ctx_chars')} chars"
                        progress(f"LLM web gate: score={llm_score if llm_score is not None else 'n/a'} label={llm_label or 'n/a'}{detail}")
                    except Exception:
                        pass
                if decided is True:
                    include_rag = True
        except Exception:
            pass

    rag_chunks: List[Dict[str, Any]] = []
    attach_chunks: List[Dict[str, Any]] = list(extra_chunks or [])
    mem_chunks: List[Dict[str, Any]] = []
    rag_scores: List[float] = []

    if (not web_enabled) or include_rag:
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
        if not hits:
            hits = search(
                question,
                k=pool,
                progress_cb=s_cb,
                timeout_sec=st,
                pool=pool,
                only_doc=only_doc,
                user_id=user_id,
            )
        if only_doc and not hits:
            _emit(progress, "No hits in selected document; expanding to all documents…")
            hits = search(
                question,
                k=pool,
                progress_cb=s_cb,
                timeout_sec=st,
                pool=pool,
                user_id=user_id,
            )
        _check_cancel(should_cancel)
        _emit(progress, f"Hits: {len(hits)}")
        rag_scores = [float(h[0]) for h in hits]
        for sc, ch in hits[:k]:
            x = dict(ch)
            x["_score"] = float(sc)
            rag_chunks.append(x)
        if not web_enabled and not hits:
            _emit_pct(progress_pct, 100)
            return {"answer": "**No local results. Try web search.**", "citations": [], "quotes": []}
    else:
        _emit(progress, "Skipping local RAG (lenient gate). Using web results.")
        if web_enabled and not web_chunks:
            _emit(progress, "Web search returned no results; falling back to local RAG…")
            try:
                # run a light search to avoid empty context
                st_base = max(8, min(tb_base // 3, 20))
                st = int(os.environ.get("SEARCH_TIMEOUT_SEC", str(st_base)))
                hits = search(
                    question,
                    k=int(os.environ.get("ASK_CANDIDATES", "150")),
                    progress_cb=lambda s: _emit(progress, s),
                    timeout_sec=st,
                    pool=int(os.environ.get("ASK_CANDIDATES", "150")),
                    only_doc=only_doc,
                    user_id=user_id,
                )
                _emit(progress, f"Fallback Hits: {len(hits)}")
                for sc, ch in hits[:k]:
                    x = dict(ch)
                    x["_score"] = float(sc)
                    rag_chunks.append(x)
            except Exception:
                pass

    rag_threshold = float(os.getenv("ASK_WEB_MIN_RAG", "0.35"))
    strict_answer_min = float(os.getenv("ASK_STRICT_MIN_RAG", "0.35"))

    if strict_docs:
        # Strict mode: use only retrieved document chunks; no web or memory
        top_chunks = rag_chunks
        # Numeric guardrails (fallback when LLM is unavailable)
        overlap_min = float(os.getenv("ASK_STRICT_MIN_OVERLAP", "0.18"))
        max_overlap = 0.0
        for c in top_chunks[:max(1, min(10, len(top_chunks)))]:
            try:
                max_overlap = max(max_overlap, _token_overlap(question, c.get("text") or ""))
            except Exception:
                pass
        # Keyword coverage across chunks
        def _q_tokens(s: str):
            toks = re.findall(r"[a-z0-9]{3,}", (s or '').lower())
            stop = {"what","which","when","where","why","how","does","do","is","are","was","were","can","could","should","would","will","might","may","the","and","for","from","this","that","these","those","into","about","your","our","their","its","of","to","in","on","at","by","it","we","you","i","a","an","plan","plans","feature","features","user","users","application","app","goal","goals"}
            return [t for t in toks if t not in stop]
        qtok = set(_q_tokens(question))
        covered = set()
        support_chunks = 0
        for c in top_chunks[:max(1, min(8, len(top_chunks)))]:
            txt = (c.get('text') or '').lower()
            if not txt:
                continue
            has = False
            for t in qtok:
                if t in txt:
                    covered.add(t)
                    has = True
            if has:
                support_chunks += 1
        min_kw = int(os.getenv("ASK_STRICT_MIN_KW", "2"))
        min_support = int(os.getenv("ASK_STRICT_MIN_SUPPORT", "1"))
        coverage_ok = (len(covered) >= min_kw)
        support_ok = (support_chunks >= min_support)
        numeric_ok = bool(rag_scores) and (rag_scores[0] >= strict_answer_min) and (max_overlap >= overlap_min)

        # LLM relevance adjudicator — primary decision if available
        llm_score = None
        llm_label = None
        llm_reason = None
        try:
            # Provide the judge with a wider slice of retrieved context than top-k
            gate_max = int(os.getenv("ASK_STRICT_LLM_MAX_HITS", "24"))
            gate_chunks = []
            for sc, ch in hits[:max(1, min(gate_max, len(hits)))] if hits else []:
                x = dict(ch)
                x["_score"] = float(sc)
                gate_chunks.append(x)
            src_for_gate = gate_chunks if gate_chunks else top_chunks
            llm_score, meta = agent.llm_relevance_gate(question, src_for_gate, doc_focus=only_doc)
            llm_label = meta.get('label') if isinstance(meta, dict) else None
            llm_reason = meta.get('reason') if isinstance(meta, dict) else None
            if progress:
                try:
                    lbl = f" {llm_label}" if llm_label else ""
                    detail = ""
                    if isinstance(meta, dict) and meta.get('ctx_snippets') is not None:
                        detail = f" • ctx={meta.get('ctx_snippets')} snippets/{meta.get('ctx_chars')} chars"
                    progress(f"LLM judge: {meta.get('model','?')} score={llm_score if llm_score is not None else 'n/a'}{lbl}{detail}")
                    if llm_reason:
                        progress(f"LLM reason: {llm_reason[:180]}")
                except Exception:
                    pass
            # Decision policy: use label if present, else treat 0.0 as Irrelevant, >0.0 as Relevant
            decided: Optional[bool] = None
            if isinstance(llm_label, str):
                decided = (llm_label.strip().lower() == 'relevant')
            elif llm_score is not None:
                decided = (float(llm_score) > 0.0)
            if decided is None:
                # No LLM score available – fall back to numeric checks
                if not (numeric_ok and coverage_ok and support_ok):
                    _emit(progress, f"Strict guard: rag={rag_scores[0] if rag_scores else 'n/a'} ov={max_overlap:.2f} kw={len(covered)} sup={support_chunks} (LLM n/a)")
                    _emit_pct(progress_pct, 100)
                    return {"answer": "**I couldn't find relevant information in your documents.**", "citations": [], "quotes": []}
            elif decided is False:
                _emit(progress, f"Strict guard: LLM veto — score={llm_score if llm_score is not None else 'n/a'} label={llm_label or 'n/a'}")
                _emit_pct(progress_pct, 100)
                msg = (
                    "**No directly relevant passage found.**\n\n"
                    "- Broaden to all documents\n"
                    "- Enable Web search\n"
                    "- Clarify or upload relevant pages"
                )
                return {"answer": msg, "citations": [], "quotes": []}
            else:
                _emit(progress, f"Strict guard: LLM pass — score={llm_score if llm_score is not None else 'n/a'} label={llm_label or 'n/a'}")
        except Exception:
            # On LLM failure, use numeric gates only
            if not (numeric_ok and coverage_ok and support_ok):
                _emit(progress, f"Strict guard: rag={rag_scores[0] if rag_scores else 'n/a'} ov={max_overlap:.2f} kw={len(covered)} sup={support_chunks} (LLM error)")
                _emit_pct(progress_pct, 100)
                msg = (
                    "**No directly relevant passage found.**\n\n"
                    "- Broaden to all documents\n"
                    "- Enable Web search\n"
                    "- Clarify or upload relevant pages"
                )
                return {"answer": msg, "citations": [], "quotes": []}
    elif web_enabled and web_chunks:
        if include_rag and rag_chunks:
            top_chunks = list(web_chunks) + rag_chunks
        else:
            top_chunks = list(web_chunks)
    else:
        top_chunks = rag_chunks

    # Prepend any extra chunks (e.g., attached image descriptions)
    if attach_chunks:
        try:
            top_chunks = list(attach_chunks) + list(top_chunks or [])
        except Exception:
            top_chunks = list(attach_chunks)

    # In-session memory (recent turns from current chat) — keep a longer window
    # Always include memory as context for pronoun/coreference resolution.
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

    # Cross-conversation memory from memory file (always include as context)
    try:
        limit = int(os.getenv("MEMORY_TOKEN_LIMIT", "1200"))
        recent = mem.load_recent(limit_tokens=limit, user_id=user_id) or []
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
    if mem_chunks and not strict_docs:
        top_chunks.extend(mem_chunks)

    # Start with whatever we have so far; refine per mode below
    combined_hits = list(hits) if hits else []
    if attach_chunks:
        try:
            combined_hits = [(0.95, ch) for ch in attach_chunks] + combined_hits
        except Exception:
            pass

    if strict_docs:
        combined_hits = list(hits)
    elif web_enabled:
        # Always include web chunks; optionally include rag if we ran it
        base_hits = list(hits) if include_rag else []
        web_extras = web_hits if web_hits else [(0.85, c) for c in web_chunks]
        combined_hits = base_hits + web_extras
    if mem_chunks and not strict_docs:
        combined_hits.extend([(0.55, ch) for ch in mem_chunks])

    if formula_mode:
        _check_cancel(should_cancel)
        _emit(progress, "Formula mode: extracting all formulas in scope…")
        total = len(top_chunks)
        if total < (len(hits) if include_rag else len(web_chunks)):
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
        ex_env = os.environ.get("ASK_EXHAUSTIVE", "false").lower() in {"1", "true", "yes", "on"}
        ex_flag = bool(exhaustive) if exhaustive is not None else ex_env
        # Dedupe similar/duplicate contexts (keep first occurrence per doc/page and near-identical text)
        def _dedupe(chs):
            seen = set(); out = []
            for c in chs:
                try:
                    key = (str(c.get('doc_name') or ''), int(c.get('page') or c.get('page_start') or 0))
                except Exception:
                    key = (str(c.get('doc_name') or ''), str(c.get('page') or c.get('page_start') or '?'))
                txt = (c.get('text') or '').strip()
                sig = (key, txt[:180])
                if sig in seen: continue
                seen.add(sig); out.append(c)
            return out
        chs = _dedupe(list(top_chunks))
        _check_cancel(should_cancel)
        out = summarize_batched(
            fmt_q,
            chs,
            history=history,
            progress_cb=lambda s: _emit(progress, s),
            max_batches=mb,
            time_budget_sec=tb,
            exhaustive=ex_flag,
            strict_docs=strict_docs,
            orig_question=_orig_user_q,
            allow_not_found=(False if (web_enabled or (attach_chunks and len(attach_chunks) > 0)) else True),
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

    # Post-process: ensure not-found responses are clearly formatted in Markdown
    try:
        if isinstance(out, dict):
            ans = (out.get("answer") or "").strip()
            low = ans.lower()
            needs_md = ("**" not in ans) and ("- " not in ans)
            not_found = ("couldn't find relevant" in low) or ("couldn’t find relevant" in low) or ("no directly relevant" in low)
            if not_found and needs_md:
                bullets = (
                    "**No directly relevant passage found.**\n\n"
                    "- Broaden to all documents\n"
                    "- Enable Web search\n"
                    "- Clarify or upload relevant pages"
                )
                # If the question invites comparative analysis, add a short general section
                try:
                    ql = (question or "").lower()
                    if any(k in ql for k in ("strength", "weakness", "compare", "versus", "vs ", "trade-off", "tradeoff")):
                        bullets += "\n\n**General perspective**\n- Typical CS strengths: strong fundamentals, algorithms/data structures, code quality\n- Typical gaps vs seasoned domain profiles: domain context, production scale experience, stakeholder comms"
                except Exception:
                    pass
                out["answer"] = bullets
    except Exception:
        pass

    # Lightweight retrieval trace for UI introspection
    try:
        def _brief(c: Dict[str, Any]) -> Dict[str, Any]:
            try:
                doc = c.get("doc_name") or "?"
                ps = c.get("page_start") or c.get("page")
                pe = c.get("page_end") or ps
                page = int(ps) if isinstance(ps, int) else ps
                if pe and pe != ps:
                    page = f"{ps}-{pe}"
            except Exception:
                page = c.get("page") or c.get("page_start") or "?"
                doc = c.get("doc_name") or "?"
            score = float(c.get("_score") or 0.0)
            kind = (
                "web" if c.get("is_web") else
                "memory" if (c.get("is_memory") or (c.get("doc_name", "") in {"Conversation memory", "Long‑term memory"})) else
                "doc"
            )
            txt = (c.get("text") or "").strip()
            snip = txt[:220].replace("\n", " ") if txt else ""
            url = c.get("web_url") if kind == "web" else None
            viewer = False
            try:
                if (kind == "doc") and user_id and doc:
                    base = pathlib.Path("artifacts") / "docs" / str(user_id) / pathlib.Path(str(doc)).name
                    viewer = base.exists() and str(base.suffix or '').lower() == '.pdf'
            except Exception:
                viewer = False
            # Provide a short "needle" text to locate on the page for highlights
            needle = (txt or "").strip().replace("\n", " ")[:160]
            return {
                "doc": doc,
                "page": page,
                "score": round(score, 4),
                "kind": kind,
                "snippet": snip,
                "url": url,
                "viewer": viewer,
                "needle": needle,
            }

        rr_name = str(os.environ.get("ASK_RERANKER", "off") or "off").lower()
        rr_applied = rr_name not in {"off", "none", ""}
        trace: Dict[str, Any] = {
            "only_doc": only_doc,
            "strict_docs": bool(strict_docs),
            "web_enabled": bool(web_enabled),
            "agents_enabled": bool(agents_enabled),
            "reranker": rr_name,
            "rerank_applied": bool(rr_applied),
            "retrieval": [
                {
                    "doc": (ch.get("doc_name") or "?"),
                    "page": (ch.get("page_start") or ch.get("page") or "?"),
                    "score": round(float(sc or 0.0), 4),
                    "kind": ("web" if (getattr(ch, 'get', None) and ch.get('is_web')) else "doc"),
                }
                for sc, ch in (hits[:10] or [])
            ],
            "selected_context": [_brief(c) for c in (top_chunks[:10] if top_chunks else [])],
            "web_sources": [c.get("web_url") for c in (web_chunks[:8] if web_chunks else []) if c.get("web_url")],
        }
        # Simple compare lists to visualize any ordering change
        try:
            trace["retrieval_head"] = [f"{(x.get('doc') or '?')} p.{(x.get('page') or '?')}" for x in trace.get("retrieval", [])[:5]]
            trace["selected_head"] = [f"{(x.get('doc') or '?')} p.{(x.get('page') or '?')}" for x in trace.get("selected_context", [])[:5]]
        except Exception:
            pass
        # Compute movement deltas for first N items
        try:
            def _key(d):
                return f"{d.get('doc') or '?'}|{d.get('page') or '?'}"
            old = { _key(x): i for i, x in enumerate(trace.get('retrieval', [])[:10]) }
            new = { _key(x): i for i, x in enumerate(trace.get('selected_context', [])[:10]) }
            moves = []
            for k, i_old in old.items():
                if k in new:
                    i_new = new[k]
                    delta = int(i_old) - int(i_new)
                    if delta != 0:
                        moves.append({"key": k, "old": int(i_old), "new": int(i_new), "delta": delta})
            moves.sort(key=lambda m: -abs(m.get('delta',0)))
            trace['rerank_moves'] = moves
        except Exception:
            pass
    except Exception:
        trace = {}

    if isinstance(out, dict):
        # Attach trace if available and include evidence quotes in trace
        try:
            if trace:
                # Add quotes (evidence) into the trace for UI display
                quotes = out.get("quotes") if isinstance(out.get("quotes"), list) else []
                if quotes:
                    trace["quotes"] = quotes
                # Citation validator: check overlap with selected context
                try:
                    checks = []
                    ans = (out.get('answer') or '')
                    def _parse_cite(tag):
                        import re
                        m = re.match(r"^\[([^]]+?)\s+p\.(.+)\]$", tag or "")
                        if not m: return None, []
                        doc = m.group(1); pages = []
                        for part in str(m.group(2)).split(','):
                            part = part.strip().replace('–','-')
                            if '-' in part:
                                a,b = part.split('-',1)
                                try:
                                    a=int(a); b=int(b)
                                    pages.extend(list(range(min(a,b), max(a,b)+1)))
                                except Exception: pass
                            else:
                                try: pages.append(int(part))
                                except Exception: pass
                        return doc, pages
                    ctx = trace.get('selected_context') or []
                    for tag in out.get('citations') or []:
                        doc, pages = _parse_cite(tag)
                        if not doc or not pages:
                            checks.append({'tag': tag, 'ok': False, 'reason': 'unparsed'})
                            continue
                        ok = False
                        for c in ctx:
                            if (c.get('doc') == doc) and (str(c.get('page')) in {str(p) for p in pages}):
                                try:
                                    ov = _token_overlap(ans, c.get('snippet') or '')
                                    if ov >= 0.15: ok = True; break
                                except Exception:
                                    pass
                        checks.append({'tag': tag, 'ok': bool(ok)})
                    trace['citation_check'] = checks
                except Exception:
                    pass
                out.setdefault("trace", trace)
        except Exception:
            pass

    if isinstance(out, dict):
        try:
            ans_low = (out.get('answer') or '').lower()
            nf = ('not found in sources' in ans_low) or ("couldn’t find relevant" in ans_low) or ("couldn't find relevant" in ans_low)
            meta = out.get('meta') if isinstance(out.get('meta'), dict) else {}
            if nf:
                meta['not_found'] = True
                if only_doc:
                    meta['suggest_broaden'] = True
                if not web_enabled:
                    meta['suggest_web'] = True
            if meta:
                out['meta'] = meta
        except Exception:
            pass

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
