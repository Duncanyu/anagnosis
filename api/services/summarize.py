from api.core.config import load_config
from api.services.embed import embed_texts
from api.services.aliases import load_aliases
import os, re, json, time, pathlib, math
import numpy as np
import unicodedata

TEMPLATE = """You are a study assistant. Answer the question using ONLY the provided context.
Start with a single bold sentence that directly answers the question. Then, if helpful, add short bullet points. Do not start the response with a bullet. Cite sources like [FileName.pdf p.3] or [FileName.pdf p.3–4].
Important terms to anchor on: {terms}
If none of the important terms appear in the context, still answer from the context you have.

Question:
{q}

Context:
{ctx}
"""

DOC_TEMPLATE = """You are a rigorous study assistant for university readings.
Return a comprehensive summary in Markdown with these sections:
## High-level overview
## Section-by-section distilled notes
## Key terms
## Formulas
Use concise bullets and include page citations like [FileName.pdf p.12] or [FileName.pdf p.4–6]. Do not invent facts.

Context:
{ctx}
"""

SUMMARIES_PATH = pathlib.Path("artifacts") / "doc_summaries.jsonl"
CTX_CHAR_BUDGET = 14000
BATCH_CHAR_BUDGET = int(os.getenv("ASK_BATCH_CHAR_BUDGET", "12000"))
MAX_BATCHES_DEFAULT = int(os.getenv("ASK_MAX_BATCHES", "6"))
TIME_BUDGET_SEC_DEFAULT = int(os.getenv("ASK_TIME_BUDGET_SEC", "120"))
MMR_TOP_N = int(os.getenv("ASK_MMR_TOP_N", "200"))
MMR_ALPHA = float(os.getenv("ASK_MMR_ALPHA", "0.7"))
ASK_EXHAUSTIVE_DEFAULT = (os.getenv("ASK_EXHAUSTIVE","0").lower() in {"1","true","yes","on"})
ASK_CANDIDATES_DEFAULT = int(os.getenv("ASK_CANDIDATES","300"))
RERANKER_NAME_DEFAULT = os.getenv("ASK_RERANKER","off").lower()
RERANK_TOP_N = int(os.getenv("ASK_RERANK_TOP_N","200"))
FORMULA_Q_WORDS = {"formula","formulas","identity","identities","equation","equations","rule","rules","laws","law"}
FORMULA_HINT_RX = re.compile(r"(=|≈|≅|≤|≥|±|∝|∑|∏|∫|√|∞|d/dx|∂|∇|\be\^|\bdy/dx\b|\b(sin|cos|tan|cot|sec|csc)\b|\btheta\b|\bpi\b)", re.I)

_HF_PIPE = None
_HF_NAME = None

CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]')

MATH_SHORT = {"sin","cos","tan","sec","csc","cot","pi","π","ln","log","rad","deg","e","ix","dx","dt"}
MATH_PHRASES = {
    "pythagorean","double-angle","half-angle","sum and difference","product-to-sum","sum-to-product",
    "angle addition","angle subtraction","unit circle","euler","complex exponential","radian","degree",
    "identity","identities","formula","formulas"
}

def _openai_model():
    cfg = load_config()
    return cfg.get("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"

def _nfkc(s):
    try:
        return unicodedata.normalize("NFKC", s or "")
    except Exception:
        return s or ""

def is_formula_query(q):
    s = (q or "").lower()
    return any(w in s for w in FORMULA_Q_WORDS)

def _split_lines_blocks(t):
    t = _nfkc(t)
    t = t.replace("\u00a0", " ")
    parts = []
    for ln in t.splitlines():
        x = ln.strip(" \t\r\n•-—–·")
        if x:
            parts.append(x)
    return parts

def _is_formula_line(s):
    s = s.strip()
    if not s:
        return False
    if FORMULA_HINT_RX.search(s):
        return True
    return False

def _dedup_ordered(items):
    out = []
    seen = set()
    for x in items:
        k = re.sub(r"\s+", " ", x).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def extract_formulas_from_chunks(chs, max_per_page=200, progress_cb=None):
    rows = []
    total = len(chs)
    hit = 0
    for i, c in enumerate(chs):
        t = c.get("text","") or ""
        if not t:
            if progress_cb and (i % 50 == 0 or i == total - 1):
                progress_cb(f"Scanning pages… {i+1}/{total} • formulas: {hit}")
            continue
        quick = FORMULA_HINT_RX.search(t)
        if not quick:
            if progress_cb and (i % 50 == 0 or i == total - 1):
                progress_cb(f"Scanning pages… {i+1}/{total} • formulas: {hit}")
            continue
        lines = _split_lines_blocks(t)
        got = []
        for ln in lines:
            if _is_formula_line(ln):
                got.append(ln.strip())
                if len(got) >= max_per_page:
                    break
        if got:
            got = _dedup_ordered(got)
            hit += len(got)
            src = _fmt_source(c)
            for g in got:
                rows.append({"formula": g[:800], "source": src})
        if progress_cb and (i % 20 == 0 or i == total - 1):
            progress_cb(f"Scanning pages… {i+1}/{total} • formulas: {hit}")
    return rows

def summarize_all_formulas(question, chunks, progress_cb=None):
    base = list(chunks)
    if not base:
        return {"answer": "Not found in sources.", "citations": [], "quotes": []}
    mathy = []
    for c in base:
        t = c.get("text","")
        if c.get("has_math") or FORMULA_HINT_RX.search(t or ""):
            mathy.append(c)
    used = mathy if mathy else base
    if progress_cb:
        progress_cb(f"Formula mode: scanning {len(used)} pages")
    rows = extract_formulas_from_chunks(used, progress_cb=progress_cb)
    if not rows:
        return {"answer": "Not found in sources.", "citations": [], "quotes": []}
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r["formula"])
    md = []
    md.append("**Formulas found**")
    for src in sorted(by_src.keys()):
        fs = _dedup_ordered(by_src[src])
        md.append(f"\n- {src}")
        for f in fs:
            md.append(f"  - `{f}`")
    cites = _dedup_ordered(list(by_src.keys()))
    return {"answer": "\n".join(md), "citations": cites, "quotes": []}

def _normalize_md(s):
    s = s.strip().replace("\u00a0", " ")
    m = re.match(r"^\s*```[\w\-]*\s*\n?(.*?)\n?\s*```\s*$", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"(?m)^\s*[–—•]\s+", "- ", s)
    lines = [ln.rstrip() for ln in s.splitlines()]
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("- "):
        lines[i] = lines[i].lstrip()[2:].strip()
    return "\n".join(lines).strip()

def _ascii_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if ord(ch) < 128)
    return n / len(s)

def _filter_chunks(chs):
    out = []
    for c in chs:
        t = _nfkc(c.get("text") or "").strip()
        if not t:
            continue
        if CJK_RE.search(t):
            if _ascii_ratio(t) < 0.20:
                continue
        out.append(c)
    return out

def _ascii_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if ord(ch) < 128)
    return n / len(s)

def _filter_chunks(chs):
    out = []
    for c in chs:
        t = _nfkc(c.get("text") or "").strip()
        if not t:
            continue
        if CJK_RE.search(t):
            if _ascii_ratio(t) < 0.20:
                continue
        out.append(c)
    return out

def _rerank(question, chs, name=RERANKER_NAME_DEFAULT, top_n=RERANK_TOP_N):
    tag = (name or "off").lower().strip()
    if tag in {"off","none",""}:
        return list(range(len(chs)))
    try:
        import torch
        from sentence_transformers import CrossEncoder
        model_map = {
            "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "bge-m3": "BAAI/bge-reranker-v2-m3",
            "bge-base": "BAAI/bge-reranker-base",
            "bge-large": "BAAI/bge-reranker-large"
        }
        mname = model_map.get(tag, tag)
        ce = CrossEncoder(mname, device="cuda" if torch.cuda.is_available() else "cpu")
        cut = min(len(chs), top_n)
        pairs = [[question, re.sub(r"\s+"," ", chs[i]["text"]).strip()[:600]] for i in range(cut)]
        scores = ce.predict(pairs)
        order = list(np.argsort(-np.array(scores)))
        rest = list(range(cut, len(chs)))
        return order + rest
    except Exception:
        return list(range(len(chs)))

def _fmt_source(c):
    doc = c.get("doc_name") or "Unknown.pdf"
    p_single = c.get("page") or c.get("page_num") or c.get("page_index")
    if p_single:
        return f"[{doc} p.{p_single}]"
    if c.get("page_start") == c.get("page_end"):
        page = f"{c.get('page_start','?')}"
    else:
        page = f"{c.get('page_start','?')}-{c.get('page_end','?')}"
    return f"[{doc} p.{page}]"

def _format_ctx(chs):
    parts = []
    for c in chs:
        parts.append(f"{_fmt_source(c)} {c['text'][:1200]}")
    return "\n---\n".join(parts)

def _clip_context(chunks, budget=CTX_CHAR_BUDGET):
    out, used = [], 0
    for c in chunks:
        s = re.sub(r"\s+", " ", c["text"]).strip()
        if not s:
            continue
        block = f"{_fmt_source(c)} {s}"
        if used + len(block) > budget:
            remain = max(0, budget - used)
            out.append(block[:remain])
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)

def summarizer_info():
    return {"backend": "openai" if load_config().get("OPENAI_API_KEY") else "hf"}

def _hf_pipe():
    global _HF_PIPE, _HF_NAME
    if _HF_PIPE is not None:
        return _HF_PIPE, _HF_NAME
    from transformers import pipeline
    import torch
    name = os.getenv("HF_LLM_NAME") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = 0 if torch.cuda.is_available() else -1
    _HF_PIPE = pipeline("text-generation", model=name, tokenizer=name, device=device)
    _HF_NAME = name
    return _HF_PIPE, _HF_NAME

def _hf_generate(prompt, max_new_tokens=800):
    pipe, _ = _hf_pipe()
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2, return_full_text=False, pad_token_id=pipe.tokenizer.eos_token_id or 0)
    return out[0]["generated_text"].strip()

def _split_sentences(t):
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[\.\!\?;:])\s+(?=[A-Za-z0-9\\(])|\n+", t)
    return [p.strip() for p in parts if p.strip()]

def _extract_quotes(question, chunks, max_quotes=5, window=1):
    kws = set(_keywords(question))
    sents = []
    meta = []
    for c in chunks:
        ss = _split_sentences(c["text"])
        for i, s in enumerate(ss):
            sents.append(s[:400])
            meta.append((c, i, ss))
    if not sents:
        return []
    qv = embed_texts([question])
    sv = embed_texts(sents)
    sims = (qv @ sv.T).flatten()
    lex = []
    for s in sents:
        s_low = s.lower()
        score = 0
        for k in kws:
            if k in s_low:
                score += 1
        if "\\sin" in s or "\\cos" in s or "\\tan" in s:
            score += 1
        if "π" in s or "pi" in s_low or "e^i" in s_low:
            score += 1
        lex.append(score)
    lex = np.array(lex, dtype=float)
    if lex.max() > 0:
        lex = lex / lex.max()
    comb = 0.8 * sims + 0.2 * lex
    order = np.argsort(-comb)
    seen = set()
    quotes = []
    for idx in order:
        c, i, ss = meta[idx]
        key = (c.get("doc_name") or "Unknown.pdf", c.get("page") or c.get("page_start"))
        if key in seen:
            continue
        start = max(0, i - window)
        end = min(len(ss), i + window + 1)
        snippet = " ".join(ss[start:end])[:500]
        quotes.append({"source": _fmt_source(c), "quote": snippet})
        seen.add(key)
        if len(quotes) >= max_quotes:
            break
    return quotes

def _filter_chunks(chs):
    out = []
    for c in chs:
        t = c.get("text") or ""
        t = _nfkc(t).strip()
        if not t:
            continue
        if CJK_RE.search(t):
            continue
        out.append(c)
    return out

def _keywords(q):
    q = q.lower()
    q = re.sub(r"[^a-z0-9\sπ]", " ", q)
    toks = [w for w in q.split() if w and w not in {"what","which","this","that","about","were","does","with","from","your","their","into","there","when","where","many","show","give","tell","list"}]
    out = set()
    for w in toks:
        if len(w) >= 4:
            out.add(w)
        elif w in MATH_SHORT:
            out.add(w)
    for p in MATH_PHRASES:
        if p.replace("-", " ") in q or p in q:
            out.add(p)
    if "trig" in q or "trigonometry" in q:
        out |= {"sin","cos","tan","identity","identities","pythagorean","double-angle","half-angle"}
    return list(out)

def _pref_string():
    a = load_aliases()
    if not a:
        return ""
    pairs = [f"{k} -> {v}" for k,v in a.items()]
    return "Use user-preferred terminology: " + "; ".join(pairs) + "."

def _estimate_tokens(s):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)

_rate_state = {"t_start":0.0,"tok":0,"r_start":0.0,"req":0}

def _rate_limit(tokens):
    cfg = load_config()
    tpm = int(cfg.get("OPENAI_TPM") or os.getenv("OPENAI_TPM") or "0")
    rpm = int(cfg.get("OPENAI_RPM") or os.getenv("OPENAI_RPM") or "0")
    now = time.time()
    if _rate_state["t_start"] == 0.0:
        _rate_state["t_start"] = now
        _rate_state["r_start"] = now
    if tpm > 0:
        if now - _rate_state["t_start"] >= 60:
            _rate_state["t_start"] = now
            _rate_state["tok"] = 0
        if _rate_state["tok"] + tokens > tpm:
            sleep = 60 - (now - _rate_state["t_start"])
            if sleep > 0:
                time.sleep(sleep)
                _rate_state["t_start"] = time.time()
                _rate_state["tok"] = 0
        _rate_state["tok"] += tokens
    if rpm > 0:
        if now - _rate_state["r_start"] >= 60:
            _rate_state["r_start"] = now
            _rate_state["req"] = 0
        if _rate_state["req"] + 1 > rpm:
            sleep = 60 - (now - _rate_state["r_start"])
            if sleep > 0:
                time.sleep(sleep)
                _rate_state["r_start"] = time.time()
                _rate_state["req"] = 0
        _rate_state["req"] += 1

def _openai_chat(messages, max_new_tokens=800):
    from openai import OpenAI
    client = OpenAI(api_key=(load_config().get("OPENAI_API_KEY")))
    s = "\n".join(m.get("content","") for m in messages)
    need = _estimate_tokens(s) + max_new_tokens
    _rate_limit(need)
    msg = client.chat.completions.create(model=_openai_model(), messages=messages)
    return _normalize_md(msg.choices[0].message.content), need


def _score_chunk_for_order(c):
    s = 0.0
    try:
        s = float(c.get("_score", 0.0))
    except Exception:
        s = 0.0
    if c.get("has_math"):
        s += 0.1
    if c.get("has_table"):
        s -= 0.05
    return -s

def _mmr_order(question, chs, top_n=MMR_TOP_N, alpha=MMR_ALPHA):
    n = min(len(chs), top_n)
    if n <= 2:
        return list(range(len(chs)))
    texts = [re.sub(r"\s+", " ", c["text"]).strip()[:800] for c in chs[:n]]
    qv = embed_texts([question])[0]
    dv = embed_texts(texts)
    qv = qv / (np.linalg.norm(qv) + 1e-8)
    dv = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8)
    sims = dv @ qv
    selected = []
    candidates = list(range(n))
    selected.append(int(np.argmax(sims)))
    candidates.remove(selected[0])
    while candidates:
        mmr_scores = []
        for idx in candidates:
            rel = sims[idx]
            if selected:
                div = max(float(dv[idx] @ dv[j]) for j in selected)
            else:
                div = 0.0
            mmr = alpha * rel - (1 - alpha) * div
            mmr_scores.append((mmr, idx))
        mmr_scores.sort(reverse=True)
        pick = mmr_scores[0][1]
        selected.append(pick)
        candidates.remove(pick)
    rest = list(range(n, len(chs)))
    return selected + rest

def _batch_chunks(chs, budget_chars):
    out = []
    cur = []
    used = 0
    for c in chs:
        t = re.sub(r"\s+", " ", c["text"]).strip()
        if not t:
            continue
        block = f"{_fmt_source(c)} {t}"
        if used + len(block) > budget_chars and cur:
            out.append(cur)
            cur = [c]
            used = len(block)
        else:
            cur.append(c)
            used += len(block)
        if used >= budget_chars * 1.2 and cur:
            out.append(cur)
            cur = []
            used = 0
    if cur:
        out.append(cur)
    return out

def summarize(question, top_chunks, history=None):
    cfg = load_config()
    filtered = _filter_chunks(top_chunks)
    used = filtered if filtered else list(top_chunks)
    if not used:
        return {"answer": "Not found in sources.", "citations": [], "quotes": []}
    terms = ", ".join(_keywords(question))
    ctx = _format_ctx(used)
    prompt = TEMPLATE.format(q=question, ctx=ctx, terms=terms)
    if cfg.get("OPENAI_API_KEY"):
        prefs = _pref_string()
        sysmsg = "You extract key ideas and cite pages. Return Markdown." + (" " + prefs if prefs else "")
        messages = [{"role": "system", "content": sysmsg}]
        if history:
            for turn in history[-8:]:
                if turn.get("q"):
                    messages.append({"role": "user", "content": turn["q"]})
                if turn.get("a"):
                    messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": prompt})
        text, _ = _openai_chat(messages, max_new_tokens=700)
    else:
        hist = ""
        if history:
            for turn in history[-8:]:
                if turn.get("q"):
                    hist += f"\nUser: {turn['q']}\n"
                if turn.get("a"):
                    hist += f"Assistant: {turn['a']}\n"
        full = "System: You extract key ideas from the provided context only. Return Markdown with citations like [FileName.pdf p.12]." + hist + "\n" + prompt + "\nAssistant:"
        text = _normalize_md(_hf_generate(full, max_new_tokens=700))
    cites = []
    for c in used:
        cites.append(_fmt_source(c))
    dedup = []
    seen = set()
    for s in cites:
        if s not in seen:
            dedup.append(s); seen.add(s)
    quotes = _extract_quotes(question, used, max_quotes=5, window=1)
    return {"answer": text, "citations": dedup, "quotes": quotes}

def summarize_batched(question, chunks, history=None, progress_cb=None, max_batches=None, time_budget_sec=None, exhaustive=None):
    cfg = load_config()
    filtered = _filter_chunks(chunks)
    base = filtered if filtered else list(chunks)
    if not base:
        return {"answer": "Not found in sources.", "citations": [], "quotes": []}
    if max_batches is None:
        max_batches = MAX_BATCHES_DEFAULT
    if time_budget_sec is None:
        time_budget_sec = TIME_BUDGET_SEC_DEFAULT
    if exhaustive is None:
        exhaustive = ASK_EXHAUSTIVE_DEFAULT
    prim = sorted(base, key=_score_chunk_for_order)
    try:
        mmr_idx = _mmr_order(question, prim)
        ordered = [prim[i] for i in mmr_idx]
    except Exception:
        ordered = prim
    try:
        rer_idx = _rerank(question, ordered, name=RERANKER_NAME_DEFAULT, top_n=RERANK_TOP_N)
        ordered = [ordered[i] for i in rer_idx]
    except Exception:
        pass
    batches = _batch_chunks(ordered, BATCH_CHAR_BUDGET)
    if cfg.get("OPENAI_API_KEY"):
        prefs = _pref_string()
        sysmsg = "You extract key ideas and cite pages. Return Markdown." + (" " + prefs if prefs else "")
        hist_msgs = []
        if history:
            for turn in history[-4:]:
                if turn.get("q"):
                    hist_msgs.append({"role": "user", "content": turn["q"]})
                if turn.get("a"):
                    hist_msgs.append({"role": "assistant", "content": turn["a"]})
        tpm = int(cfg.get("OPENAI_TPM") or os.getenv("OPENAI_TPM") or "0")
        rpm = int(cfg.get("OPENAI_RPM") or os.getenv("OPENAI_RPM") or "0")
        tok_cap = math.inf if tpm <= 0 else int(tpm * (time_budget_sec / 60.0))
        req_cap = math.inf if rpm <= 0 else int(rpm * (time_budget_sec / 60.0))
        used_tok = 0
        used_req = 0
        parts = []
        seen_sources = []
        total = min(len(batches), max_batches)
        t_end = time.time() + time_budget_sec
        plateau = 0
        for i, batch in enumerate(batches[:total], 1):
            if time.time() >= t_end:
                break
            ctx = _format_ctx(batch)
            prompt = TEMPLATE.format(q=question, ctx=ctx, terms=", ".join(_keywords(question)))
            messages = [{"role": "system", "content": sysmsg}] + hist_msgs + [{"role": "user", "content": prompt}]
            pre_tokens = _estimate_tokens("\n".join(m["content"] for m in messages)) + 600
            if used_tok + pre_tokens > tok_cap or used_req + 1 > req_cap:
                break
            if progress_cb:
                try:
                    left = int(max(0, t_end - time.time()))
                    progress_cb(f"Batch {i}/{total} • {used_tok}/{tok_cap if tok_cap!=math.inf else 0} tok • {left}s left")
                except Exception:
                    pass
            ans, spent = _openai_chat(messages, max_new_tokens=600)
            used_tok += spent
            used_req += 1
            parts.append(ans)
            prev = set(seen_sources)
            for c in batch:
                seen_sources.append(_fmt_source(c))
            delta = len(set(seen_sources) - prev)
            if not exhaustive and i >= 2 and delta == 0:
                break
        if not parts:
            return summarize(question, ordered[:max(1, min(6, len(ordered)))], history=history)
        fuse_ctx = "\n\n---\n\n".join(parts)[:40000]
        fuse_prompt = "You are consolidating multiple partial answers derived strictly from course readings. Merge them into a single, non-redundant answer. Keep formulas, be precise, and keep citations from the partials in place.\n\nQuestion:\n" + question + "\n\nPartials:\n" + fuse_ctx
        final, _ = _openai_chat([{"role": "system", "content": "Return a single clean Markdown answer with citations kept as-is."}, {"role": "user", "content": fuse_prompt}], max_new_tokens=800)
    else:
        parts = []
        seen_sources = []
        total = min(len(batches), max_batches)
        t_end = time.time() + time_budget_sec
        for i, batch in enumerate(batches[:total], 1):
            if time.time() >= t_end:
                break
            if progress_cb:
                try:
                    left = int(max(0, t_end - time.time()))
                    progress_cb(f"Batch {i}/{total} • {left}s left")
                except Exception:
                    pass
            ctx = _format_ctx(batch)
            prompt = TEMPLATE.format(q=question, ctx=ctx, terms=", ".join(_keywords(question)))
            hist = ""
            if history:
                for turn in history[-4:]:
                    if turn.get("q"):
                        hist += f"\nUser: {turn['q']}\n"
                    if turn.get("a"):
                        hist += f"Assistant: {turn['a']}\n"
            full = "System: You extract key ideas from the provided context only. Return Markdown with citations like [FileName.pdf p.12]." + hist + "\n" + prompt + "\nAssistant:"
            ans = _normalize_md(_hf_generate(full, max_new_tokens=600))
            parts.append(ans)
            prev = set(seen_sources)
            for c in batch:
                seen_sources.append(_fmt_source(c))
            delta = len(set(seen_sources) - prev)
            if not exhaustive and i >= 2 and delta == 0:
                break
        if not parts:
            return summarize(question, ordered[:max(1, min(6, len(ordered)))], history=history)
        fuse_ctx = "\n\n---\n\n".join(parts)[:40000]
        fuse_prompt = "You are consolidating multiple partial answers derived strictly from course readings. Merge them into a single, non-redundant answer. Keep formulas, be precise, and keep citations from the partials in place.\n\nQuestion:\n" + question + "\n\nPartials:\n" + fuse_ctx
        final = _normalize_md(_hf_generate("System: Consolidate the partials into one answer, preserving citations.\n" + fuse_prompt + "\nAssistant:", max_new_tokens=800))
    cites = []
    seen = set()
    for s in seen_sources:
        if s not in seen:
            cites.append(s); seen.add(s)
        if len(cites) >= 15:
            break
    quotes = _extract_quotes(question, ordered[:min(len(ordered), 200)], max_quotes=5, window=1)
    return {"answer": final, "citations": cites, "quotes": quotes}

def summarize_document(chunks):
    cfg = load_config()
    filtered = _filter_chunks(chunks)
    if not filtered:
        summary = "**No clean extractable text found in the provided pages.**"
        SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARIES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": int(time.time()), "keywords": [], "formulas": [], "summary": summary}, ensure_ascii=False) + "\n")
        return {"summary": summary, "keywords": [], "formulas": []}
    ctx = _clip_context(filtered)
    if cfg.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
        prompt = DOC_TEMPLATE.format(ctx=ctx)
        msg = client.chat.completions.create(
            model=_openai_model(),
            messages=[
                {"role": "system", "content": "Return structured, factual notes with page citations in Markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        summary = _normalize_md(msg.choices[0].message.content)
    else:
        full = "System: Produce a comprehensive study summary in Markdown with bullets and citations like [FileName.pdf p.12].\n" + DOC_TEMPLATE.format(ctx=ctx) + "\nAssistant:"
        summary = _normalize_md(_hf_generate(full, max_new_tokens=1200))
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARIES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "keywords": [], "formulas": [], "summary": summary[:40000]}, ensure_ascii=False) + "\n")
    return {"summary": summary, "keywords": [], "formulas": []}
