from typing import List, Dict
from api.core.config import load_config
from api.services.embed import embed_texts
import os, re, json, time, pathlib
import numpy as np

TEMPLATE = """You are a study assistant. Answer the question using ONLY the provided context.
Return the answer in Markdown. Use short bullet points. Cite sources like [FileName.pdf p.3] or [FileName.pdf p.3–4].

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

_HF_PIPE = None
_HF_NAME = None

def _normalize_md(s: str):
    s = s.strip().replace("\u00a0", " ")
    m = re.match(r"^\s*```[\w\-]*\s*\n?(.*?)\n?\s*```\s*$", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"(?m)^\s*[–—•]\s+", "- ", s)
    return s

def _fmt_source(c: Dict):
    doc = c.get("doc_name") or "Unknown.pdf"
    if c.get("page_start") == c.get("page_end"):
        page = f"{c.get('page_start','?')}"
    else:
        page = f"{c.get('page_start','?')}-{c.get('page_end','?')}"
    return f"[{doc} p.{page}]"

def _format_ctx(chs: List[Dict]):
    parts = []
    for c in chs:
        parts.append(f"{_fmt_source(c)} {c['text'][:1200]}")
    return "\n---\n".join(parts)

def _clip_context(chunks: List[Dict], budget: int = CTX_CHAR_BUDGET):
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

def _hf_generate(prompt: str, max_new_tokens: int = 800):
    pipe, _ = _hf_pipe()
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2, return_full_text=False, pad_token_id=pipe.tokenizer.eos_token_id or 0)
    return out[0]["generated_text"].strip()

def _split_sentences(t: str):
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9(])", t)
    return [p.strip() for p in parts if p.strip()]

def _extract_quotes(question: str, chunks: List[Dict], max_quotes: int = 5, window: int = 1):
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
    order = np.argsort(-sims)
    seen = set()
    quotes = []
    for idx in order:
        c, i, ss = meta[idx]
        key = (c.get("doc_name") or "Unknown.pdf", c.get("page_start"))
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

def summarize(question: str, top_chunks: List[Dict], history: List[Dict] = None):
    cfg = load_config()
    ctx = _format_ctx(top_chunks)
    prompt = TEMPLATE.format(q=question, ctx=ctx)

    if cfg.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
        messages = [{"role": "system", "content": "You extract key ideas and cite pages. Return Markdown."}]
        if history:
            for turn in history[-8:]:
                if turn.get("q"):
                    messages.append({"role": "user", "content": turn["q"]})
                if turn.get("a"):
                    messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": prompt})
        msg = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
        text = _normalize_md(msg.choices[0].message.content)
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
    for c in top_chunks:
        cites.append(_fmt_source(c))
    dedup = []
    seen = set()
    for s in cites:
        if s not in seen:
            dedup.append(s); seen.add(s)

    quotes = _extract_quotes(question, top_chunks, max_quotes=5, window=1)
    return {"answer": text, "citations": dedup, "quotes": quotes}

def summarize_document(chunks: List[Dict]):
    cfg = load_config()
    ctx = _clip_context(chunks)

    if cfg.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
        prompt = DOC_TEMPLATE.format(ctx=ctx)
        msg = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return structured, factual notes with page citations in Markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        summary = _normalize_md(msg.choices[0].message.content)
    else:
        full = "System: Produce a comprehensive study summary in Markdown with bullets and citations like [FileName.pdf p.12].\n" + DOC_TEMPLATE.format(ctx=ctx) + "\nAssistant:"
        summary = _normalize_md(_hf_generate(full, max_new_tokens=1200))

    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARIES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "keywords": [], "formulas": [], "summary": summary[:40000]}, ensure_ascii=False) + "\n")

    return {"summary": summary, "keywords": [], "formulas": []}
