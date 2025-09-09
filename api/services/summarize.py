from typing import List, Dict
from api.core.config import load_config

TEMPLATE = """You are a study assistant. Answer the question using ONLY the provided context.
Return concise bullets with citations like [p.{page}].

Question:
{q}

Context:
{ctx}
"""

def _format_ctx(chs: List[Dict]):
    parts = []
    for c in chs:
        page = f"{c.get('page_start','?')}-{c.get('page_end','?')}" if c.get("page_start") != c.get("page_end") else f"{c.get('page_start','?')}"
        parts.append(f"[p.{page}] {c['text'][:1200]}")
    return "\n---\n".join(parts)

def summarize(question: str, top_chunks: List[Dict]):
    cfg = load_config()
    ctx = _format_ctx(top_chunks)
    prompt = TEMPLATE.format(q=question, ctx=ctx)

    if not cfg.get("OPENAI_API_KEY"):
        bullets = []
        for c in top_chunks:
            page = c.get("page_start")
            snippet = c["text"].split("\n")[0][:220].strip()
            bullets.append(f"• {snippet} [p.{page}]")
        return {"answer": "\n".join(bullets), "citations": [f"p.{c.get('page_start')}" for c in top_chunks]}

    from openai import OpenAI
    client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
    msg = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract key ideas and cite pages."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = msg.choices[0].message.content
    cites = []
    for c in top_chunks:
        pg = c.get("page_start")
        cites.append(f"p.{pg}")
    return {"answer": text, "citations": cites}
