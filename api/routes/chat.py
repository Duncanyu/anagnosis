from __future__ import annotations

import os
import re
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth
from api.db.models import User
from api.core.config import load_config

router = APIRouter(prefix="/api", tags=["chat"])


def _sanitize_title(s: str, max_words: int = 4, max_len: int = 64) -> str:
    s = (s or "").strip().strip('"\'\u201c\u201d').strip()
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\-\s:/]", "", s)
    words = [w for w in s.split() if w]
    if len(words) > max_words:
        words = words[:max_words]
    title = " ".join(words).title()
    return (title[:max_len] or "New Chat").strip()


def _heuristic_title(text: str) -> str:
    import collections
    raw = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]{3,}", raw)
    stop = {
        "what","does","this","that","with","from","your","about","the","and","for","are","was","were","will","would","should","could","please","help","explain","write","make","into","give","show","list","how","can","you","him","her","their","they","them","its","it's","our","mine","yours","new","chat","message","reply","answer","question","pdf","doc","file","files","document","documents","image","images","screenshot","photo","analysis"
    }
    freq = collections.Counter(t for t in tokens if t not in stop)
    top = [w for w, _ in freq.most_common(8)]
    top.sort(key=lambda w: (-freq[w], -len(w), w))
    words = top[:4]
    if not words:
        return "New Chat"
    return _sanitize_title(" ".join(words), max_words=4)


def _generate_chat_title(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "New Chat"
    # Prefer OpenAI if configured
    try:
        cfg = load_config() or {}
        key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        if key:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            sys = (
                "You generate chat titles. Return ONLY a concise Title Case title, 2–4 words, summarizing the whole conversation (both questions and answers). "
                "No quotes, no trailing punctuation, avoid generic words like 'Chat' or 'Conversation'."
            )
            prompt = text[:3000]
            msg = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = (msg.choices[0].message.content or "").strip()
            cand = _sanitize_title(raw, max_words=4)
            # Guard against echoing the opening question; if similar, fall back to heuristic
            try:
                first_q = ""
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    if line.strip().lower().startswith("q:"):
                        first_q = line.split(":",1)[1].strip()
                        break
                    else:
                        first_q = line.strip()
                        break
                # Compare overlapping tokens
                import re as _re
                def toks(s):
                    return [w for w in _re.findall(r"[a-z0-9]{3,}", (s or "").lower())]
                qtok = toks(first_q)[:6]
                ttok = toks(cand)
                overlap = len(set(ttok) & set(qtok)) / max(1, len(set(ttok)))
                if overlap >= 0.75:
                    return _heuristic_title(text)
            except Exception:
                pass
            return cand
    except Exception:
        pass

    # Heuristic fallback: keyword-based
    return _heuristic_title(text)


@router.post("/chat/title")
async def api_chat_title(req: Request, user: User = Depends(require_auth)) -> JSONResponse:
    try:
        data = await req.json()
    except Exception:
        data = {}
    text = str(data.get("text") or data.get("summary") or "").strip()
    title = _generate_chat_title(text)
    return JSONResponse({"title": title})
