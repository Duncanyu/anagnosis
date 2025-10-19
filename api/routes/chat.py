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


@router.post("/chat/title")
async def api_chat_title(req: Request, user: User = Depends(require_auth)) -> JSONResponse:
    try:
        data = await req.json()
    except Exception:
        data = {}
    text = str(data.get("text") or data.get("summary") or "").strip()
    title = _generate_chat_title(text)
    return JSONResponse({"title": title})

