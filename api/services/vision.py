from __future__ import annotations

import base64
import os
from typing import Optional

from api.core.config import load_config


def describe_image_openai(image_bytes: bytes, *, prompt: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """Return a concise description of an image using an OpenAI vision-capable model.

    If no OpenAI API key is configured, returns None.
    """
    cfg = load_config() or {}
    key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI(api_key=key)
        model_name = model or cfg.get("OPENAI_CHAT_MODEL") or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        # Default prompt tuned for screenshots and documents
        p = prompt or (
            "You are an image understanding assistant. First, transcribe any readable text verbatim in plain text, "
            "preserving line breaks and order. Then give a detailed, well‑structured description of the visual content. "
            "Use short section headings like 'Overview', 'Details', and 'Text (if any)'. Mention notable objects, layout, actions, and any relevant context. "
            "Unless the user asks for brevity, aim for 180–320 words."
        )
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        msg = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You analyze images and return concise, factual descriptions. Keep it under 120 words."},
                {"role": "user", "content": [
                    {"type": "text", "text": p},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            temperature=0.2,
        )
        text = (msg.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None
