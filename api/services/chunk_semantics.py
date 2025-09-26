"""Optional semantic chunk classifier loader.

Attempts to load a fine-tuned sequence classifier from
``artifacts/models/chunk_classifier_sft`` (or ``CHUNK_SEM_MODEL`` env var).
If the model or dependencies are missing, the helpers simply return without
modifying the chunks so the rest of the pipeline continues to work.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore
    _HAS_TORCH = False

_MODEL = None
_TOKENIZER = None
_ID2LABEL: dict[int, str] = {}
_INFO = {"loaded": False, "path": "", "device": "cpu"}


def _model_path() -> Path:
    cfg = os.getenv("CHUNK_SEM_MODEL")
    if cfg:
        return Path(cfg)
    return Path("artifacts/models/chunk_classifier_sft")


def semantics_info() -> dict:
    """Expose loader state for debugging."""
    return dict(_INFO)


def _ensure_model():
    global _MODEL, _TOKENIZER, _ID2LABEL
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    if not _HAS_TORCH:
        return None, None
    path = _model_path()
    if not path.exists():
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        device = os.getenv("CHUNK_SEM_DEVICE")
        if device:
            model.to(device)
        else:
            # prefer GPU if accessible
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(dev)
            device = dev
        model.eval()
        _MODEL = model
        _TOKENIZER = tok
        _INFO.update({"loaded": True, "path": str(path), "device": device})
        _ID2LABEL.clear()
        for k, v in getattr(model.config, "id2label", {}).items():
            try:
                _ID2LABEL[int(k)] = str(v)
            except Exception:
                continue
        return _MODEL, _TOKENIZER
    except Exception:
        _INFO.update({"loaded": False, "path": str(path)})
        _MODEL = None
        _TOKENIZER = None
        return None, None


def classify_chunks(chunks: Iterable[dict], batch_size: int = 64, progress_cb=None) -> List[dict]:
    """Annotate chunks with semantic labels if the classifier is available."""
    chunks = list(chunks)
    model, tokenizer = _ensure_model()
    if model is None or tokenizer is None or not chunks:
        return chunks
    if not _HAS_TORCH:
        return chunks

    texts: List[str] = []
    for ch in chunks:
        t = ch.get("text") or ""
        t = re.sub(r"\s+", " ", t).strip()[:1200]
        texts.append(t if t else "(empty)")

    device = next(model.parameters()).device
    labels: List[str] = []
    probs: List[float] = []
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                pred = logits.argmax(dim=-1)
                soft = torch.nn.functional.softmax(logits, dim=-1)
                labels.extend([_ID2LABEL.get(int(p), str(int(p))) for p in pred])
                probs.extend([float(soft[j, int(pred[j])]) for j in range(len(pred))])
                if progress_cb:
                    try:
                        progress_cb(min(len(texts), i + len(batch)), len(texts))
                    except Exception:
                        pass
    except Exception:
        return chunks

    for ch, lbl, pr in zip(chunks, labels, probs):
        ch["semantic_label"] = lbl
        ch["semantic_confidence"] = float(pr)
        ch["semantic_model"] = _INFO.get("path", "")
    return chunks


def available() -> bool:
    model, _ = _ensure_model()
    return model is not None


__all__ = ["classify_chunks", "available", "semantics_info"]
