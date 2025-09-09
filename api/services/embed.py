from typing import List
import numpy as np
from api.core.config import load_config

_EMB = None
_DIM = None

def _init_hf():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/e5-small-v2")
    return model, 384

def _init_openai():
    from openai import OpenAI
    cfg = load_config()
    client = OpenAI(api_key=cfg.get("OPENAI_API_KEY"))
    return client, 1536

def _normalize(v: np.ndarray):
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def get_embedder():
    global _EMB, _DIM
    if _EMB is not None:
        return _EMB, _DIM
    backend = (load_config().get("EMBED_BACKEND") or "hf").lower()
    if backend == "openai":
        _EMB, _DIM = _init_openai()
    else:
        _EMB, _DIM = _init_hf()
    return _EMB, _DIM

def embed_texts(texts: List[str]):
    backend = (load_config().get("EMBED_BACKEND") or "hf").lower()
    model, _ = get_embedder()
    if backend == "openai":
        out = model.embeddings.create(model="text-embedding-3-small", input=texts)
        vecs = np.array([d.embedding for d in out.data], dtype=np.float32)
        return _normalize(vecs)
    else:
        from sentence_transformers import SentenceTransformer
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)
