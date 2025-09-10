import os, time, numpy as np
from api.core.config import load_config

def embedding_info():
    cfg = load_config()
    backend = (cfg.get("EMBED_BACKEND") or "hf").lower()
    if backend == "openai":
        model = cfg.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small"
    else:
        model = cfg.get("HF_EMBED_MODEL") or "intfloat/e5-small-v2"
    return {"backend": backend, "model": model}

def _norm_rows(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    return a / n

def _embed_openai(texts, model, batch_size, progress_cb=None):
    from openai import OpenAI
    cfg = load_config()
    client = OpenAI(api_key=cfg.get("OPENAI_API_KEY"))
    out = []
    i = 0
    total = len(texts)
    done = 0
    while i < total:
        batch = texts[i:i+batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
                if attempt == 4:
                    raise
        i += batch_size
        done = min(i, total)
        if progress_cb:
            progress_cb(done, total)
    arr = np.asarray(out, dtype="float32")
    return _norm_rows(arr)

def _embed_hf(texts, model_name, progress_cb=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    out = []
    bs = 64
    total = len(texts)
    done = 0
    for i in range(0, total, bs):
        vecs = model.encode(texts[i:i+bs], batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        out.append(np.asarray(vecs, dtype="float32"))
        done = min(i + bs, total)
        if progress_cb:
            progress_cb(done, total)
    arr = np.vstack(out) if out else np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return arr

def embed_texts(texts, show_progress_bar=False, normalize_embeddings=True, progress_cb=None):
    cfg = load_config()
    backend = (cfg.get("EMBED_BACKEND") or "hf").lower()
    if backend == "openai":
        model = cfg.get("OPENAI_EMBED_MODEL") or "text-embedding-3-small"
        batch_size = int(cfg.get("EMBED_BATCH") or 128)
        arr = _embed_openai(texts, model, batch_size, progress_cb=progress_cb)
    else:
        model = cfg.get("HF_EMBED_MODEL") or "intfloat/e5-small-v2"
        arr = _embed_hf(texts, model, progress_cb=progress_cb)
    if normalize_embeddings:
        arr = _norm_rows(arr)
    return arr
