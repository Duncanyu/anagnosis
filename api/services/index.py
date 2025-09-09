import json, pathlib
import numpy as np
import faiss
from api.services.embed import embed_texts
from api.core.config import load_config

ART = pathlib.Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
INDEX_PATH = ART / "index.faiss"
META_PATH = ART / "meta.json"
CHUNKS_PATH = ART / "chunks.jsonl"

_index = None
_dim = None

def _meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}

def _write_meta(m):
    META_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

def clear_index():
    global _index, _dim
    for p in [INDEX_PATH, META_PATH, CHUNKS_PATH]:
        if p.exists():
            p.unlink()
    _index, _dim = None, None

def _reset_index(dim, backend, model):
    global _index, _dim
    clear_index()
    m = {"dim": dim, "backend": backend, "model": model, "count": 0}
    _write_meta(m)
    _index = faiss.IndexFlatIP(dim)
    faiss.write_index(_index, str(INDEX_PATH))
    _dim = dim

def _ensure_index(dim):
    global _index, _dim
    cfg = load_config()
    backend = (cfg.get("EMBED_BACKEND") or "hf").lower()
    model = "text-embedding-3-small" if backend == "openai" else "intfloat/e5-small-v2"
    if INDEX_PATH.exists():
        if _index is None:
            _index = faiss.read_index(str(INDEX_PATH)); _dim = _index.d
        if _index.d != dim:
            _reset_index(dim, backend, model); return
        m = _meta()
        if not m or m.get("dim") != dim:
            _reset_index(dim, backend, model); return
        return
    _reset_index(dim, backend, model)

def add_chunks(chunks):
    texts, kept = [], []
    for c in chunks:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        if not c.get("doc_name"):
            c["doc_name"] = "Unknown.pdf"
        texts.append(t)
        kept.append(c)
    if not texts:
        return []
    vecs = embed_texts(texts)
    dim = int(vecs.shape[1])
    _ensure_index(dim)
    index = _index if _index is not None else faiss.read_index(str(INDEX_PATH))
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    m = _meta()
    start = int(m.get("count", 0))
    ids = list(range(start, start + len(kept)))
    m["count"] = start + len(kept)
    _write_meta(m)
    with CHUNKS_PATH.open("a", encoding="utf-8") as f:
        for rid, c in zip(ids, kept):
            obj = dict(c); obj["rid"] = rid
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return ids


def _all_chunks():
    out = []
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
    return out

def search(text, k=5):
    if not INDEX_PATH.exists():
        return []
    vec = embed_texts([text])
    dim = int(vec.shape[1])
    idx = faiss.read_index(str(INDEX_PATH))
    if idx.d != dim:
        return []
    D, I = idx.search(vec, k)
    rows = _all_chunks()
    hits = []
    for j, s in zip(I[0], D[0]):
        if j < 0 or j >= len(rows):
            continue
        hits.append((float(s), rows[j]))
    return hits
