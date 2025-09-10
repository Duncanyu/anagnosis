import json, pathlib, re, math
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

def add_chunks(chunks, progress_cb=None):
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
    vecs = embed_texts(texts, progress_cb=progress_cb)
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

def _tok(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [w for w in text.split() if w]

def _bm25_scores(query, docs, k1=1.5, b=0.75):
    q = _tok(query)
    if not q:
        return np.zeros(len(docs), dtype=float)
    toks = [_tok(d.get("text","")) for d in docs]
    N = len(docs)
    df = {}
    for ts in toks:
        for w in set(ts):
            df[w] = df.get(w,0)+1
    idf = {}
    for w in set(q):
        n = df.get(w,0)
        idf[w] = math.log(1 + (N - n + 0.5)/(n + 0.5)) if N>0 else 0.0
    avgdl = sum(len(ts) for ts in toks)/N if N>0 else 0.0
    scores = np.zeros(N, dtype=float)
    for i, ts in enumerate(toks):
        dl = len(ts) or 1
        tf = {}
        for w in ts:
            tf[w]=tf.get(w,0)+1
        s = 0.0
        for w in q:
            if w not in idf:
                continue
            f = tf.get(w,0)
            denom = f + k1*(1 - b + b*dl/(avgdl or 1))
            s += idf[w] * (f*(k1+1))/(denom or 1)
        scores[i]=s
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores

def _reconstruct_batch(index, ids):
    vecs = []
    rec = getattr(index, "reconstruct", None)
    if rec is None:
        return None
    for i in ids:
        v = np.array(rec(int(i)), dtype="float32")
        vecs.append(v)
    return np.vstack(vecs) if vecs else None

def _mmr_select_vectors(qv, dvs, base_scores, k=5, lambda_weight=0.7):
    k = min(k, len(dvs))
    if k <= 0:
        return []
    qv = qv.astype("float32")
    dvs = dvs.astype("float32")
    sims_q = (qv @ dvs.T).flatten()
    chosen = []
    pool = list(range(len(dvs)))
    while pool and len(chosen) < k:
        if not chosen:
            i = int(np.argmax(lambda_weight * base_scores + (1 - lambda_weight) * sims_q))
            chosen.append(i)
            pool.remove(i)
            continue
        best_i, best_score = None, -1e9
        for i in pool:
            sim_to_sel = 0.0
            if chosen:
                sel = dvs[chosen]
                sim_to_sel = float(np.max(sel @ dvs[i:i+1].T))
            score = lambda_weight * base_scores[i] - (1 - lambda_weight) * sim_to_sel
            if score > best_score:
                best_score, best_i = score, i
        chosen.append(best_i)
        pool.remove(best_i)
    return chosen

def search(text, k=5):
    if not INDEX_PATH.exists():
        return []
    qv = embed_texts([text])
    idx = faiss.read_index(str(INDEX_PATH))
    pool = max(128, k)
    D, I = idx.search(qv, pool)
    rows_all = _all_chunks()
    cand_rows, cand_ids, emb_scores = [], [], []
    for j, s in zip(I[0], D[0]):
        if j < 0 or j >= len(rows_all):
            continue
        cand_rows.append(rows_all[j])
        cand_ids.append(int(j))
        emb_scores.append(float(max(0.0, s)))
    if not cand_rows:
        return []
    emb_scores = np.array(emb_scores, dtype=float)
    if emb_scores.max() > 0:
        emb_scores = emb_scores / emb_scores.max()
    bm25 = _bm25_scores(text, cand_rows)
    hybrid = 0.7 * emb_scores + 0.3 * bm25

    dvs = _reconstruct_batch(idx, cand_ids)
    if dvs is None:
        texts = [r.get("text","") for r in cand_rows]
        try:
            dvs = embed_texts(texts)
        except Exception:
            order = np.argsort(-hybrid)[:k]
            return [(float(hybrid[i]), cand_rows[i]) for i in order]

    order_local = _mmr_select_vectors(qv[0], dvs, hybrid, k=k, lambda_weight=0.75)
    out = []
    for i in order_local:
        out.append((float(hybrid[i]), cand_rows[i]))
    return out