import json, pathlib, re, math, hashlib, os, time
import numpy as np
import faiss
from api.services.embed import embed_texts, embedding_info
from api.core.config import load_config

ART = pathlib.Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)

_index = None
_dim = None
_ns_key = None
_rows_cache = None
_rows_mtime = 0

def _safe(s):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s or "")

def _ns_from_embed():
    info = embedding_info() if callable(embedding_info) else {}
    backend = (info.get("backend") or (load_config().get("EMBED_BACKEND") or "hf")).lower()
    model = info.get("model") or ("text-embedding-3-small" if backend == "openai" else "intfloat/e5-small-v2")
    tag = f"{backend}|{model}"
    h = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    key = f"{backend}_{_safe(model)}_{h}"
    idx = ART / f"index_{key}.faiss"
    meta = ART / f"meta_{key}.json"
    ch = ART / f"chunks_{key}.jsonl"
    return idx, meta, ch, backend, model, key

def _legacy_paths():
    return ART / "index.faiss", ART / "meta.json", ART / "chunks.jsonl"

def _active_paths():
    idx, meta, ch, backend, model, key = _ns_from_embed()
    l_idx, l_meta, l_ch = _legacy_paths()
    if idx.exists() or meta.exists() or ch.exists():
        return idx, meta, ch, backend, model, key
    if l_idx.exists() or l_meta.exists() or l_ch.exists():
        return l_idx, l_meta, l_ch, backend, model, key
    return idx, meta, ch, backend, model, key

def _ns_only_paths():
    idx, meta, ch, backend, model, key = _ns_from_embed()
    return idx, meta, ch, backend, model, key

def _meta():
    _, meta_path, _, _, _, _ = _active_paths()
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

def _write_meta(m):
    _, meta_path, _, _, _, _ = _active_paths()
    meta_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

def clear_index():
    global _index, _dim, _ns_key
    idx, meta, ch, _, _, _ = _active_paths()
    for p in [idx, meta, ch]:
        if p.exists():
            p.unlink()
    _index, _dim, _ns_key = None, None, None

def _reset_index(dim, backend, model):
    global _index, _dim, _ns_key
    idx, meta, ch, _, _, key = _ns_only_paths()
    for p in [idx, meta, ch]:
        if p.exists():
            p.unlink()
    m = {"dim": dim, "backend": backend, "model": model, "count": 0}
    meta.write_text(json.dumps(m, indent=2), encoding="utf-8")
    _index = faiss.IndexFlatIP(dim)
    faiss.write_index(_index, str(idx))
    _dim = dim
    _ns_key = key

def _ensure_index(dim):
    global _index, _dim, _ns_key
    idx, meta, ch, backend, model, key = _active_paths()
    if _ns_key != key:
        _index, _dim = None, None
        _ns_key = key
    if idx.exists():
        if _index is None:
            _index = faiss.read_index(str(idx)); _dim = _index.d
        if _index.d != dim:
            _reset_index(dim, backend, model); return
        m = _meta()
        if not m or m.get("dim") != dim or m.get("backend") != backend or m.get("model") != model:
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
    idx_path, meta_path, ch_path, _, _, _ = _active_paths()
    index = _index if _index is not None else faiss.read_index(str(idx_path))
    index.add(vecs)
    faiss.write_index(index, str(idx_path))
    m = _meta()
    start = int(m.get("count", 0))
    ids = list(range(start, start + len(kept)))
    m["count"] = start + len(kept)
    _write_meta(m)
    with ch_path.open("a", encoding="utf-8") as f:
        for rid, c in zip(ids, kept):
            obj = dict(c); obj["rid"] = rid
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return ids

def _all_chunks():
    global _rows_cache, _rows_mtime
    _, _, ch_path, _, _, _ = _active_paths()
    if not ch_path.exists():
        _rows_cache, _rows_mtime = [], 0
        return []
    st = ch_path.stat().st_mtime
    if _rows_cache is not None and _rows_mtime == st:
        return _rows_cache
    out = []
    with ch_path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    _rows_cache, _rows_mtime = out, st
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

def _reconstruct_batch(index, ids, progress_cb=None, t_end=None):
    vecs = []
    rec = getattr(index, "reconstruct", None)
    if rec is None:
        return None
    n = len(ids)
    for i, idv in enumerate(ids):
        if t_end is not None and time.time() >= t_end:
            break
        v = np.array(rec(int(idv)), dtype="float32")
        vecs.append(v)
        if progress_cb and (i % 200 == 0 or i == n - 1):
            progress_cb(f"Search: reconstructing vectors {i+1}/{n}")
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

def search(text, k=5, progress_cb=None, timeout_sec=None, pool=None):
    idx_path, _, _, _, _, _ = _active_paths()
    if not idx_path.exists():
        return []
    t_end = None
    if timeout_sec and timeout_sec > 0:
        t_end = time.time() + float(timeout_sec)
    if progress_cb:
        progress_cb("Search: encoding query")
    qv = embed_texts([text])
    if progress_cb:
        progress_cb("Search: loading index")
    idx = faiss.read_index(str(idx_path))
    if pool is None:
        pool = max(128, k)
    if progress_cb:
        progress_cb(f"Search: FAISS top-{pool}")
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
    if progress_cb:
        progress_cb(f"Search: candidates {len(cand_rows)}")
    emb_scores = np.array(emb_scores, dtype=float)
    if emb_scores.max() > 0:
        emb_scores = emb_scores / emb_scores.max()
    if progress_cb:
        progress_cb("Search: BM25 blending")
    bm25 = _bm25_scores(text, cand_rows)
    hybrid = 0.7 * emb_scores + 0.3 * bm25
    if t_end is not None and time.time() >= t_end:
        order = np.argsort(-hybrid)[:k]
        return [(float(hybrid[i]), cand_rows[i]) for i in order]
    if progress_cb:
        progress_cb("Search: reconstructing candidate vectors")
    dvs = _reconstruct_batch(idx, cand_ids, progress_cb=progress_cb, t_end=t_end)
    if dvs is None or (t_end is not None and time.time() >= t_end):
        order = np.argsort(-hybrid)[:k]
        return [(float(hybrid[i]), cand_rows[i]) for i in order]
    if progress_cb:
        progress_cb("Search: MMR selection")
    order_local = _mmr_select_vectors(qv[0], dvs, hybrid, k=k, lambda_weight=0.75)
    out = []
    for i in order_local:
        out.append((float(hybrid[i]), cand_rows[i]))
    return out