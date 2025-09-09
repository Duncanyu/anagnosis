import json, pathlib
from typing import List, Dict, Tuple
import numpy as np
import faiss
from api.services.embed import embed_texts, get_embedder

ART = pathlib.Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
INDEX_PATH = ART / "index.faiss"
MAP_PATH = ART / "map.json"
CHUNKS_PATH = ART / "chunks.jsonl"

_index = None
_dim = None
_id_counter = 0

def _load_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    return {"next_id": 0}

def _save_map(m: Dict):
    MAP_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

def _open_index(dim: int):
    global _index, _dim
    if _index is not None:
        return _index
    _dim = dim
    if INDEX_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
    else:
        _index = faiss.IndexFlatIP(dim)
    return _index

def add_chunks(chunks: List[Dict]):
    global _id_counter
    m = _load_map()
    _id_counter = int(m.get("next_id", 0))

    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    index = _open_index(vecs.shape[1])

    ids = list(range(_id_counter, _id_counter + len(chunks)))
    _id_counter += len(chunks)
    m["next_id"] = _id_counter
    _save_map(m)

    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    with CHUNKS_PATH.open("a", encoding="utf-8") as f:
        for rid, c in zip(ids, chunks):
            f.write(json.dumps({"rid": rid, **c}, ensure_ascii=False) + "\n")

    return ids

def _load_all_chunks():
    out = []
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
    return out

def search(text: str, k: int = 5):
    vec = embed_texts([text])
    index = _open_index(vec.shape[1])
    D, I = index.search(vec, k)
    chunks = _load_all_chunks()
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        ch = chunks[idx]
        hits.append((float(score), ch))
    return hits
