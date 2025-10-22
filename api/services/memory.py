import json, pathlib, time, os

ART_DIR = pathlib.Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

def _mem_path(user_id: str = None) -> pathlib.Path:
    if user_id:
        return ART_DIR / f"memory_{str(user_id)}.jsonl"
    return ART_DIR / "memory.jsonl"

def _approx_tokens(s):
    return max(1, len(s) // 4)

def load_recent(limit_tokens=1200, *, user_id: str = None):
    path = _mem_path(user_id)
    turns = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "q" in obj or "a" in obj:
                        turns.append({"q": obj.get("q"), "a": obj.get("a")})
                except Exception:
                    continue
    sel, budget = [], int(limit_tokens or 0)
    for t in reversed(turns):
        cost = _approx_tokens((t.get("q") or "") + (t.get("a") or ""))
        if budget and budget - cost < 0:
            break
        sel.append(t)
        if budget:
            budget -= cost
    return list(reversed(sel))

def append_turn(q, a, *, user_id: str = None):
    path = _mem_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "q": q, "a": a}, ensure_ascii=False) + "\n")

def prune_file(max_mb=50, *, user_id: str = None):
    path = _mem_path(user_id)
    if not path.exists():
        return
    limit = int(max_mb) * 1024 * 1024
    size = path.stat().st_size
    if size <= limit:
        return
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    keep, cur = [], 0
    for line in reversed(lines):
        b = len(line.encode("utf-8")) + 1
        cur += b
        keep.append(line)
        if cur >= limit:
            break
    keep = list(reversed(keep))
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))

def clear(*, user_id: str = None):
    path = _mem_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        pass

def prune_by_doc_names(doc_names, *, user_id: str = None) -> int:
    """Remove memory turns that reference any of the given doc names in citations.

    A simple heuristic: drop lines whose question or answer contains
    "[<DocName> p." (case-insensitive, base filename match only).
    Returns number of removed turns.
    """
    names = {str(n).strip().lower() for n in (doc_names or []) if str(n).strip()}
    if not names:
        return 0
    path = _mem_path(user_id)
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except Exception:
        return 0
    keep, removed = [], 0
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            keep.append(ln)
            continue
        txt = ((obj.get("q") or "") + "\n" + (obj.get("a") or "")).lower()
        hit = False
        for n in names:
            base = n
            # Compare against base filename only
            if "/" in base or "\\" in base:
                import pathlib as _p
                base = _p.Path(base).name
            if base and f"[{base} p" in txt:
                hit = True
                break
        if hit:
            removed += 1
        else:
            keep.append(ln)
    try:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
    except Exception:
        pass
    return removed
