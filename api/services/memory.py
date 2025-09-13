import json, pathlib, time, os

MEM_PATH = pathlib.Path("artifacts") / "memory.jsonl"
MEM_PATH.parent.mkdir(parents=True, exist_ok=True)

def _approx_tokens(s):
    return max(1, len(s) // 4)

def load_recent(limit_tokens=1200):
    turns = []
    if MEM_PATH.exists():
        with MEM_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "q" in obj or "a" in obj:
                        turns.append({"q": obj.get("q"), "a": obj.get("a")})
                except:
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

def append_turn(q, a):
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "q": q, "a": a}, ensure_ascii=False) + "\n")

def prune_file(max_mb=50):
    if not MEM_PATH.exists():
        return
    limit = int(max_mb) * 1024 * 1024
    size = MEM_PATH.stat().st_size
    if size <= limit:
        return
    with MEM_PATH.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    keep, cur = [], 0
    for line in reversed(lines):
        b = len(line.encode("utf-8")) + 1
        cur += b
        keep.append(line)
        if cur >= limit:
            break
    keep = list(reversed(keep))
    with MEM_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
        
def clear():
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEM_PATH.open("w", encoding="utf-8") as f:
        pass
