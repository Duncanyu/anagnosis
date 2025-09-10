import json, pathlib, re

ALIASES_PATH = pathlib.Path("artifacts") / "aliases.json"
ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_aliases():
    if ALIASES_PATH.exists():
        try:
            return json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_aliases(d):
    ALIASES_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def apply_aliases(text, aliases):
    if not aliases:
        return text
    out = text
    for src, dst in aliases.items():
        if not src or src == dst:
            continue
        pat = re.compile(rf"\b{re.escape(src)}\b")
        out = pat.sub(dst, out)
    return out

def _norm_token(s):
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1]
    return s.strip()

def maybe_learn_alias(utterance):
    u = utterance.strip()
    found = []
    r1 = re.findall(r"(?:refer to|call)\s+([A-Za-z0-9_.+\-]+)\s+(?:as|=)\s+(['\"].+?['\"]|[A-Za-z0-9_.+\-]+)", u, flags=re.I)
    r2 = re.findall(r"use\s+(['\"].+?['\"]|[A-Za-z0-9_.+\-]+)\s+instead of\s+(['\"].+?['\"]|[A-Za-z0-9_.+\-]+)", u, flags=re.I)
    aliases = load_aliases()
    for a, b in r1:
        src = _norm_token(a)
        dst = _norm_token(b)
        if src and dst and src != dst:
            aliases[src] = dst
            found.append((src, dst))
    for a, b in r2:
        dst = _norm_token(a)
        src = _norm_token(b)
        if src and dst and src != dst:
            aliases[src] = dst
            found.append((src, dst))
    if found:
        save_aliases(aliases)
    return found