from __future__ import annotations

import pathlib

PATH = pathlib.Path("artifacts") / "doc_summaries.jsonl"


def prune_by_doc_names(names):
    names = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    if not names or not PATH.exists():
        return 0
    try:
        lines = PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0
    keep = []
    removed = 0
    for ln in lines:
        hit = False
        low = ln.lower()
        for n in names:
            base = n
            try:
                base = pathlib.Path(n).name.lower()
            except Exception:
                pass
            if base and base in low:
                hit = True
                break
        if hit:
            removed += 1
        else:
            keep.append(ln)
    try:
        PATH.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    except Exception:
        pass
    return removed

