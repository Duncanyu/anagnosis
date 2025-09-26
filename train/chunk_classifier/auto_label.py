import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_jsonl  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_jsonl
    from ..utils.logging import banner, log_kv

_DEFINITION_RX = re.compile(r"^(definition\b|let\s+.*\s+be\s+the\s+definition|is\s+defined\s+as)", re.I)
_THEOREM_HEAD_RX = re.compile(r"\b(theorem|lemma|corollary|proposition|axiom)\b", re.I)
_RULE_HEAD_RX = re.compile(r"\b(rule|law|identity|formula)\b", re.I)
_EXAMPLE_HEAD_RX = re.compile(r"\b(example|worked example)\b", re.I)
_EXERCISE_RX = re.compile(r"(exercise|problem|practice|show\s+that|compute|determine|find)\b", re.I)


def choose_label(row: Dict, min_chars: int) -> Optional[str]:
    text = (row.get("text") or row.get("chunk_text") or "").strip()
    if len(text) < min_chars:
        return None

    heading = (row.get("heading_path") or "").lower()
    section = (row.get("section_tag") or "").lower()

    def has(rx: re.Pattern, target: str) -> bool:
        return bool(rx.search(target))

    if has(_THEOREM_HEAD_RX, heading):
        return "theorem"
    if has(_RULE_HEAD_RX, heading):
        return "rule"
    if heading.startswith("definition") or has(_DEFINITION_RX, text):
        return "definition"
    if has(_EXAMPLE_HEAD_RX, heading) or text.lower().startswith("example"):
        return "example"
    if section == "exercises" or has(_EXERCISE_RX, heading) or has(_EXERCISE_RX, text):
        return "exercise"
    if row.get("is_equation") and row.get("has_math") and has(_RULE_HEAD_RX, text):
        return "rule"
    if row.get("is_formula") and row.get("formula_confidence", 0) >= 0.8:
        return "rule"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label chunks with heuristics.")
    parser.add_argument("chunk_file", type=Path, help="Input chunk JSONL (from artifacts)")
    parser.add_argument("out_file", type=Path, help="Where to write labeled JSONL")
    parser.add_argument("--min-chars", type=int, default=60, help="Skip very short chunks")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on rows")
    args = parser.parse_args()

    rows = load_jsonl(args.chunk_file)
    labeled = []
    for row in rows:
        label = choose_label(row, args.min_chars)
        if not label:
            continue
        labeled.append({
            "chunk_text": row.get("text", ""),
            "doc_name": row.get("doc_name"),
            "page": row.get("page_start"),
            "label": label,
        })
        if args.limit and len(labeled) >= args.limit:
            break

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    banner("auto labeling complete")
    log_kv(total_input=len(rows), labeled=len(labeled), output=str(args.out_file))


if __name__ == "__main__":
    main()
