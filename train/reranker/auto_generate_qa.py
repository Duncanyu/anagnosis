import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_jsonl  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_jsonl
    from ..utils.logging import banner, log_kv

_DEFINITION_HEAD_RX = re.compile(r"definition", re.I)
_THEOREM_HEAD_RX = re.compile(r"theorem|lemma|corollary|proposition", re.I)
_RULE_HEAD_RX = re.compile(r"rule|law|identity|formula", re.I)


def last_heading(heading_path: str) -> str:
    parts = [p.strip() for p in heading_path.split(">") if p.strip()]
    return parts[-1] if parts else ""


def build_question(row: Dict) -> Optional[str]:
    heading = row.get("heading_path") or ""
    title = last_heading(heading)
    text = (row.get("text") or "").strip()
    if not title and len(text.split()) < 8:
        return None

    heading_lower = heading.lower()
    if _DEFINITION_HEAD_RX.search(heading_lower) or text.lower().startswith("definition"):
        subject = title or text.split(".")[0]
        return f"What is the definition of {subject}?"
    if _THEOREM_HEAD_RX.search(heading_lower):
        subject = title or "this theorem"
        return f"State the {subject}."
    if _RULE_HEAD_RX.search(heading_lower) or row.get("is_formula"):
        subject = title or "this rule"
        return f"What is the formula for {subject}?"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs from chunk metadata.")
    parser.add_argument("chunk_file", type=Path)
    parser.add_argument("out_file", type=Path, help="Output JSONL with fields question, positives, negatives")
    parser.add_argument("--negatives", type=int, default=4, help="Negatives per question")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.chunk_file)
    pool = [row for row in rows if (row.get("text") or "").strip()]

    qa_rows: List[Dict] = []
    for row in rows:
        question = build_question(row)
        if not question:
            continue
        positive_text = (row.get("text") or "").strip()
        if not positive_text:
            continue
        negatives: List[str] = []
        attempts = 0
        while len(negatives) < args.negatives and attempts < args.negatives * 6:
            cand = rng.choice(pool)
            c_text = (cand.get("text") or "").strip()
            if c_text == positive_text:
                attempts += 1
                continue
            if cand.get("doc_name") == row.get("doc_name") and cand.get("heading_path") == row.get("heading_path"):
                attempts += 1
                continue
            negatives.append(c_text)
            attempts += 1
        qa_rows.append({
            "question": question,
            "positives": [positive_text],
            "negatives": negatives,
            "doc_name": row.get("doc_name"),
            "heading": row.get("heading_path"),
            "page": row.get("page_start"),
        })

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    banner("qa generation complete")
    log_kv(total_chunks=len(rows), generated=len(qa_rows), output=str(args.out_file))


if __name__ == "__main__":
    main()
