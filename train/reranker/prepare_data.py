import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_jsonl  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_jsonl
    from ..utils.logging import banner, log_kv


def load_pool(chunk_path: Path) -> List[str]:
    rows = load_jsonl(chunk_path)
    pool = []
    for row in rows:
        text = row.get("text") or row.get("chunk_text") or ""
        if text:
            pool.append(text.strip())
    return pool


def build_examples(entry: Dict, pool: List[str], negatives_per_pos: int, rng: random.Random) -> List[Dict]:
    question = entry["question"].strip()
    positives = [p.strip() for p in entry.get("positives", []) if p.strip()]
    negatives = [n.strip() for n in entry.get("negatives", []) if n.strip()]
    examples = []
    for pos in positives:
        examples.append({"question": question, "chunk_text": pos, "score": 1.0})
        chosen = negatives[:negatives_per_pos]
        negatives = negatives[negatives_per_pos:]
        while len(chosen) < negatives_per_pos and pool:
            cand = rng.choice(pool)
            if cand not in positives and cand not in chosen:
                chosen.append(cand)
        for neg in chosen:
            examples.append({"question": question, "chunk_text": neg, "score": 0.0})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reranker JSONL from QA annotations.")
    parser.add_argument("qa_file", type=Path, help="JSONL with fields: question, positives (list), negatives (optional list)")
    parser.add_argument("out_file", type=Path, help="Destination JSONL file")
    parser.add_argument("--chunk-source", type=Path, default=None, help="Optional chunk JSONL to sample negatives from")
    parser.add_argument("--neg-per-pos", type=int, default=3, help="Number of negatives per positive example")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    qa_rows = load_jsonl(args.qa_file)
    pool: List[str] = []
    if args.chunk_source:
        banner("loading negative pool")
        pool = load_pool(args.chunk_source)
        log_kv(total_candidates=len(pool))

    out_rows: List[Dict] = []
    for entry in qa_rows:
        if "question" not in entry or "positives" not in entry:
            continue
        ex = build_examples(entry, pool, args.neg_per_pos, rng)
        out_rows.extend(ex)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    banner("done")
    log_kv(output=str(args.out_file), rows=len(out_rows))


if __name__ == "__main__":
    main()
