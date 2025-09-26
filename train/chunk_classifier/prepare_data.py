import argparse
import json
import random
from pathlib import Path
from typing import Dict

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_jsonl  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_jsonl
    from ..utils.logging import banner, log_kv

LABELS = ["definition", "theorem", "rule", "example", "exercise"]


def filter_chunk(row: Dict, keep_math: bool) -> bool:
    if keep_math and not row.get("has_math", False):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample chunks for manual labeling")
    parser.add_argument("chunk_file", type=Path)
    parser.add_argument("out_file", type=Path)
    parser.add_argument("--keep-math", action="store_true", help="Restrict to chunks with math heuristics")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [row for row in load_jsonl(args.chunk_file) if filter_chunk(row, args.keep_math)]
    rng.shuffle(rows)
    rows = rows[: args.sample]

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {
                "chunk_text": row.get("text", ""),
                "doc_name": row.get("doc_name"),
                "page": row.get("page_start"),
                "label": "",
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    banner("sample ready")
    log_kv(samples=len(rows), output=str(args.out_file), labels="/".join(LABELS))


if __name__ == "__main__":
    main()
