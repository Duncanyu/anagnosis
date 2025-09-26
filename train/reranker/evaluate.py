import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_table  # type: ignore
    from utils.metrics import ndcg_at_k  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_table
    from ..utils.metrics import ndcg_at_k
    from ..utils.logging import banner, log_kv


def group_metrics(df: pd.DataFrame, model: CrossEncoder, k: int) -> dict:
    grouped = df.groupby("question")
    ndcgs = []
    for question, group in grouped:
        pairs = [[question, chunk] for chunk in group["chunk_text"].tolist()]
        preds = model.predict(pairs)
        ndcgs.append(ndcg_at_k(preds, group["score"].tolist(), k=k))
    return {f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained reranker")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("data_file", type=Path)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    banner("loading")
    model = CrossEncoder(str(args.model_dir))
    df = load_table(Path(args.data_file))

    metrics = group_metrics(df, model, args.k)
    log_kv(**metrics)


if __name__ == "__main__":
    main()
