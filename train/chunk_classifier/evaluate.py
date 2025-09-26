import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_table  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_table
    from ..utils.logging import banner, log_kv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunk classifier")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("data_file", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    df = load_table(Path(args.data_file))

    texts = df["chunk_text"].astype(str).tolist()
    labels = df["label"].tolist()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    outputs = model(**enc)
    preds = outputs.logits.argmax(dim=-1).numpy()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    pred_labels = [id2label[i] for i in preds]

    report = classification_report(labels, pred_labels, output_dict=True, zero_division=0)
    log_kv(accuracy=report["accuracy"], macro_f1=report["macro avg"]["f1-score"])


if __name__ == "__main__":
    main()
