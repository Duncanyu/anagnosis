import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
yaml = __import__("yaml")
from sklearn.metrics import classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import inspect

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_table, train_valid_split  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_table, train_valid_split
    from ..utils.logging import banner, log_kv


@dataclass
class EncodedDataset:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def encode(df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int) -> EncodedDataset:
    texts = df["chunk_text"].astype(str).tolist()
    labels = df["label"].map(label2id).astype(int).tolist()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    return EncodedDataset(enc["input_ids"], enc["attention_mask"], np.array(labels, dtype="int64"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chunk label classifier")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    df = load_table(Path(cfg["train_file"]).expanduser())
    if "label_list" in cfg:
        label_list: List[str] = cfg["label_list"]
    else:
        label_list = sorted(df["label"].dropna().unique())
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    split = train_valid_split(df, valid_frac=cfg.get("valid_frac", 0.1), seed=cfg.get("seed", 21))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    max_len = cfg.get("max_length", 256)
    train_ds = encode(split.train, tokenizer, label2id, max_len)
    valid_ds = encode(split.valid, tokenizer, label2id, max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = Path(cfg.get("output_dir", "./outputs/chunk_classifier")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": cfg.get("lr", 3e-5),
        "per_device_train_batch_size": cfg.get("batch_size", 32),
        "per_device_eval_batch_size": cfg.get("batch_size", 32),
        "num_train_epochs": cfg.get("epochs", 4),
        "logging_steps": cfg.get("logging_steps", 100),
        "warmup_ratio": cfg.get("warmup_ratio", 0.1),
        "seed": cfg.get("seed", 21),
        "fp16": (cfg.get("mixed_precision", "fp16") == "fp16"),
        "load_best_model_at_end": True,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    eval_name = None
    if "evaluation_strategy" in sig.parameters:
        eval_name = "evaluation_strategy"
    elif "eval_strategy" in sig.parameters:
        eval_name = "eval_strategy"

    if eval_name:
        training_kwargs[eval_name] = cfg.get("evaluation_strategy", cfg.get("eval_strategy", "epoch"))
    if "save_strategy" in sig.parameters:
        training_kwargs["save_strategy"] = cfg.get("save_strategy", "epoch")
    if "metric_for_best_model" in sig.parameters:
        training_kwargs["metric_for_best_model"] = "macro_f1"
    if "greater_is_better" in sig.parameters:
        training_kwargs["greater_is_better"] = True

    if not eval_name:
        training_kwargs["load_best_model_at_end"] = False

    training_args = TrainingArguments(**training_kwargs)

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        report = classification_report(labels, preds, target_names=[id2label[i] for i in range(len(label_list))], output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]
        return {"accuracy": report["accuracy"], "macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    banner("training")
    trainer.train()

    banner("evaluation")
    metrics = trainer.evaluate()
    log_kv(**metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
