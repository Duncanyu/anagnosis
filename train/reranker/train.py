import argparse
from pathlib import Path

import pandas as pd
yaml = __import__("yaml")
from sentence_transformers import CrossEncoder, InputExample

if __package__ is None or __package__ == "":
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from utils.dataset import load_table, train_valid_split  # type: ignore
    from utils.logging import banner, log_kv  # type: ignore
else:
    from ..utils.dataset import load_table, train_valid_split
    from ..utils.logging import banner, log_kv


def to_examples(df: pd.DataFrame) -> list[InputExample]:
    ex = []
    for _, row in df.iterrows():
        q = str(row["question"])
        chunk = str(row["chunk_text"])
        score = float(row.get("score", 0.0))
        ex.append(InputExample(texts=[q, chunk], label=score))
    return ex


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cross-encoder reranker")
    parser.add_argument("config", type=Path, help="YAML config path")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_path = Path(cfg["train_file"]).expanduser()
    df = load_table(data_path)
    split = train_valid_split(df, valid_frac=cfg.get("valid_frac", 0.1), seed=cfg.get("seed", 42))

    model = CrossEncoder(cfg["model_name"], num_labels=1, max_length=cfg.get("max_length", 256))

    train_samples = to_examples(split.train)
    valid_samples = to_examples(split.valid)

    output_dir = Path(cfg.get("output_dir", "./outputs/reranker")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("training")
    epochs = cfg.get("epochs", 3)
    batch_size = cfg.get("batch_size", 32)
    warmup = cfg.get("warmup_ratio", 0.1)
    use_amp = (cfg.get("mixed_precision", "fp16") == "fp16")

    try:
        model.fit(
            train_samples,
            evaluator=None,
            epochs=epochs,
            batch_size=batch_size,
            warmup_ratio=warmup,
            output_path=str(output_dir),
            show_progress_bar=True,
            use_amp=use_amp,
        )
    except TypeError:
        model.fit(train_samples)

    banner("validation")
    preds = model.predict([[ex.texts[0], ex.texts[1]] for ex in valid_samples])
    out = add_predictions(split.valid, preds)
    out_path = output_dir / "valid_predictions.csv"
    out.to_csv(out_path, index=False)
    log_kv(validation_predictions=str(out_path))


def add_predictions(df: pd.DataFrame, preds) -> pd.DataFrame:
    df = df.copy()
    df["score_pred"] = preds
    return df


if __name__ == "__main__":
    main()
