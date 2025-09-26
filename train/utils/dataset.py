import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


def load_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = load_jsonl(path)
        return pd.DataFrame(rows)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported dataset format: {path}")


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame


def train_valid_split(df: pd.DataFrame, valid_frac: float = 0.1, seed: int = 13) -> Split:
    if not 0.0 < valid_frac < 1.0:
        raise ValueError("valid_frac must be between 0 and 1")
    rng = random.Random(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    cut = int(len(idx) * (1.0 - valid_frac))
    train_idx = idx[:cut]
    valid_idx = idx[cut:]
    return Split(train=df.loc[train_idx].reset_index(drop=True), valid=df.loc[valid_idx].reset_index(drop=True))


def batch_iter(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
