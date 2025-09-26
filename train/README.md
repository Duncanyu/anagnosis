# Training Utilities

This directory hosts scripts and configuration templates for training the SFT components that plug into the RAG pipeline:

1. `reranker` – question/chunk relevance scorer (MiniLM cross-encoder).
2. `chunk_classifier` – chunk semantic labeler (definition/theorem/example/exercise).

Each module is self-contained so you can copy the folder to a GPU machine, install the listed requirements, and launch training.

```
train/
  README.md
  requirements.txt
  data/
    schemas/
      reranker.json
      chunk_classifier.json
  reranker/
    config_example.yaml
    auto_generate_qa.py
    prepare_data.py
    train.py
    evaluate.py
  chunk_classifier/
    config_example.yaml
    auto_label.py
    prepare_data.py
    train.py
    evaluate.py
  utils/
    dataset.py
    logging.py
    metrics.py
```

The scripts assume access to HuggingFace models/datasets and run with PyTorch + Accelerate. Update the configs with your storage paths before launching on the cloud GPU.

## Quickstart: build training data

1. **Export raw chunks** – After ingesting PDFs, copy the relevant `artifacts/chunks_*.jsonl` into `train/data/raw/` (or pass the path directly to the scripts below).
2. **Auto-label semantic classes** – Run `python3 chunk_classifier/auto_label.py artifacts/chunks_*.jsonl data/chunk_auto_labels.jsonl`. This uses heading/section heuristics to assign labels (definition/theorem/rule/example/exercise). Review/edit the JSONL to keep confident rows, then use it as input for `chunk_classifier/train.py`.
3. **Generate synthetic QA pairs** – Run `python3 reranker/auto_generate_qa.py artifacts/chunks_*.jsonl data/qa_auto.jsonl`. This produces entries with fields `question`, `positives`, `negatives`. Feed that into `reranker/prepare_data.py` to expand positives/negatives into the flat training rows required by `train.py`.
4. **Manual refinement (optional)** – Append or edit the auto-generated JSONL files to inject high-quality questions or relabel ambiguous chunks. The JSON schemas under `data/schemas/` describe the accepted structure.

With these helpers you can bootstrap both datasets quickly, then iterate on accuracy by mixing in hand-checked examples from additional courses.
