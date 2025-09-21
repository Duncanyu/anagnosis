import os, pathlib, datetime, re
from typing import List, Dict
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

_LOADED = False
_SESS = None
_TOK = None
_INFO = {"loaded": False, "path": "", "max_len": 512, "backend": "onnxruntime"}

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _has_math_surface(t: str) -> bool:
    if not t:
        return False
    return bool(_MATH_TOKENS.search(t))

_ISBN13 = re.compile(r"\b97[89][\-\s]?\d{1,5}[\-\s]?\d+[\-\s]?\d+[\-\s]?\d\b")
_ISBN10 = re.compile(r"\b(?:\d[\-\s]?){9}[\dX]\b")
_MATH_TOKENS = re.compile(r"(=|\+|−|\-|×|÷|/|\^|\\|∑|∫|√|≤|≥|<|>|≈|≠|→|⇒|⇔|∧|∨|¬|⊕|⊢|⊨|⊥|⊤|∀|∃|∴|∵|∈|∉|∩|∪|∂|Δ|∇|\b(sin|cos|tan|log|ln|exp|lim)\b|d/dx|dy/dx|dx|dt|\bfor all\b|\bthere exists\b)", re.IGNORECASE)

def _veto_not_formula(t: str) -> bool:
    s = t.strip()
    if not s:
        return True
    if "isbn" in s.lower():
        return True
    if _ISBN13.search(s) or _ISBN10.search(s):
        return True
    if not _MATH_TOKENS.search(s):
        return True
    return False

def _model_dir():
    p = os.environ.get("FORMULA_MINILM_DIR", "artifacts/models/formula_minilm")
    return pathlib.Path(p)

def _load():
    global _LOADED, _SESS, _TOK, _INFO
    if _LOADED:
        return _SESS, _TOK
    d = _model_dir()
    onnx_path = d / "model.onnx"
    tok = AutoTokenizer.from_pretrained(str(d))
    so = ort.SessionOptions()
    thr = _env_int("FORMULA_CLS_THREADS", 0)
    if thr > 0:
        so.intra_op_num_threads = thr
        so.inter_op_num_threads = thr
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    except Exception:
        pass
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    _SESS, _TOK = sess, tok
    _INFO = {"loaded": True, "path": str(d), "max_len": 512, "backend": "onnxruntime", "threads": thr}
    _LOADED = True
    return _SESS, _TOK

def cls_info():
    _load()
    return dict(_INFO)

def _prep(texts: List[str], tok: AutoTokenizer, max_len: int):
    enc = tok(texts, return_tensors="np", padding=True, truncation=True, max_length=max_len)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

def classify_texts(texts: List[str], batch_size: int = 32, max_len: int = 512):
    sess, tok = _load()
    if sess is None or tok is None:
        return ["NOT_FORMULA" for _ in texts]
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        feeds = _prep(chunk, tok, max_len)
        logits = sess.run(["logits"], feeds)[0]
        probs = _softmax(logits)
        labs = np.argmax(probs, axis=-1)
        for j in range(len(chunk)):
            out.append("FORMULA" if int(labs[j]) == 1 else "NOT_FORMULA")
    return out

def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    s = e / np.sum(e, axis=-1, keepdims=True)
    return s

def classify_chunks(chunks: List[Dict], progress_cb=None, batch_size: int = 64, max_len: int = 512):
    if not chunks:
        return
    texts = []
    for c in chunks:
        t = c.get("text") or ""
        t = re.sub(r"\s+", " ", t).strip()[:4000]
        texts.append(t)
    n = len(texts)
    preds = ["NOT_FORMULA"] * n
    confs = [0.5] * n
    sess, tok = _load()
    if sess is None or tok is None:
        pass
    else:
        bs_env = _env_int("FORMULA_CLS_BATCH_SIZE", batch_size)
        ml_math = _env_int("FORMULA_CLS_MAX_TOKENS_MATH", 384)
        ml_text = _env_int("FORMULA_CLS_MAX_TOKENS_TEXT", 256)
        idx_math = [i for i,t in enumerate(texts) if _has_math_surface(t)]
        idx_text = [i for i,t in enumerate(texts) if not _has_math_surface(t)]
        done = 0
        for group, max_len in ((idx_math, ml_math), (idx_text, ml_text)):
            group_sorted = sorted(group, key=lambda j: len(texts[j]))
            for k in range(0, len(group_sorted), bs_env):
                ids = group_sorted[k:k+bs_env]
                if progress_cb:
                    progress_cb(min(n, done + len(ids)), n)
                batch_txt = [texts[j] for j in ids]
                feeds = _prep(batch_txt, tok, max_len)
                logits = sess.run(["logits"], feeds)[0]
                probs = _softmax(logits)
                labs = np.argmax(probs, axis=-1)
                for j, lab, p in zip(ids, labs, probs):
                    if int(lab) == 1:
                        preds[j] = "FORMULA"
                        confs[j] = float(p[1])
                    else:
                        preds[j] = "NOT_FORMULA"
                        confs[j] = float(1.0 - p[1])
                done += len(ids)
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    tag = _INFO["backend"] + ":" + _INFO.get("path", "") if _INFO.get("loaded") else "disabled"
    for c, txt, lab, p in zip(chunks, texts, preds, confs):
        veto = _veto_not_formula(txt)
        is_form = (lab == "FORMULA" and not veto)
        c["is_formula"] = bool(is_form)
        c["formula_label"] = "FORMULA" if is_form else "NOT_FORMULA"
        c["formula_model"] = tag
        c["formula_timestamp"] = ts
        c["formula_confidence"] = float(p)
    return chunks