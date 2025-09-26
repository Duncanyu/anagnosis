import os, pathlib, datetime, re
from typing import List, Dict, Iterable, Optional, Union
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from api.services import formula_cls

_LOADED = False
_SESS = None
_TOK = None
_INFO = {"loaded": False, "path": "", "max_len": 512, "backend": "onnxruntime"}
_TARGET_DIR = None

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


def _resolve_target_dir(name: Optional[str] = None, path: Optional[str] = None) -> pathlib.Path:
    if path:
        tgt = pathlib.Path(path)
        if tgt.exists():
            return tgt
    env_path = os.environ.get("FORMULA_SFT_PATH")
    if env_path:
        tgt = pathlib.Path(env_path)
        if tgt.exists():
            return tgt
    if name:
        tgt = pathlib.Path(name)
        if tgt.exists():
            return tgt
    env_name = os.environ.get("FORMULA_SFT_NAME")
    if env_name:
        tgt = pathlib.Path(env_name)
        if tgt.exists():
            return tgt
    return _model_dir()


def _load(target_dir: Optional[Union[pathlib.Path, str]] = None):
    global _LOADED, _SESS, _TOK, _INFO, _TARGET_DIR
    if target_dir is None:
        target_dir = _TARGET_DIR or _model_dir()
    target_dir = pathlib.Path(target_dir)
    if _LOADED and _TARGET_DIR is not None and target_dir.resolve() == pathlib.Path(_TARGET_DIR).resolve():
        return _SESS, _TOK
    onnx_path = target_dir / "model.onnx"
    if not onnx_path.exists():
        _LOADED = False
        _SESS, _TOK = None, None
        _INFO = {
            "loaded": False,
            "path": str(target_dir),
            "max_len": 512,
            "backend": "onnxruntime",
        }
        try:
            print(f"[formula_cls] Warning: ONNX model not found at {onnx_path}; SFT scoring disabled.")
        except Exception:
            pass
        return _SESS, _TOK
    try:
        tok = AutoTokenizer.from_pretrained(str(target_dir))
    except Exception as exc:
        _LOADED = False
        _SESS, _TOK = None, None
        _INFO = {
            "loaded": False,
            "path": str(target_dir),
            "max_len": 512,
            "backend": "onnxruntime",
            "error": str(exc),
        }
        try:
            print(f"[formula_cls] Warning: failed to load tokenizer from {target_dir}: {exc}")
        except Exception:
            pass
        return _SESS, _TOK
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
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    except Exception as exc:
        _LOADED = False
        _SESS, _TOK = None, None
        _INFO = {
            "loaded": False,
            "path": str(target_dir),
            "max_len": 512,
            "backend": "onnxruntime",
            "error": str(exc),
        }
        try:
            print(f"[formula_cls] Warning: failed to load ONNX session from {onnx_path}: {exc}")
        except Exception:
            pass
        return _SESS, _TOK
    _SESS, _TOK = sess, tok
    _INFO = {
        "loaded": True,
        "path": str(target_dir),
        "max_len": 512,
        "backend": "onnxruntime",
        "threads": thr,
    }
    _TARGET_DIR = str(target_dir)
    _LOADED = True
    return _SESS, _TOK

def cls_info():
    _load()
    return dict(_INFO)

def _prep(texts: List[str], tok: AutoTokenizer, max_len: int):
    enc = tok(texts, return_tensors="np", padding=True, truncation=True, max_length=max_len)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

def classify_texts(texts: List[str], batch_size: int = 32, max_len: int = 512):
    scores = score_texts(texts, batch_size=batch_size, max_len=max_len)
    return ["FORMULA" if s >= 0.5 else "NOT_FORMULA" for s in scores]

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
        for group, max_len_val in ((idx_math, ml_math), (idx_text, ml_text)):
            group_sorted = sorted(group, key=lambda j: len(texts[j]))
            for k in range(0, len(group_sorted), bs_env):
                ids = group_sorted[k:k+bs_env]
                if progress_cb:
                    progress_cb(min(n, done + len(ids)), n)
                batch_txt = [texts[j] for j in ids]
                probs = score_texts(batch_txt, batch_size=len(batch_txt), max_len=max_len_val)
                for j, p in zip(ids, probs):
                    if p >= 0.5:
                        preds[j] = "FORMULA"
                        confs[j] = float(p)
                    else:
                        preds[j] = "NOT_FORMULA"
                        confs[j] = float(1.0 - p)
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

def _ensure_formula_labels(chunks, progress_cb=None):
    if not chunks:
        return chunks
    need = [c for c in chunks if "is_formula" not in c]
    if need:
        formula_cls.classify_chunks(need, progress_cb=progress_cb)
    return chunks


def _clean_iterable(texts: Iterable[str]) -> List[str]:
    cleaned = []
    for t in texts:
        s = re.sub(r"\s+", " ", (t or "")).strip()
        cleaned.append(s[:4000])
    return cleaned


def _score_probabilities(texts: Iterable[str], batch_size: int = 64, max_len: int = 256) -> List[float]:
    cleaned = _clean_iterable(texts)
    if not cleaned:
        return []
    sess, tok = _load()
    if sess is None or tok is None:
        return [0.5 for _ in cleaned]
    probs_out: List[float] = []
    bs = max(1, batch_size)
    for i in range(0, len(cleaned), bs):
        chunk = cleaned[i:i+bs]
        try:
            feeds = _prep(chunk, tok, max_len)
            logits = sess.run(["logits"], feeds)[0]
            probs = _softmax(logits)
        except Exception as exc:
            try:
                print(f"[formula_cls] Warning: span scoring failed: {exc}")
            except Exception:
                pass
            probs_out.extend([0.5 for _ in chunk])
            continue
        for logit_vec, prob_vec in zip(logits, probs):
            if prob_vec.shape[-1] == 1:
                val = float(1.0 / (1.0 + np.exp(-float(logit_vec[0]))))
            else:
                val = float(prob_vec[1])
            probs_out.append(val)
    return probs_out


def score_texts(texts: Iterable[str], batch_size: int = 64, max_len: int = 256) -> List[float]:
    bs = _env_int("FORMULA_SFT_BATCH_SIZE", batch_size)
    ml = _env_int("FORMULA_SFT_MAX_LEN", max_len)
    return _score_probabilities(texts, batch_size=bs, max_len=ml)


def score_spans(spans: Iterable[str], batch_size: int = 64, max_len: int = 256) -> List[float]:
    return score_texts(spans, batch_size=batch_size, max_len=max_len)


def score(text: str, max_len: int = 256) -> float:
    scores = score_texts([text], batch_size=1, max_len=max_len)
    return float(scores[0]) if scores else 0.5


def predict(texts: Iterable[str], batch_size: int = 64, max_len: int = 256) -> List[str]:
    probs = score_texts(texts, batch_size=batch_size, max_len=max_len)
    return ["FORMULA" if p >= 0.5 else "NOT_FORMULA" for p in probs]


def predict_proba(texts: Iterable[str], batch_size: int = 64, max_len: int = 256) -> List[List[float]]:
    probs = score_texts(texts, batch_size=batch_size, max_len=max_len)
    return [[float(1.0 - p), float(p)] for p in probs]


def ensure_span_scorer(name: Optional[str] = None, path: Optional[str] = None):
    target = _resolve_target_dir(name=name, path=path)
    global _TARGET_DIR
    _TARGET_DIR = str(target)
    try:
        _load(target)
    except Exception:
        pass
    return dict(_INFO)


def ensure_scorer(name: Optional[str] = None, path: Optional[str] = None):
    return ensure_span_scorer(name=name, path=path)
