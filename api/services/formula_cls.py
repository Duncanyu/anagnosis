import os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_TOK = None
_MDL = None
_DEVICE = None

def _load():
    global _TOK, _MDL, _DEVICE
    if _MDL is not None:
        return _TOK, _MDL, _DEVICE
    path = os.getenv("FORMULA_CLS_PATH", "artifacts/models/formula_cls")
    _TOK = AutoTokenizer.from_pretrained(path)
    if torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")
    _MDL = AutoModelForSequenceClassification.from_pretrained(path).to(_DEVICE).eval()
    return _TOK, _MDL, _DEVICE

@torch.no_grad()
def score(texts):
    tok, mdl, dev = _load()
    batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
    logits = mdl(**batch).logits
    prob = torch.softmax(logits, dim=-1)[:, 1]
    return prob.detach().cpu().numpy().tolist()