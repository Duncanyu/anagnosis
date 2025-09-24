from api.core.config import load_config
from api.services.embed import embed_texts
from api.services.aliases import load_aliases
import os, re, json, time, pathlib, math
import numpy as np
import unicodedata

from api.services import formula_cls

HAS_FCLS = True

def _formula_score_fn():
    try:
        for name in ("score", "score_texts", "score_spans", "predict", "predict_proba"):
            fn = getattr(formula_cls, name, None)
            if callable(fn):
                return fn
    except Exception:
        pass
    return None

_FORMULA_SCORE_FN = _formula_score_fn()

def _maybe_init_sft(progress_cb=None):
    global _FORMULA_SCORE_FN
    if _FORMULA_SCORE_FN is not None:
        return
    try:
        ensure = getattr(formula_cls, "ensure_span_scorer", None) or getattr(formula_cls, "ensure_scorer", None)
        name = os.getenv("FORMULA_SFT_NAME", "")
        path = os.getenv("FORMULA_SFT_PATH", "")
        if ensure:
            try:
                ensure(name=name, path=path)
            except TypeError:
                try:
                    ensure(name or path or "tinybert")
                except Exception:
                    pass
        _FORMULA_SCORE_FN = _formula_score_fn()
        if _FORMULA_SCORE_FN and progress_cb:
            try:
                progress_cb("SFT span scorer ready")
            except Exception:
                pass
    except Exception:
        pass
FORMULA_CLS_ENABLE = True
FORMULA_CLS_ALWAYS_REFRESH = True
FORMULA_P_MIN = float(os.getenv("FORMULA_P_MIN", "0.80"))
FORMULA_MIN_PROB = float(os.getenv("FORMULA_MIN_PROB", "0.45"))
FORMULA_KEEP_FRAC = float(os.getenv("FORMULA_KEEP_FRAC", "0.50"))
FORMULA_MIN_FORMULA_RATIO = float(os.getenv("FORMULA_MIN_FORMULA_RATIO", "0.25"))
FORMULA_MIN_KEEP_PER_PAGE = int(os.getenv("FORMULA_MIN_KEEP_PER_PAGE","3"))
FORMULA_STRONG_DENS = float(os.getenv("FORMULA_STRONG_DENS","0.55"))
ASK_FORMULA_DEBUG = (os.getenv("ASK_FORMULA_DEBUG", "1").lower() in {"1","true","yes","on"})
ASK_CITATION_NOTE = (os.getenv("ASK_CITATION_NOTE", "1").lower() in {"1","true","yes","on"})

TEMPLATE = """You are a study assistant. Answer the question using ONLY the provided context.
Start with a single bold sentence that directly answers the question. Then, if helpful, add short bullet points. Do not start the response with a bullet. Cite sources like [FileName.pdf p.3] or [FileName.pdf p.3–4].
Important terms to anchor on: {terms}
If none of the important terms appear in the context, still answer from the context you have.

Question:
{q}

Context:
{ctx}
"""

DOC_TEMPLATE = """You are a rigorous study assistant for university readings.
Return a comprehensive summary in Markdown with these sections:
## High-level overview
## Section-by-section distilled notes
## Key terms
## Formulas
Use concise bullets and include page citations like [FileName.pdf p.12] or [FileName.pdf p.4–6]. Do not invent facts.

Context:
{ctx}
"""

CANONICAL_FORMULA_TEMPLATE = """You are a formula librarian. From ONLY the provided extracted math spans and snippets, produce a concise list of the CANONICAL formulas, laws, rules, identities, and named theorems relevant to the user’s question. 
Requirements:
- Group by short category headers (e.g., Derivatives, Integrals, Trigonometry, Series, Linear Algebra, Probability, Logic), but only if present.
- For each item: show a compact formula in inline code, then 3–12 words of label/meaning.
- Prefer definitions/theorems/rules over worked examples or exercise prompts.
- Exclude problem statements, instructions (prove/show/compute), and numeric plug‑ins for particular values.
- Keep each item to one line. 8–20 total items is ideal.
- Include page citations in the form [FileName.pdf p.N] pulled from the supplied context next to each item.
- Do not invent formulas. If unsure, omit.

User question:
{q}

Extracted spans (with sources):
{ctx}
"""

SUMMARIES_PATH = pathlib.Path("artifacts") / "doc_summaries.jsonl"
CTX_CHAR_BUDGET = 14000
BATCH_CHAR_BUDGET = int(os.getenv("ASK_BATCH_CHAR_BUDGET", "12000"))
MAX_BATCHES_DEFAULT = int(os.getenv("ASK_MAX_BATCHES", "6"))
TIME_BUDGET_SEC_DEFAULT = int(os.getenv("ASK_TIME_BUDGET_SEC", "120"))
MMR_TOP_N = int(os.getenv("ASK_MMR_TOP_N", "200"))
MMR_ALPHA = float(os.getenv("ASK_MMR_ALPHA", "0.7"))
ASK_EXHAUSTIVE_DEFAULT = (os.getenv("ASK_EXHAUSTIVE","0").lower() in {"1","true","yes","on"})
ASK_CANDIDATES_DEFAULT = int(os.getenv("ASK_CANDIDATES","300"))
RERANKER_NAME_DEFAULT = os.getenv("ASK_RERANKER","off").lower()
RERANK_TOP_N = int(os.getenv("ASK_RERANK_TOP_N","200"))
ASK_RERANK_IN_FORMULA = (os.getenv("ASK_RERANK_IN_FORMULA","1").lower() in {"1","true","yes","on"})
FORMULA_EXHAUSTIVE = (os.getenv("FORMULA_EXHAUSTIVE", "1").lower() in {"1","true","yes","on"})
FORMULA_MAX_PER_PAGE = int(os.getenv("FORMULA_MAX_PER_PAGE", "200"))
FORMULA_Q_WORDS = {"formula","formulas","identity","identities","equation","equations","rule","rules","laws","law"}
FORMULA_POS_TERMS = {"formula","formulas","identity","identities","rule","rules","law","laws","property","properties","theorem","theorems","definition","definitions","summary","key formulas","double-angle","half-angle","sum","difference"}
FORMULA_NEG_TERMS = {"exercise","exercises","problem","problems","example","examples","solution","solutions","answer","answers","practice","review"}
FORMULA_HINT_RX = re.compile(r"(=|≈|≅|≤|≥|≠|±|∝|∑|∏|∫|√|∞|∇|∂|d/dx|\bdy/dx\b|\be\^|\^\s*\w|\|.+\||\b(sin|cos|tan|cot|sec|csc|log|ln|exp)\s*\()", re.I)
PROMPT_PREFIX_RX = re.compile(r"^\s*(determine|how many|prove|show|find|compute|evaluate|solve)\b", re.I)
HEAD_VETO_RX = re.compile(r"(answers to odd|supplementary exercises|references|bibliography)", re.I)
BIB_HINT_RX = re.compile(r"(ISBN|Prentice|Wiley|Springer|Addison|Pearson|McGraw|Elsevier|Englewood|Cliffs|NJ|NY:)", re.I)
YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")
LINE_VETO_RX = re.compile(r"^(we write|often we give|this function is defined|if f\(a\) = b|answers to odd|supplementary exercises)\b", re.I)
EXERCISE_ANY_RX = re.compile(r"\b(determine|how many|prove|show|find|compute|evaluate|solve)\b", re.I)
EQ_SPAN_RX = re.compile(r"[^.;:\n]{1,120}(?:=|≈|≅|≤|≥|≠|∝)[^.;:\n]{1,120}")
FUNC_SPAN_RX = re.compile(r"\b(?:sin|cos|tan|cot|sec|csc|log|ln|exp)\s*\([^)(\n]{1,80}\)", re.I)
DERIV_SPAN_RX = re.compile(r"(?:d\s*[a-zA-Z]\s*/\s*d\s*[a-zA-Z]|∂\s*[a-zA-Z]\s*/\s*∂\s*[a-zA-Z]|dy/dx)", re.I)
INT_SPAN_RX = re.compile(r"∫[^.;:\n]{1,160}")
SUM_SPAN_RX = re.compile(r"(?:∑|∏)[^.;:\n]{1,160}")
ABS_SPAN_RX = re.compile(r"\|[^|\n]{1,160}\|")
POW_SPAN_RX = re.compile(r"\b[a-zA-Z]\s*\^\s*[0-9a-zA-Z]+")
NONFORM_RX = re.compile(r"\b(proof|example|exercise|problem|show that|find|assume|consider|let then|hence|therefore|solution|answer)\b", re.I)
ENUM_RX = re.compile(r"^\s*(?:\(?[a-z]\)|[ivxlcdm]+\)|\d+\)|\(?\d+\))\s+", re.I)
QUESTION_RX = re.compile(r"\?\s*$|^\s*(what|which|find|show|prove|compute|determine)\b", re.I)
LET_ASSIGN_RX = re.compile(r"^\s*(let|suppose)\b", re.I)
SIMPLE_ARITH_RX = re.compile(r"^\s*(?:\d+[^\w\n]{1,3}){1,8}\d+\s*=\s*\d+\s*$")
LINEAR_EX_RX   = re.compile(r"^\s*[a-z]\s*[+\-*/]\s*\d+\s*=\s*\d+\s*$", re.I)
EQUIV_SYMS_RX   = re.compile(r"(↔|⇔|≡)")
QUANT_RX        = re.compile(r"[∀∃]")
SPECIAL_CONST_RX= re.compile(r"(π|phi|φ|\be\b)")
FUNC_RX         = re.compile(r"\b(sin|cos|tan|cot|sec|csc|log|ln|exp|det|rank)\b", re.I)


FORI_RX = re.compile(r"\bfor\s+i\s*=\s*\d", re.I)
USING_RX = re.compile(r"^\s*using\b", re.I)

_HF_PIPE = None
_HF_NAME = None

CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]')

MATH_SHORT = {"sin","cos","tan","sec","csc","cot","pi","π","ln","log","rad","deg","e","ix","dx","dt"}
MATH_PHRASES = {
    "pythagorean","double-angle","half-angle","sum and difference","product-to-sum","sum-to-product",
    "angle addition","angle subtraction","unit circle","euler","complex exponential","radian","degree",
    "identity","identities","formula","formulas"
}


DOC_CODE_RX = re.compile(r"\b[A-Z]{2,}\d{2,}\b")

def _doc_name(c):
    return (c.get("doc_name") or "").strip()

def _dominant_doc(chs):
    counts = {}
    for c in chs:
        dn = _doc_name(c)
        if dn:
            counts[dn] = counts.get(dn, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]

def _target_doc_from_question(q, chs):
    q = (q or "").strip()
    m = DOC_CODE_RX.search(q)
    if m:
        code = m.group(0).lower()
        for c in chs:
            dn = _doc_name(c)
            if code in dn.lower():
                return dn
    toks = [t for t in re.findall(r"[A-Za-z0-9_.-]{4,}", q)]
    for c in chs:
        dn = _doc_name(c)
        for t in toks:
            if t.lower() in dn.lower():
                return dn
    return ""

def _scope_chunks(question, chs):
    if not chs:
        return chs
    strict = os.getenv("ASK_STRICT_DOC", "1").lower() in {"1","true","yes","on"}
    tgt = _target_doc_from_question(question, chs)
    if not tgt:
        tgt = _dominant_doc(chs)
    if not tgt:
        return chs
    if strict:
        return [c for c in chs if _doc_name(c) == tgt] or chs
    pri, sec = [], []
    for c in chs:
        (pri if _doc_name(c) == tgt else sec).append(c)
    return pri + sec

CHAPTER_RX = re.compile(r"\b(?:chapter|chap|ch)\s*([0-9]{1,3})\b", re.I)
SECTION_RX = re.compile(r"\b(?:section|sec|§)\s*([0-9]+(?:\.[0-9]+)*)\b", re.I)
HEADING_CH_RX = re.compile(r"\bchapter\s*([0-9]{1,3})\b", re.I)
HEADING_SEC_RX = re.compile(r"\bsection\s*([0-9]+(?:\.[0-9]+)*)\b", re.I)
LEAD_NUM_RX = re.compile(r"^\s*([0-9]{1,3})(?:\.[0-9]+)*\b")
CHAPTER_PAGE_WINDOW = int(os.getenv("ASK_CHAPTER_PAGE_WINDOW", "40"))

def _parse_chapter_query(q):
    q = (q or "").strip()
    ch = None
    sec = None
    m = CHAPTER_RX.search(q)
    if m:
        try:
            ch = int(m.group(1))
        except Exception:
            ch = None
    m2 = SECTION_RX.search(q)
    if m2:
        sec = m2.group(1)
    return ch, sec

def _chapter_from_heading(hp):
    if not hp:
        return None, None
    m = HEADING_CH_RX.search(hp)
    ch = None
    if m:
        try:
            ch = int(m.group(1))
        except Exception:
            ch = None
    m2 = HEADING_SEC_RX.search(hp)
    sec = m2.group(1) if m2 else None
    if ch is None:
        m3 = LEAD_NUM_RX.search(hp)
        if m3:
            try:
                ch = int(m3.group(1))
            except Exception:
                ch = None
    return ch, sec

def _filter_by_chapter(question, chs, soft=True):
    ch_req, sec_req = _parse_chapter_query(question)
    if not ch_req and not sec_req:
        return chs
    prim, soft_pool = [], []
    pages = []
    for c in chs:
        hp = (c.get("heading_path") or c.get("section_tag") or "").lower()
        ch, sec = _chapter_from_heading(hp)
        if ch_req and ch == ch_req:
            prim.append(c)
            p = _page_num(c)
            if p:
                pages.append(p)
        elif (not ch_req) and sec_req and sec and sec.startswith(sec_req):
            prim.append(c)
            p = _page_num(c)
            if p:
                pages.append(p)
        else:
            soft_pool.append(c)
    if prim:
        if pages:
            import numpy as _np
            mid = int(_np.median(pages))
            lo = max(1, mid - CHAPTER_PAGE_WINDOW)
            hi = mid + CHAPTER_PAGE_WINDOW
            near = [c for c in soft_pool if lo <= _page_num(c) <= hi]
            prim.extend(near)
        return prim
    if soft:
        toks = re.findall(r"[a-z0-9\\.]+", f"chapter {ch_req or ''} {sec_req or ''}")
        def _score(c):
            hp = (c.get("heading_path") or "").lower()
            s = 0
            for t in toks:
                if t and t in hp:
                    s += 1
            return -s
        return sorted(chs, key=_score)
    return chs

def _page_num(c):
    for k in ("page","page_num","page_index","page_start"):
        v = c.get(k)
        try:
            if v is not None:
                iv = int(v)
                if iv > 0:
                    return iv
        except Exception:
            pass
    return 0

def _openai_model():
    cfg = load_config()
    return cfg.get("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"

def _nfkc(s):
    try:
        return unicodedata.normalize("NFKC", s or "")
    except Exception:
        return s or ""

def is_formula_query(q):
    s = (q or "").lower()
    return any(w in s for w in FORMULA_Q_WORDS)

def _ensure_formula_labels(chunks, progress_cb=None):
    if not chunks:
        return chunks
    if not HAS_FCLS:
        return chunks
    try:
        info = getattr(formula_cls, "cls_info", lambda: {})() or {}
        if info.get("loaded"):
            tag = f"{info.get('base','')} + {info.get('adapter','')}"
        else:
            tag = "disabled"
    except Exception:
        tag = ""
    need = []
    for c in chunks:
        cm = c.get("formula_model") or ""
        if FORMULA_CLS_ALWAYS_REFRESH or ("is_formula" not in c) or (tag and cm != tag):
            need.append(c)
    if need:
        try:
            formula_cls.classify_chunks(need, progress_cb=progress_cb)
        except Exception:
            pass
    return chunks

def _split_lines_blocks(t):
    t = _nfkc(t)
    t = t.replace("\u00a0", " ")
    parts = []
    for ln in t.splitlines():
        x = ln.strip(" \t\r\n•-—–·")
        if x:
            parts.append(x)
    return parts

def _context_score(c):
    hp = (c.get("heading_path") or "").lower()
    s = 0
    for w in ("key formula","formulas","identities","identity","law","laws","rule","rules","theorem","definition","properties","summary"):
        if w in hp: s += 3
    for w in ("exercise","exercises","problems","examples","answers","solution","review"):
        if w in hp: s -= 4
    return s

def _extract_formula_spans(s):
    s = _nfkc(s or "")
    spans = []
    spans += EQ_SPAN_RX.findall(s)
    spans += FUNC_SPAN_RX.findall(s)
    spans += DERIV_SPAN_RX.findall(s)
    spans += INT_SPAN_RX.findall(s)
    spans += SUM_SPAN_RX.findall(s)
    spans += ABS_SPAN_RX.findall(s)
    spans += POW_SPAN_RX.findall(s)
    out = []
    for x in spans:
        t = x if isinstance(x, str) else " ".join([y for y in x if isinstance(y, str)])
        t = re.sub(r"\s+", " ", t).strip(" ,;.")
        if t:
            out.append(t[:200])
    return _dedup_ordered(out)

def _digit_ratio(s):
    s2 = re.sub(r"\s+", "", s or "")
    d = sum(ch.isdigit() for ch in s2)
    n = len(s2) or 1
    return d / n


def _numiness(span):
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", span or "")
    return len(set(nums))

def _looks_binary(s):
    return bool(re.fullmatch(r"[01\s]{24,}", s or ""))

def _is_good_span(span, ctx_score):
    words = len(re.findall(r"[A-Za-z]{3,}", span))
    ops = len(re.findall(r"[=<>±∑∏∫√∞∇∂\^*/+\-|]", span))
    dens = ops / max(1, ops + words)
    if FORI_RX.search(span):
        return False
    if _numiness(span) >= 3 and not re.search(r"[A-Za-zα-ωΑ-Ω]{2,}", span):
        return False
    if _looks_binary(span) or _digit_ratio(span) > 0.7:
        return False
    if ctx_score <= -1 and dens < 0.45:
        return False
    if NONFORM_RX.search(span) and dens < 0.35:
        return False
    if len(span) > 160 and dens < 0.30:
        return False
    has_sym = bool(re.search(r"[=≤≥≠∑∏∫√∞∇∂^]|[∀∃¬∧∨→⇒⇔↔]", span))
    has_var = bool(re.search(r"[A-Za-zα-ωΑ-Ω]", span))
    if not (has_sym and has_var):
        return False
    if HAS_FCLS and _FORMULA_SCORE_FN is not None:
        try:
            res = _FORMULA_SCORE_FN([span])
            p = float(res[0]) if isinstance(res, (list, tuple)) else float(res)
            if p < FORMULA_MIN_PROB:
                return False
        except Exception:
            p = None
    if ctx_score <= -1 and HAS_FCLS and _FORMULA_SCORE_FN is not None:
        try:
            res2 = _FORMULA_SCORE_FN([span])
            p2 = float(res2[0]) if isinstance(res2, (list, tuple)) else float(res2)
            if p2 < (FORMULA_MIN_PROB + 0.08):
                return False
        except Exception:
            pass
    return True

def _dedup_ordered(items):
    out = []
    seen = set()
    for x in items:
        k = re.sub(r"\s+", " ", x).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _rows_to_context(rows, char_budget=14000):
    parts = []
    used = 0
    rows2 = sorted(rows, key=lambda r: (r.get("source",""), r.get("page",0)))
    for r in rows2:
        line = f"{r.get('source','[?]')} {r.get('formula','')}"
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if used + len(line) + 1 > char_budget:
            break
        parts.append(line)
        used += len(line) + 1
    return "\n".join(parts)

def _canonicalize_formulas(question, rows):
    cfg = load_config()
    ctx = _rows_to_context(rows)
    prompt = CANONICAL_FORMULA_TEMPLATE.format(q=question, ctx=ctx)
    if cfg.get("OPENAI_API_KEY"):
        sys = "Return only canonical formulas with labels and page citations; no examples."
        text, _ = _openai_chat([{"role":"system","content":sys},{"role":"user","content":prompt}], max_new_tokens=700)
    else:
        full = "System: Return only canonical formulas with labels and page citations; no examples.\n" + prompt + "\nAssistant:"
        text = _normalize_md(_hf_generate(full, max_new_tokens=700))
    return text

def extract_formulas_from_chunks(chs, max_per_page=FORMULA_MAX_PER_PAGE, progress_cb=None, time_budget_sec=None, wants_explain=False, exhaustive_scan=False):
    if HAS_FCLS:
        _maybe_init_sft(progress_cb)
        if _FORMULA_SCORE_FN is not None:
            try:
                _FORMULA_SCORE_FN(["x = x"])
            except Exception:
                pass
    rows = []
    diag = {"pages": 0, "spans": 0, "scored": 0, "kept": 0, "p_sum": 0.0, "skipped_no_hint": 0, "scanned": 0}
    total = len(chs)
    found = 0
    if exhaustive_scan:
        time_budget_sec = None
        max_per_page = max_per_page if max_per_page and max_per_page > 0 else 1000

    t_end = time.time() + time_budget_sec if time_budget_sec and time_budget_sec > 0 else None
    p_cache = {}
    for i, c in enumerate(chs):
        diag["pages"] += 1
        if t_end and time.time() >= t_end:
            break
        t = c.get("text","") or ""
        if not t:
            if progress_cb and (i % 50 == 0 or i == total - 1):
                progress_cb(f"Scanning pages… {i+1}/{total} • formulas: {found}")
            continue
        hint_ok = bool(FORMULA_HINT_RX.search(t))
        soft_page = False
        if not hint_ok and not exhaustive_scan:
            soft_page = True
        ctxs = _context_score(c)
        hp = (c.get("heading_path") or "")
        st = (c.get("section_tag") or "")
        if st == "exercises":
            ctxs -= 2
        if not wants_explain and (HEAD_VETO_RX.search(hp) or HEAD_VETO_RX.search(st)):
            if progress_cb and (i % 20 == 0 or i == total - 1):
                progress_cb(f"Scanning pages… {i+1}/{total} • formulas: {found}")
            continue
        lines = _split_lines_blocks(t)
        diag["scanned"] += 1
        kept = []
        for ln in lines:
            if not wants_explain and (BIB_HINT_RX.search(ln) or YEAR_RX.search(ln)):
                continue
            if LINE_VETO_RX.search(ln):
                continue
            if USING_RX.search(ln):
                continue
            if QUESTION_RX.search(ln):
                continue
            if ENUM_RX.search(ln) and not re.search(r"[=≤≥≠≈∑∏∫√∞∇∂\^→↔⇔]", ln):
                continue
            if PROMPT_PREFIX_RX.search(ln) and not wants_explain and not re.search(r"[=≤≥≠≈∑∏∫√∞∇∂\^→↔⇔]", ln):
                continue
            if EXERCISE_ANY_RX.search(ln) and not wants_explain and not re.search(r"[=≤≥≠≈∑∏∫√∞∇∂\^→↔⇔]", ln):
                continue
            spans = _extract_formula_spans(ln)
            diag["spans"] += len(spans)
            if not spans:
                continue
            for s in spans:
                base_ok = _is_good_span(s, ctxs)
                words = len(re.findall(r"[A-Za-z]{3,}", s))
                ops = len(re.findall(r"[=<>±∑∏∫√∞∇∂\^*/+\-|]", s))
                dens = ops / max(1, ops + words)
                keep_by_density = dens >= FORMULA_STRONG_DENS
                p = None
                if HAS_FCLS and _FORMULA_SCORE_FN is not None:
                    try:
                        if s in p_cache:
                            p = p_cache[s]
                        else:
                            res = _FORMULA_SCORE_FN([s])
                            p = float(res[0]) if isinstance(res, (list, tuple)) else float(res)
                            p_cache[s] = p
                        diag["scored"] += 1
                        diag["p_sum"] += float(p)
                    except Exception:
                        p = None
                page_prob = FORMULA_MIN_PROB + (0.08 if soft_page else 0.0)
                keep_by_prob = (p is not None and p >= page_prob)
                if base_ok or keep_by_prob or keep_by_density:
                    sym = s.count("=")*0.35 + s.count("∑")*0.45 + s.count("∫")*0.45 + s.count("→")*0.25 + s.count("↔")*0.25
                    sym = min(1.4, sym)
                    lp = -0.15 if len(s) > 140 else 0.0
                    base_ctx = 0.8 * ctxs
                    if p is None:
                        sc = 1.5 + base_ctx + sym + lp + (0.2 if keep_by_density else 0.0)
                    else:
                        sc = 3.0 * p + base_ctx + sym + lp
                    kept.append((sc, s))
                    diag["kept"] += 1
                    if len(kept) >= max_per_page:
                        break
            if len(kept) >= max_per_page:
                break
        if kept:
            kept.sort(key=lambda z: -z[0])
            scored = kept
            min_keep = max(FORMULA_MIN_KEEP_PER_PAGE, int(len(scored) * FORMULA_KEEP_FRAC))
            filter_by_prob = []
            page_prob = FORMULA_MIN_PROB + (0.08 if 'soft_page' in locals() and soft_page else 0.0)
            for sc, s in scored:
                p = None
                if HAS_FCLS and _FORMULA_SCORE_FN is not None:
                    try:
                        if s in p_cache:
                            p = p_cache[s]
                        else:
                            res = _FORMULA_SCORE_FN([s])
                            p = float(res[0]) if isinstance(res, (list, tuple)) else float(res)
                            p_cache[s] = p
                    except Exception:
                        p = None
                if p is not None:
                    if p >= page_prob:
                        filter_by_prob.append((sc, s, p))
                else:
                    filter_by_prob.append((sc, s, None))
            if len(filter_by_prob) < min_keep:
                top_min = []
                for i, (sc, s) in enumerate(scored):
                    p = None
                    if HAS_FCLS and _FORMULA_SCORE_FN is not None:
                        try:
                            if s in p_cache:
                                p = p_cache[s]
                            else:
                                res = _FORMULA_SCORE_FN([s])
                                p = float(res[0]) if isinstance(res, (list, tuple)) else float(res)
                                p_cache[s] = p
                        except Exception:
                            p = None
                    top_min.append((sc, s, p))
                    if len(top_min) >= min_keep:
                        break
                final_kept = top_min
            else:
                final_kept = filter_by_prob
            src = _fmt_source(c)
            pg = _page_num(c)
            for _, g, _ in final_kept:
                rows.append({"formula": g, "source": src, "page": pg})
            found += len(final_kept)
        if progress_cb and (i % 20 == 0 or i == total - 1):
            progress_cb(f"Scanning pages… {i+1}/{total} • formulas: {found}")
    if progress_cb:
        try:
            avg_p = (diag["p_sum"] / diag["scored"]) if diag["scored"] > 0 else 0.0
            min_p = (min(p_cache.values()) if p_cache else 0.0)
            label = "SFT" if (_FORMULA_SCORE_FN is not None) else "heuristics"
            cov = f"coverage {diag['scanned']}/{total} pages (skipped_no_hint {diag['skipped_no_hint']})"
            progress_cb(f"{label}: pages {diag['pages']} • spans {diag['spans']} • scored {diag['scored']} • kept {diag['kept']} • avg_p {avg_p:.3f} (min {min_p:.2f}, thresh {FORMULA_MIN_PROB:.2f}) • {cov}")
        except Exception:
            pass
    return rows, diag

def summarize_all_formulas(question, chunks, progress_cb=None, exhaustive=None, time_budget_sec=None):
    if exhaustive is None:
        exhaustive = (os.environ.get("ASK_EXHAUSTIVE","false").lower() in {"1","true","yes","on"}) or FORMULA_EXHAUSTIVE

    if exhaustive:
        tb = None
    else:
        tb = time_budget_sec if time_budget_sec else int(os.environ.get("ASK_TIME_BUDGET_SEC","120"))

    base = list(chunks)
    if exhaustive:
        try:
            from api.services.index import list_chunks
            allc = list_chunks()
            if allc:
                base = allc
                if progress_cb:
                    progress_cb(f"Formula mode: exhaustive scan of {len(base)} chunks")
        except Exception:
            pass

    base = _scope_chunks(question, base)
    base = _filter_by_chapter(question, base)
    base = _ensure_formula_labels(base, progress_cb)
    total_scoped = len(base)
    labeled = [c for c in base if c.get("is_formula")]
    if total_scoped > 0 and len(labeled) / total_scoped < FORMULA_MIN_FORMULA_RATIO:
        extras = []
        for c in base:
            t = c.get("text", "") or ""
            hp = (c.get("heading_path") or "").lower()
            if c in labeled:
                continue
            if c.get("has_math"):
                extras.append(c); continue
            if FORMULA_HINT_RX.search(t):
                extras.append(c); continue
            if any(w in hp for w in ("formula", "formulas", "identity", "identities", "theorem", "rule", "laws")):
                extras.append(c); continue
        base = labeled + [x for x in extras if x not in labeled]
        if progress_cb:
            try:
                progress_cb(f"Formula fallback engaged: {len(labeled)}/{total_scoped} labeled; added {len(base)-len(labeled)} extras")
            except Exception:
                pass
    else:
        base = labeled
    if not base:
        return {"answer":"Not found in sources.", "citations":[], "quotes":[]}

    eq_non = [c for c in base if c.get("is_equation") and (c.get("section_tag") or "") != "exercises"]
    eq_ex  = [c for c in base if c.get("is_equation") and (c.get("section_tag") or "") == "exercises"]
    ma_non = [c for c in base if not c.get("is_equation") and c.get("has_math") and (c.get("section_tag") or "") != "exercises"]
    ma_ex  = [c for c in base if not c.get("is_equation") and c.get("has_math") and (c.get("section_tag") or "") == "exercises"]

    ordered = eq_non + ma_non + eq_ex + ma_ex
    try:
        use_name = RERANKER_NAME_DEFAULT if RERANKER_NAME_DEFAULT not in {"", "off", "none"} else "minilm"
        rer_idx = _rerank(question, ordered, name=use_name, top_n=RERANK_TOP_N)
        ordered = [ordered[i] for i in rer_idx]
    except Exception:
        pass
    ask = (question or "").lower()
    wants_explain = any(k in ask for k in ("explain","why","derive","derivation","proof","prove","show"))
    rows, diag = extract_formulas_from_chunks(ordered, progress_cb=progress_cb, time_budget_sec=tb, wants_explain=wants_explain, exhaustive_scan=exhaustive)
    if not rows:
        return {"answer":"Not found in sources.", "citations":[], "quotes":[]}
    rows.sort(key=lambda r: (r.get("source",""), r.get("page",0)))
    answer = _canonicalize_formulas(question, rows)
    raw_cites = _dedup_ordered([r.get("source","") for r in rows])
    cites = _compress_citations(raw_cites)[:10]
    if ASK_FORMULA_DEBUG:
        cov = f"Coverage {diag.get('scanned',0)}/{len(ordered)} pages; skipped_no_hint {diag.get('skipped_no_hint',0)}"
        answer += "\n\n_Diagnostics: SFT filtering active; {}. Min prob = {:.2f}_".format(cov, FORMULA_P_MIN)
    if ASK_CITATION_NOTE:
        answer += "\n\n*Grounded in the cited pages below.*"
    return {"answer": answer, "citations": cites, "quotes": []}
    return {"answer": answer, "citations": cites, "quotes": []}

def _normalize_md(s):
    s = s.strip().replace("\u00a0", " ")
    m = re.match(r"^\s*```[\w\-]*\s*\n?(.*?)\n?\s*```\s*$", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"(?m)^\s*[–—•]\s+", "- ", s)
    lines = [ln.rstrip() for ln in s.splitlines()]
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("- "):
        lines[i] = lines[i].lstrip()[2:].strip()
    return "\n".join(lines).strip()

def _ascii_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if ord(ch) < 128)
    return n / len(s)

def _filter_chunks(chs):
    out = []
    for c in chs:
        t = _nfkc(c.get("text") or "").strip()
        if not t:
            continue
        if CJK_RE.search(t):
            if _ascii_ratio(t) < 0.20:
                continue
        out.append(c)
    return out

def _ascii_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if ord(ch) < 128)
    return n / len(s)

def _filter_chunks(chs):
    out = []
    for c in chs:
        t = _nfkc(c.get("text") or "").strip()
        if not t:
            continue
        if CJK_RE.search(t):
            if _ascii_ratio(t) < 0.20:
                continue
        out.append(c)
    return out


_RERANKER = {"name": None, "ce": None}

def _load_cross_encoder(tag):
    try:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        try:
            torch.set_num_threads(int(os.getenv("RERANK_THREADS", "1")))
        except Exception:
            pass
        from sentence_transformers import CrossEncoder
        model_map = {
            "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "bge-m3": "BAAI/bge-reranker-v2-m3",
            "bge-base": "BAAI/bge-reranker-base",
            "bge-large": "BAAI/bge-reranker-large",
        }
        mname = model_map.get(tag, tag)
        ce = CrossEncoder(mname, device="cpu")
        return ce, mname
    except Exception:
        return None, None


def _rerank(question, chs, name=RERANKER_NAME_DEFAULT, top_n=RERANK_TOP_N):
    tag = (name or "off").lower().strip()
    if tag in {"off", "none", ""}:
        return list(range(len(chs)))
    try:
        if _RERANKER["ce"] is None or _RERANKER["name"] != tag:
            ce, mname = _load_cross_encoder(tag)
            if ce is None:
                return list(range(len(chs)))
            _RERANKER["ce"] = ce
            _RERANKER["name"] = tag
        ce = _RERANKER["ce"]
        cut = min(len(chs), max(1, int(top_n)))
        pairs = []
        for i in range(cut):
            t = chs[i].get("text", "")
            t = re.sub(r"\s+", " ", t).strip()[:800]
            if not t:
                t = "(empty)"
            pairs.append([question, t])
        if not pairs:
            return list(range(len(chs)))
        scores = ce.predict(pairs)
        try:
            import numpy as _np
            order = list(_np.argsort(-_np.array(scores)))
        except Exception:
            order = list(range(len(pairs)))
            order.sort(key=lambda i: float(scores[i]) if i < len(scores) else 0.0, reverse=True)
        rest = list(range(cut, len(chs)))
        return order + rest
    except Exception:
        return list(range(len(chs)))

def _fmt_source(c):
    doc = c.get("doc_name") or "Unknown.pdf"
    p = _page_num(c)
    if p:
        return f"[{doc} p.{p}]"
    ps = c.get("page_start")
    pe = c.get("page_end")
    if ps and pe and ps != pe:
        return f"[{doc} p.{ps}-{pe}]"
    if ps:
        return f"[{doc} p.{ps}]"
    return f"[{doc} p.?]"

def _compress_citations(cites):
    by_doc = {}
    rx = re.compile(r"^\[([^]]+?)\s+p\.(\d+)(?:-(\d+))?\]$")
    for tag in cites:
        m = rx.match(tag or "")
        if not m:
            by_doc.setdefault(tag or "", set())
            continue
        doc = m.group(1)
        a = int(m.group(2))
        b = int(m.group(3) or m.group(2))
        pages = by_doc.setdefault(doc, set())
        for p in range(min(a, b), max(a, b) + 1):
            pages.add(p)
    out = []
    for doc, pages in by_doc.items():
        if not pages:
            out.append(f"[{doc}]")
            continue
        ps = sorted(pages)
        ranges = []
        s = e = ps[0]
        for p in ps[1:]:
            if p == e + 1:
                e = p
            else:
                ranges.append((s, e))
                s = e = p
        ranges.append((s, e))
        parts = [(str(s) if s == e else f"{s}–{e}") for s, e in ranges]
        out.append(f"[{doc} p.{', '.join(parts)}]")
    out.sort()
    return out

def _format_ctx(chs):
    parts = []
    for c in chs:
        parts.append(f"{_fmt_source(c)} {c['text'][:1200]}")
    return "\n---\n".join(parts)

def _clip_context(chunks, budget=CTX_CHAR_BUDGET):
    out, used = [], 0
    for c in chunks:
        s = re.sub(r"\s+", " ", c["text"]).strip()
        if not s:
            continue
        block = f"{_fmt_source(c)} {s}"
        if used + len(block) > budget:
            remain = max(0, budget - used)
            out.append(block[:remain])
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)

def summarizer_info():
    return {"backend": "openai" if load_config().get("OPENAI_API_KEY") else "hf"}

def _hf_pipe():
    global _HF_PIPE, _HF_NAME
    if _HF_PIPE is not None:
        return _HF_PIPE, _HF_NAME
    from transformers import pipeline
    import torch
    name = os.getenv("HF_LLM_NAME") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = 0 if torch.cuda.is_available() else -1
    _HF_PIPE = pipeline("text-generation", model=name, tokenizer=name, device=device)
    _HF_NAME = name
    return _HF_PIPE, _HF_NAME

def _hf_generate(prompt, max_new_tokens=800):
    pipe, _ = _hf_pipe()
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2, return_full_text=False, pad_token_id=pipe.tokenizer.eos_token_id or 0)
    return out[0]["generated_text"].strip()

def _split_sentences(t):
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[\.\!\?;:])\s+(?=[A-Za-z0-9\\(])|\n+", t)
    return [p.strip() for p in parts if p.strip()]

def _extract_quotes(question, chunks, max_quotes=5, window=1):
    kws = set(_keywords(question))
    sents = []
    meta = []
    for c in chunks:
        ss = _split_sentences(c["text"])
        for i, s in enumerate(ss):
            sents.append(s[:400])
            meta.append((c, i, ss))
    if not sents:
        return []
    qv = embed_texts([question])
    sv = embed_texts(sents)
    sims = (qv @ sv.T).flatten()
    lex = []
    for s in sents:
        s_low = s.lower()
        score = 0
        for k in kws:
            if k in s_low:
                score += 1
        if "\\sin" in s or "\\cos" in s or "\\tan" in s:
            score += 1
        if "π" in s or "pi" in s_low or "e^i" in s_low:
            score += 1
        lex.append(score)
    lex = np.array(lex, dtype=float)
    if lex.max() > 0:
        lex = lex / lex.max()
    comb = 0.8 * sims + 0.2 * lex
    order = np.argsort(-comb)
    seen = set()
    quotes = []
    for idx in order:
        c, i, ss = meta[idx]
        key = (c.get("doc_name") or "Unknown.pdf", c.get("page") or c.get("page_start"))
        if key in seen:
            continue
        start = max(0, i - window)
        end = min(len(ss), i + window + 1)
        snippet = " ".join(ss[start:end])[:500]
        quotes.append({"source": _fmt_source(c), "quote": snippet})
        seen.add(key)
        if len(quotes) >= max_quotes:
            break
    return quotes

def _filter_chunks(chs):
    out = []
    for c in chs:
        t = c.get("text") or ""
        t = _nfkc(t).strip()
        if not t:
            continue
        if CJK_RE.search(t):
            continue
        out.append(c)
    return out

def _keywords(q):
    q = q.lower()
    q = re.sub(r"[^a-z0-9\sπ]", " ", q)
    toks = [w for w in q.split() if w and w not in {"what","which","this","that","about","were","does","with","from","your","their","into","there","when","where","many","show","give","tell","list"}]
    out = set()
    for w in toks:
        if len(w) >= 4:
            out.add(w)
        elif w in MATH_SHORT:
            out.add(w)
    for p in MATH_PHRASES:
        if p.replace("-", " ") in q or p in q:
            out.add(p)
    if "trig" in q or "trigonometry" in q:
        out |= {"sin","cos","tan","identity","identities","pythagorean","double-angle","half-angle"}
    return list(out)

def _pref_string():
    a = load_aliases()
    if not a:
        return ""
    pairs = [f"{k} -> {v}" for k,v in a.items()]
    return "Use user-preferred terminology: " + "; ".join(pairs) + "."

def _estimate_tokens(s):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)

_rate_state = {"t_start":0.0,"tok":0,"r_start":0.0,"req":0}

def _rate_limit(tokens):
    cfg = load_config()
    tpm = int(cfg.get("OPENAI_TPM") or os.getenv("OPENAI_TPM") or "0")
    rpm = int(cfg.get("OPENAI_RPM") or os.getenv("OPENAI_RPM") or "0")
    now = time.time()
    if _rate_state["t_start"] == 0.0:
        _rate_state["t_start"] = now
        _rate_state["r_start"] = now
    if tpm > 0:
        if now - _rate_state["t_start"] >= 60:
            _rate_state["t_start"] = now
            _rate_state["tok"] = 0
        if _rate_state["tok"] + tokens > tpm:
            sleep = 60 - (now - _rate_state["t_start"])
            if sleep > 0:
                time.sleep(sleep)
                _rate_state["t_start"] = time.time()
                _rate_state["tok"] = 0
        _rate_state["tok"] += tokens
    if rpm > 0:
        if now - _rate_state["r_start"] >= 60:
            _rate_state["r_start"] = now
            _rate_state["req"] = 0
        if _rate_state["req"] + 1 > rpm:
            sleep = 60 - (now - _rate_state["r_start"])
            if sleep > 0:
                time.sleep(sleep)
                _rate_state["r_start"] = time.time()
                _rate_state["req"] = 0
        _rate_state["req"] += 1

def _openai_chat(messages, max_new_tokens=800):
    from openai import OpenAI
    client = OpenAI(api_key=(load_config().get("OPENAI_API_KEY")))
    s = "\n".join(m.get("content","") for m in messages)
    need = _estimate_tokens(s) + max_new_tokens
    _rate_limit(need)
    msg = client.chat.completions.create(model=_openai_model(), messages=messages)
    return _normalize_md(msg.choices[0].message.content), need


def _score_chunk_for_order(c):
    s = 0.0
    try:
        s = float(c.get("_score", 0.0))
    except Exception:
        s = 0.0
    if c.get("has_math"):
        s += 0.1
    if c.get("has_table"):
        s -= 0.05
    return -s

def _mmr_order(question, chs, top_n=MMR_TOP_N, alpha=MMR_ALPHA):
    n = min(len(chs), top_n)
    if n <= 2:
        return list(range(len(chs)))
    texts = [re.sub(r"\s+", " ", c["text"]).strip()[:800] for c in chs[:n]]
    qv = embed_texts([question])[0]
    dv = embed_texts(texts)
    qv = qv / (np.linalg.norm(qv) + 1e-8)
    dv = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8)
    sims = dv @ qv
    selected = []
    candidates = list(range(n))
    selected.append(int(np.argmax(sims)))
    candidates.remove(selected[0])
    while candidates:
        mmr_scores = []
        for idx in candidates:
            rel = sims[idx]
            if selected:
                div = max(float(dv[idx] @ dv[j]) for j in selected)
            else:
                div = 0.0
            mmr = alpha * rel - (1 - alpha) * div
            mmr_scores.append((mmr, idx))
        mmr_scores.sort(reverse=True)
        pick = mmr_scores[0][1]
        selected.append(pick)
        candidates.remove(pick)
    rest = list(range(n, len(chs)))
    return selected + rest

def _batch_chunks(chs, budget_chars):
    out = []
    cur = []
    used = 0
    for c in chs:
        t = re.sub(r"\s+", " ", c["text"]).strip()
        if not t:
            continue
        block = f"{_fmt_source(c)} {t}"
        if used + len(block) > budget_chars and cur:
            out.append(cur)
            cur = [c]
            used = len(block)
        else:
            cur.append(c)
            used += len(block)
        if used >= budget_chars * 1.2 and cur:
            out.append(cur)
            cur = []
            used = 0
    if cur:
        out.append(cur)
    return out

def summarize(question, top_chunks, history=None):
    cfg = load_config()
    scoped = _scope_chunks(question, top_chunks)
    scoped = _filter_by_chapter(question, scoped)
    filtered = _filter_chunks(scoped)
    used = filtered if filtered else list(scoped)
    if not used:
        return {"answer": "Not found in sources.", "citations": [], "quotes": []}
    terms = ", ".join(_keywords(question))
    ctx = _format_ctx(used)
    prompt = TEMPLATE.format(q=question, ctx=ctx, terms=terms)
    if cfg.get("OPENAI_API_KEY"):
        prefs = _pref_string()
        sysmsg = "You extract key ideas and cite pages. Return Markdown." + (" " + prefs if prefs else "")
        messages = [{"role": "system", "content": sysmsg}]
        if history:
            for turn in history[-8:]:
                if turn.get("q"):
                    messages.append({"role": "user", "content": turn["q"]})
                if turn.get("a"):
                    messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": prompt})
        text, _ = _openai_chat(messages, max_new_tokens=700)
    else:
        hist = ""
        if history:
            for turn in history[-8:]:
                if turn.get("q"):
                    hist += f"\nUser: {turn['q']}\n"
                if turn.get("a"):
                    hist += f"Assistant: {turn['a']}\n"
        full = "System: You extract key ideas from the provided context only. Return Markdown with citations like [FileName.pdf p.12]." + hist + "\n" + prompt + "\nAssistant:"
        text = _normalize_md(_hf_generate(full, max_new_tokens=700))
    cites = []
    for c in used:
        cites.append(_fmt_source(c))
    dedup = []
    seen = set()
    for s in cites:
        if s not in seen:
            dedup.append(s); seen.add(s)
    cites = _compress_citations(dedup)[:15]
    if ASK_CITATION_NOTE:
        text += "\n\n*Grounded in the cited pages below.*"
    quotes = _extract_quotes(question, used, max_quotes=5, window=1)
    return {"answer": text, "citations": cites, "quotes": quotes}

def summarize_batched(question, chunks, history=None, progress_cb=None, max_batches=None, time_budget_sec=None, exhaustive=None):
    cfg = load_config()
    scoped = _scope_chunks(question, chunks)
    scoped = _filter_by_chapter(question, scoped)
    filtered = _filter_chunks(scoped)
    base = filtered if filtered else list(scoped)
    if not base:
        return {"answer": "Not found in sources.", "citations": [], "quotes": []}
    if max_batches is None:
        max_batches = MAX_BATCHES_DEFAULT
    if time_budget_sec is None:
        time_budget_sec = TIME_BUDGET_SEC_DEFAULT
    if exhaustive is None:
        exhaustive = ASK_EXHAUSTIVE_DEFAULT
    sweep = bool(exhaustive)
    if not sweep:
        max_batches = 1
    prim = sorted(base, key=_score_chunk_for_order)
    try:
        mmr_idx = _mmr_order(question, prim)
        ordered = [prim[i] for i in mmr_idx]
    except Exception:
        ordered = prim
    try:
        rer_idx = _rerank(question, ordered, name=RERANKER_NAME_DEFAULT, top_n=RERANK_TOP_N)
        ordered = [ordered[i] for i in rer_idx]
    except Exception:
        pass
    batches = _batch_chunks(ordered, BATCH_CHAR_BUDGET)
    if cfg.get("OPENAI_API_KEY"):
        prefs = _pref_string()
        sysmsg = "You extract key ideas and cite pages. Return Markdown." + (" " + prefs if prefs else "")
        hist_msgs = []
        if history:
            for turn in history[-4:]:
                if turn.get("q"):
                    hist_msgs.append({"role": "user", "content": turn["q"]})
                if turn.get("a"):
                    hist_msgs.append({"role": "assistant", "content": turn["a"]})
        tpm = int(cfg.get("OPENAI_TPM") or os.getenv("OPENAI_TPM") or "0")
        rpm = int(cfg.get("OPENAI_RPM") or os.getenv("OPENAI_RPM") or "0")
        tok_cap = math.inf if tpm <= 0 else int(tpm * (time_budget_sec / 60.0))
        req_cap = math.inf if rpm <= 0 else int(rpm * (time_budget_sec / 60.0))
        used_tok = 0
        used_req = 0
        parts = []
        seen_sources = []
        total = min(len(batches), max_batches)
        if not sweep:
            total = min(total, 1)
        t_end = time.time() + time_budget_sec if sweep else None
        plateau = 0
        for i, batch in enumerate(batches[:total], 1):
            if t_end and time.time() >= t_end:
                break
            ctx = _format_ctx(batch)
            prompt = TEMPLATE.format(q=question, ctx=ctx, terms=", ".join(_keywords(question)))
            messages = [{"role": "system", "content": sysmsg}] + hist_msgs + [{"role": "user", "content": prompt}]
            pre_tokens = _estimate_tokens("\n".join(m["content"] for m in messages)) + 600
            if used_tok + pre_tokens > tok_cap or used_req + 1 > req_cap:
                break
            tok_label = ("∞" if tok_cap == math.inf else str(tok_cap))
            if progress_cb and sweep:
                try:
                    left = int(max(0, (t_end - time.time()) if t_end else 0))
                    progress_cb(f"Batch {i}/{total} • {used_tok}/{tok_label} tok • {left}s left")
                except Exception:
                    pass
            ans, spent = _openai_chat(messages, max_new_tokens=600)
            used_tok += spent
            used_req += 1
            parts.append(ans)
            prev = set(seen_sources)
            for c in batch:
                seen_sources.append(_fmt_source(c))
            delta = len(set(seen_sources) - prev)
            if not sweep:
                break
            elif i >= 2 and delta == 0:
                break
        if not parts:
            return summarize(question, ordered[:max(1, min(6, len(ordered)))], history=history)
        fuse_ctx = "\n\n---\n\n".join(parts)[:40000]
        fuse_prompt = "You are consolidating multiple partial answers derived strictly from course readings. Merge them into a single, non-redundant answer. Keep formulas, be precise, and keep citations from the partials in place.\n\nQuestion:\n" + question + "\n\nPartials:\n" + fuse_ctx
        final, _ = _openai_chat([{"role": "system", "content": "Return a single clean Markdown answer with citations kept as-is."}, {"role": "user", "content": fuse_prompt}], max_new_tokens=800)
    else:
        parts = []
        seen_sources = []
        total = min(len(batches), max_batches)
        if not sweep:
            total = min(total, 1)
        t_end = time.time() + time_budget_sec if sweep else None
        for i, batch in enumerate(batches[:total], 1):
            if t_end and time.time() >= t_end:
                break
            if progress_cb and sweep:
                try:
                    left = int(max(0, (t_end - time.time()) if t_end else 0))
                    progress_cb(f"Batch {i}/{total} • {left}s left")
                except Exception:
                    pass
            ctx = _format_ctx(batch)
            prompt = TEMPLATE.format(q=question, ctx=ctx, terms=", ".join(_keywords(question)))
            hist = ""
            if history:
                for turn in history[-4:]:
                    if turn.get("q"):
                        hist += f"\nUser: {turn['q']}\n"
                    if turn.get("a"):
                        hist += f"Assistant: {turn['a']}\n"
            full = "System: You extract key ideas from the provided context only. Return Markdown with citations like [FileName.pdf p.12]." + hist + "\n" + prompt + "\nAssistant:"
            ans = _normalize_md(_hf_generate(full, max_new_tokens=600))
            parts.append(ans)
            prev = set(seen_sources)
            for c in batch:
                seen_sources.append(_fmt_source(c))
            delta = len(set(seen_sources) - prev)
            if not sweep:
                break
            elif i >= 2 and delta == 0:
                break
        if not parts:
            return summarize(question, ordered[:max(1, min(6, len(ordered)))], history=history)
        fuse_ctx = "\n\n---\n\n".join(parts)[:40000]
        fuse_prompt = "You are consolidating multiple partial answers derived strictly from course readings. Merge them into a single, non-redundant answer. Keep formulas, be precise, and keep citations from the partials in place.\n\nQuestion:\n" + question + "\n\nPartials:\n" + fuse_ctx
        final = _normalize_md(_hf_generate("System: Consolidate the partials into one answer, preserving citations.\n" + fuse_prompt + "\nAssistant:", max_new_tokens=800))
    cites = []
    seen = set()
    for s in seen_sources:
        if s not in seen:
            cites.append(s); seen.add(s)
        if len(cites) >= 15:
            break
    quotes = _extract_quotes(question, ordered[:min(len(ordered), 200)], max_quotes=5, window=1)
    cites = _compress_citations(cites)[:15]
    if ASK_CITATION_NOTE:
        final += "\n\n*Grounded in the cited pages below.*"
    quotes = _extract_quotes(question, ordered[:min(len(ordered), 200)], max_quotes=5, window=1)
    return {"answer": final, "citations": cites, "quotes": quotes}

def summarize_document(chunks):
    cfg = load_config()
    filtered = _filter_chunks(chunks)
    if not filtered:
        summary = "**No clean extractable text found in the provided pages.**"
        SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARIES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": int(time.time()), "keywords": [], "formulas": [], "summary": summary}, ensure_ascii=False) + "\n")
        return {"summary": summary, "keywords": [], "formulas": []}
    ctx = _clip_context(filtered)
    if cfg.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
        prompt = DOC_TEMPLATE.format(ctx=ctx)
        msg = client.chat.completions.create(
            model=_openai_model(),
            messages=[
                {"role": "system", "content": "Return structured, factual notes with page citations in Markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        summary = _normalize_md(msg.choices[0].message.content)
    else:
        full = "System: Produce a comprehensive study summary in Markdown with bullets and citations like [FileName.pdf p.12].\n" + DOC_TEMPLATE.format(ctx=ctx) + "\nAssistant:"
        summary = _normalize_md(_hf_generate(full, max_new_tokens=1200))
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARIES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "keywords": [], "formulas": [], "summary": summary[:40000]}, ensure_ascii=False) + "\n")
    return {"summary": summary, "keywords": [], "formulas": []}
