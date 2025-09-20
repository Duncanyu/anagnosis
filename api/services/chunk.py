import re, math
try:
    import tiktoken
    _tok = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tok = None

CHAR_GAP_COL = 80
TOK_TARGET_DEFAULT = 750
TOK_OVERLAP_DEFAULT = 90

BULLET_RX = re.compile(r"^\s*([\-–—•·▪‣]\s+|\d+\.\s+|\(\d+\)\s+|\([a-zA-Z]\)\s+)")
HEADING_RX = re.compile(r"^([A-Z][A-Z0-9 \-:&]{3,}|(\d+(\.\d+){0,3})\s+[A-Z].+)$")
MATH_RX = re.compile(r"[=±×÷≤≥∑∏∫√∞≈≠⊂⊆⊃⊇∈∉∧∨⇒⇔αβγδΔεζηθικλμνξπρστυφχψωΩ]")
TABLE_HINT_RX = re.compile(r"\b(Table|tabular|columns?|rows?)\b", re.I)
MATH_FONT_RX = re.compile(r"(cmr|cmmi|cmsy|cmex|stix|cambria|symbol|math)", re.I)
EXERCISE_RX = re.compile(r"\b(exercises?|problems?|answers?|solution|review)\b", re.I)

def _norm(s):
    s = s.replace("\u00a0", " ").replace("\u00ad", "")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()

def _sentences(t):
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?;:])\s+(?=[A-Z0-9(])", t)
    return [p.strip() for p in parts if p.strip()]

def _tokens_len(s):
    if _tok:
        try:
            return len(_tok.encode(s))
        except Exception:
            pass
    return max(1, len(s) // 4)

def _dehyphen_join(prev, nxt):
    if prev.endswith("-") and nxt[:1].islower():
        return prev[:-1] + nxt
    return prev + " " + nxt

def _detect_repeat_lines(pages_text, top_n=3, thresh=0.6):
    tops, bots = {}, {}
    for txt in pages_text:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        head = lines[:top_n]
        tail = lines[-top_n:]
        for l in head:
            tops[l] = tops.get(l, 0) + 1
        for l in tail:
            bots[l] = bots.get(l, 0) + 1
    n = len(pages_text)
    headers = {l for l,c in tops.items() if c / n >= thresh}
    footers = {l for l,c in bots.items() if c / n >= thresh}
    return headers, footers

def _page_blocks_in_reading_order(page):
    bs = page.get("blocks") or []
    if not bs:
        text = _norm(page.get("text",""))
        lines = [l for l in text.splitlines() if l.strip()]
        seq = [{"text": l.strip(), "bbox": [0, i*12, 1000, i*12+12], "fonts": []} for i,l in enumerate(lines)]
        return seq
    blocks = []
    for b in bs:
        t = _norm(b.get("text",""))
        if not t:
            continue
        x0,y0,x1,y1 = b.get("bbox",[0,0,0,0])
        fs = b.get("fonts", [])
        blocks.append({"text": t, "bbox": [x0,y0,x1,y1], "fonts": fs})
    if not blocks:
        return blocks
    xs = sorted([b["bbox"][0] for b in blocks])
    gaps = [xs[i+1]-xs[i] for i in range(len(xs)-1)]
    split = None
    if gaps:
        m = max(gaps)
        if m > CHAR_GAP_COL:
            i = gaps.index(m)
            split = (xs[i] + xs[i+1]) / 2.0
    if split is None:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1],1), round(b["bbox"][0],1)))
    left = [b for b in blocks if b["bbox"][0] < split]
    right = [b for b in blocks if b["bbox"][0] >= split]
    left = sorted(left, key=lambda b: (round(b["bbox"][1],1), round(b["bbox"][0],1)))
    right = sorted(right, key=lambda b: (round(b["bbox"][1],1), round(b["bbox"][0],1)))
    return left + right

def _mathiness_from_block(b):
    txt = b.get("text","")
    fonts = " ".join(b.get("fonts", []))
    f_hit = 1 if MATH_FONT_RX.search(fonts) else 0
    sym = len(re.findall(r"[=±×÷≤≥∑∏∫√∞∇∂^]", txt))
    dens = sym / max(1, len(txt))
    score = 0.6*f_hit + 0.4*min(1.0, 8*dens)
    return score

def _blocks_from_page(page, headers, footers):
    ordered = _page_blocks_in_reading_order(page)
    out = []
    for b in ordered:
        txt = b["text"]
        if txt in headers or txt in footers:
            continue
        b2 = dict(b)
        b2["mathiness"] = _mathiness_from_block(b)
        out.append(b2)
    return out

def _emit(chunks, buf, meta, has_math, has_table, start_idx, end_idx, heading_path, eq_flag, sect_tag):
    if not buf:
        return
    txt = " ".join(buf).strip()
    if not txt:
        return
    chunks.append({
        "text": txt,
        "page_start": meta["page"],
        "page_end": meta["page"],
        "doc_name": meta.get("doc_name"),
        "has_math": bool(has_math),
        "is_equation": bool(eq_flag),
        "section_tag": sect_tag,
        "char_start": start_idx,
        "char_end": end_idx,
        "heading_path": heading_path
    })

def _split_by_lists_and_headings(lines):
    blocks = []
    cur = []
    for ln in lines:
        if HEADING_RX.match(ln):
            if cur:
                blocks.append(("para", " ".join(cur).strip()))
                cur = []
            blocks.append(("heading", ln.strip()))
            continue
        if BULLET_RX.match(ln):
            if cur:
                blocks.append(("para", " ".join(cur).strip()))
                cur = []
            blocks.append(("bullet", ln.strip()))
            continue
        if ln.strip():
            cur.append(ln.strip())
        else:
            if cur:
                blocks.append(("para", " ".join(cur).strip()))
                cur = []
    if cur:
        blocks.append(("para", " ".join(cur).strip()))
    return blocks

def _section_tag(heading_path):
    hp = (heading_path or "").lower()
    if EXERCISE_RX.search(hp):
        return "exercises"
    return ""

def chunk_pages(pages):
    pages_text = [_norm(p.get("text","")) for p in pages]
    headers, footers = _detect_repeat_lines(pages_text)
    total_chars = sum(len(t) for t in pages_text)
    if len(pages) <= 2 and total_chars < 6000:
        tok_target = 520
        tok_overlap = 70
    else:
        tok_target = TOK_TARGET_DEFAULT
        tok_overlap = TOK_OVERLAP_DEFAULT

    chunks = []
    for p in pages:
        meta = {"page": p.get("page"), "doc_name": p.get("doc_name")}
        text = _norm(p.get("text",""))
        if not text:
            continue
        blocks = _blocks_from_page(p, headers, footers)
        if not blocks:
            blocks = [{"text": text, "bbox":[0,0,0,0], "fonts": [], "mathiness": 0.0}]
        page_buf = []
        for b in blocks:
            page_buf.append(b["text"])
        normalized = "\n".join(page_buf)
        lines = [l for l in normalized.splitlines()]
        blocks2 = _split_by_lists_and_headings(lines)

        heading_stack = []
        sect_tag = _section_tag(" > ".join(heading_stack))
        buf = []
        buf_tok = 0
        char_cursor = 0
        buf_start = 0
        has_math = False
        has_table = False
        eq_hint = any((b.get("mathiness",0) >= 0.6 and len((b.get("text") or "")) <= 220) for b in blocks)
        eq_flag = False

        for kind, content in blocks2:
            if kind == "heading":
                if buf:
                    _emit(chunks, buf, meta, has_math, has_table, buf_start, char_cursor, " > ".join(heading_stack), eq_flag, sect_tag)
                    buf, buf_tok, has_math, has_table = [], 0, False, False
                    eq_flag = False
                heading_stack = [content] if not heading_stack or HEADING_RX.match(content) else heading_stack + [content]
                sect_tag = _section_tag(" > ".join(heading_stack))
                char_cursor += len(content) + 1
                continue
            if kind == "bullet":
                if buf and buf_tok > tok_target * 0.8:
                    _emit(chunks, buf, meta, has_math, has_table, buf_start, char_cursor, " > ".join(heading_stack), eq_flag, sect_tag)
                    tail = " ".join(buf)
                    tail_toks = _tokens_len(tail)
                    keep_toks = min(tok_overlap, tail_toks // 2)
                    if keep_toks > 0 and _tok:
                        ids = _tok.encode(tail)
                        tail_text = _tok.decode(ids[-keep_toks:])
                    else:
                        tail_text = tail[-min(len(tail), tok_overlap*4):]
                    buf = [tail_text]
                    buf_tok = _tokens_len(tail_text)
                    buf_start = max(0, char_cursor - len(tail_text))
                    has_math = False
                    has_table = False
                    eq_flag = False
                buf.append(content)
                buf_tok += _tokens_len(content)
                has_math = has_math or bool(MATH_RX.search(content))
                has_table = has_table or bool(TABLE_HINT_RX.search(content)) or ("|" in content and len(content) > 10)
                if not eq_flag:
                    ops = len(re.findall(r"[=±×÷≤≥∑∏∫√∞∇∂^]", content))
                    words = len(re.findall(r"[A-Za-z]{3,}", content))
                    eq_flag = eq_hint and ops >= 1 and words <= 40
                char_cursor += len(content) + 1
                continue
            if kind == "para":
                sents = _sentences(content) if len(content) > 20 else [content]
                for s in sents:
                    s2 = s
                    if buf and s2 and buf[-1].endswith("-"):
                        s2 = _dehyphen_join(buf[-1], s2)
                        buf.pop()
                        buf_tok = _tokens_len(" ".join(buf))
                    stoks = _tokens_len(s2)
                    if buf_tok + stoks > tok_target and buf:
                        _emit(chunks, buf, meta, has_math, has_table, buf_start, char_cursor, " > ".join(heading_stack), eq_flag, sect_tag)
                        tail = " ".join(buf)
                        tail_toks = _tokens_len(tail)
                        keep_toks = min(tok_overlap, max(1, tail_toks // 2))
                        if keep_toks > 0 and _tok:
                            ids = _tok.encode(tail)
                            tail_text = _tok.decode(ids[-keep_toks:])
                        else:
                            tail_text = tail[-min(len(tail), tok_overlap*4):]
                        buf = [tail_text, s2]
                        buf_tok = _tokens_len(tail_text) + stoks
                        buf_start = max(0, char_cursor - len(tail_text))
                        has_math = bool(MATH_RX.search(s2))
                        has_table = bool(TABLE_HINT_RX.search(s2)) or ("|" in s2 and len(s2) > 10)
                        ops = len(re.findall(r"[=±×÷≤≥∑∏∫√∞∇∂^]", s2))
                        words = len(re.findall(r"[A-Za-z]{3,}", s2))
                        eq_flag = eq_hint and ops >= 1 and words <= 40
                    else:
                        if not buf:
                            buf_start = char_cursor
                        buf.append(s2)
                        buf_tok += stoks
                        has_math = has_math or bool(MATH_RX.search(s2))
                        has_table = has_table or bool(TABLE_HINT_RX.search(s2)) or ("|" in s2 and len(s2) > 10)
                        if not eq_flag:
                            ops = len(re.findall(r"[=±×÷≤≥∑∏∫√∞∇∂^]", s2))
                            words = len(re.findall(r"[A-Za-z]{3,}", s2))
                            eq_flag = eq_hint and ops >= 1 and words <= 40
                    char_cursor += len(s) + 1
        if buf:
            _emit(chunks, buf, meta, has_math, has_table, buf_start, char_cursor, " > ".join(heading_stack), eq_flag, sect_tag)
    return chunks