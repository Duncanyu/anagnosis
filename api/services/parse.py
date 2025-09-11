import io, pathlib, re, unicodedata, tempfile, subprocess, shutil, json, os, time
import fitz
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uF900-\uFAFF\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]')
PUA_RE = re.compile(r'[\uE000-\uF8FF]')

ALT_MAX_TIMEOUT = 3.0
ALT_DISABLE_AFTER = 5
ALT_SLOW_AVG = 0.6

def _nfkc(s):
    try:
        return unicodedata.normalize('NFKC', s or '')
    except Exception:
        return s or ''

def _looks_bad(s):
    if not s:
        return True
    if CJK_RE.search(s):
        return True
    return False

def _ascii_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if ord(ch) < 128)
    return n / len(s)

def _suspicious(s):
    if len(s) < 10:
        return True
    if _looks_bad(s):
        return True
    if PUA_RE.search(s):
        return True
    if _ascii_ratio(s) < 0.55:
        return True
    return False

def _extract_text_mupdf(doc, i):
    page = doc.load_page(i)
    txt = page.get_text("text") or ""
    return _nfkc(txt).strip()

def _extract_text_pdfminer(pdf_bytes, i):
    t0 = time.perf_counter()
    try:
        from pdfminer.high_level import extract_text
        from io import BytesIO
        t = extract_text(BytesIO(pdf_bytes), page_numbers=[i]) or ""
        out = _nfkc(t).strip()
    except Exception:
        return ""
    dt = time.perf_counter() - t0
    return out, dt

def _extract_text_poppler(pdf_bytes, i):
    if not shutil.which("pdftotext"):
        return "", 0.0
    tf = None
    of = None
    t0 = time.perf_counter()
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf.write(pdf_bytes); tf.flush()
        of = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        of.close()
        subprocess.run(
            ["pdftotext","-layout","-q","-f",str(i+1),"-l",str(i+1),tf.name, of.name],
            check=False, timeout=ALT_MAX_TIMEOUT
        )
        s = pathlib.Path(of.name).read_text(encoding="utf-8", errors="ignore")
        out = _nfkc(s).strip()
    except Exception:
        return "", ALT_MAX_TIMEOUT + 1.0
    finally:
        try:
            if tf: os.unlink(tf.name)
            if of: os.unlink(of.name)
        except Exception:
            pass
    dt = time.perf_counter() - t0
    return out, dt

def _available_langs():
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception:
        return {"eng"}

def _prep_image(img, scale=2.0):
    w, h = img.size
    img = img.convert("L")
    img = img.resize((int(w*scale), int(h*scale)))
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img

def _ocr_span(img):
    langs = _available_langs()
    lang = "eng+equ" if "equ" in langs else "eng"
    best = ""
    best_len = -1
    for psm in (7,6):
        cfg = f"--psm {psm} -l {lang} -c preserve_interword_spaces=1"
        t = _nfkc(pytesseract.image_to_string(img, config=cfg)).strip()
        if len(t) > best_len and not _looks_bad(t):
            best, best_len = t, len(t)
    return best or _nfkc(pytesseract.image_to_string(img, config=f"--psm 6 -l {lang}")).strip()

def _ocr_page(doc, i, dpi=400):
    page = doc.load_page(i)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes()))
    img2 = _prep_image(img, scale=2.2)
    return _ocr_span(img2)

_FONT_MAPS = None

def _load_font_maps():
    global _FONT_MAPS
    if _FONT_MAPS is not None:
        return _FONT_MAPS
    maps = {}
    base = pathlib.Path("artifacts") / "font_maps"
    try:
        if base.exists():
            for fp in base.glob("*.json"):
                try:
                    d = json.loads(fp.read_text(encoding="utf-8"))
                    m = {}
                    for k, v in d.items():
                        try:
                            code = int(k, 16) if isinstance(k, str) else int(k)
                            m[code] = v
                        except Exception:
                            pass
                    maps[fp.stem.lower()] = m
                except Exception:
                    pass
    except Exception:
        pass
    _FONT_MAPS = maps
    return _FONT_MAPS

def _rebuild_with_pua_maps(doc, i):
    maps = _load_font_maps()
    page = doc.load_page(i)
    rd = page.get_text("rawdict")
    if not isinstance(rd, dict):
        return ""
    lines = []
    for b in rd.get("blocks", []):
        for ln in b.get("lines", []):
            parts = []
            for sp in ln.get("spans", []):
                t = _nfkc(sp.get("text") or "")
                if not t:
                    continue
                if PUA_RE.search(t):
                    font = (sp.get("font") or "").lower()
                    mp = maps.get(font)
                    if mp:
                        chars = []
                        for ch in t:
                            o = ord(ch)
                            chars.append(mp.get(o, ch))
                        parts.append("".join(chars))
                    else:
                        parts.append(t)
                else:
                    parts.append(t)
            line = "".join(parts).strip()
            if line:
                lines.append(line)
    return _nfkc("\n".join(lines)).strip()

def _salvage_bad_spans(doc, i, dpi=400, pad=2):
    page = doc.load_page(i)
    rd = page.get_text("rawdict")
    if not isinstance(rd, dict):
        return ""
    lines = []
    for b in rd.get("blocks", []):
        for ln in b.get("lines", []):
            parts = []
            for sp in ln.get("spans", []):
                t = _nfkc(sp.get("text") or "")
                if not t:
                    continue
                if _looks_bad(t) or PUA_RE.search(t):
                    x0,y0,x1,y1 = sp.get("bbox",[0,0,0,0])
                    r = fitz.Rect(x0-pad, y0-pad, x1+pad, y1+pad)
                    pix = page.get_pixmap(clip=r, dpi=dpi, alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    img2 = _prep_image(img, scale=2.2)
                    o = _ocr_span(img2).strip()
                    parts.append(o if o else t)
                else:
                    parts.append(t)
            line = "".join(parts).strip()
            if line:
                lines.append(line)
    return _nfkc("\n".join(lines)).strip()

_alt_state = {"pdfminer": {"n": 0, "tot": 0.0, "enabled": True}, "poppler": {"n": 0, "tot": 0.0, "enabled": True}}

def _maybe_alt_extract(pdf_bytes, i):
    picks = []
    if _alt_state["pdfminer"]["enabled"]:
        txt_dt = _extract_text_pdfminer(pdf_bytes, i)
        if isinstance(txt_dt, tuple):
            t, dt = txt_dt
            _alt_state["pdfminer"]["n"] += 1
            _alt_state["pdfminer"]["tot"] += dt
            avg = _alt_state["pdfminer"]["tot"] / max(1, _alt_state["pdfminer"]["n"])
            if avg > ALT_SLOW_AVG or _alt_state["pdfminer"]["n"] >= ALT_DISABLE_AFTER:
                _alt_state["pdfminer"]["enabled"] = False
            if t:
                picks.append(("pdfminer", t))
    if _alt_state["poppler"]["enabled"]:
        t, dt = _extract_text_poppler(pdf_bytes, i)
        _alt_state["poppler"]["n"] += 1
        _alt_state["poppler"]["tot"] += dt
        avg = _alt_state["poppler"]["tot"] / max(1, _alt_state["poppler"]["n"])
        if dt > ALT_MAX_TIMEOUT or avg > ALT_SLOW_AVG or _alt_state["poppler"]["n"] >= ALT_DISABLE_AFTER:
            _alt_state["poppler"]["enabled"] = False
        if t:
            picks.append(("poppler", t))
    return picks

def _score(s):
    if not s:
        return -1e9
    if _looks_bad(s):
        return -1e6
    a = _ascii_ratio(s)
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    return 5*a + 0.001*len(s) + 0.0005*letters + 0.0003*digits

def _log(cb, s):
    if cb:
        try:
            cb(s)
        except Exception:
            pass

def parse_pdf_bytes(pdf_bytes, progress_cb=None):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    ocr_pages = 0
    ocr_page_numbers = []
    suspect_pages = []
    salvaged_pages = []
    glyphmap_pages = []
    alt_pick_pages = []
    n = len(doc)
    _log(progress_cb, f"Pages: {n}")
    for i in range(n):
        if i == 0 or (i + 1) % 10 == 0 or i == n - 1:
            _log(progress_cb, f"p {i+1}/{n}")
        base = _extract_text_mupdf(doc, i)
        if not _suspicious(base):
            text = base
            suspect = False
        else:
            pua_fixed = _rebuild_with_pua_maps(doc, i) if PUA_RE.search(base) else ""
            cands = [("mupdf", base)]
            if pua_fixed:
                cands.append(("pua", pua_fixed))
            for name, t in _maybe_alt_extract(pdf_bytes, i):
                cands.append((name, t))
            best_name, best_text, best_score = None, "", -1e9
            for name, t in cands:
                sc = _score(t)
                if sc > best_score:
                    best_name, best_text, best_score = name, t, sc
            if best_name in ("pdfminer","poppler"):
                alt_pick_pages.append(i + 1)
                _log(progress_cb, f"p {i+1}: alt {best_name}")
            if best_name == "pua":
                glyphmap_pages.append(i + 1)
                _log(progress_cb, f"p {i+1}: glyphmap")
            if len(best_text) < 10 or _looks_bad(best_text) or _suspicious(best_text):
                salv = _salvage_bad_spans(doc, i)
                if salv and not _looks_bad(salv):
                    text = salv
                    salvaged_pages.append(i + 1)
                    _log(progress_cb, f"p {i+1}: salvaged")
                    suspect = False
                else:
                    if len(base) < 10:
                        text = _ocr_page(doc, i)
                        ocr_pages += 1
                        ocr_page_numbers.append(i + 1)
                        _log(progress_cb, f"p {i+1}: OCR")
                        suspect = False
                    else:
                        text = best_text or base
                        suspect = _looks_bad(text)
                        if suspect:
                            _log(progress_cb, f"p {i+1}: suspect")
            else:
                text = best_text
                suspect = False
        pages.append({"page": i + 1, "text": text, "suspect": suspect})
        if suspect:
            suspect_pages.append(i + 1)
    _log(progress_cb, "Parsing done")
    return {
        "num_pages": n,
        "ocr_pages": ocr_pages,
        "ocr_page_numbers": ocr_page_numbers,
        "suspect_pages": suspect_pages,
        "salvaged_pages": salvaged_pages,
        "glyphmap_pages": glyphmap_pages,
        "alt_extractor_pages": alt_pick_pages,
        "pages": pages
    }

def parse_any_bytes(filename, data, progress_cb=None):
    ext = pathlib.Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf_bytes(data, progress_cb=progress_cb)
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        img = Image.open(io.BytesIO(data))
        t = pytesseract.image_to_string(_prep_image(img), config="--psm 6 -l eng")
        return {"num_pages": 1, "ocr_pages": 1, "ocr_page_numbers": [1], "suspect_pages": [], "salvaged_pages": [], "glyphmap_pages": [], "alt_extractor_pages": [], "pages": [{"page": 1, "text": _nfkc(t).strip(), "suspect": False}]}
    raise ValueError("Unsupported file type: " + ext)