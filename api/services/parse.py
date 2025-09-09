import io, pathlib
from typing import List, Dict
import fitz
from PIL import Image
import pytesseract

def _page_needs_ocr(doc, i: int):
    page = doc.load_page(i)
    text = page.get_text("text").strip()
    return len(text) < 10

def _ocr_page(doc, i: int, dpi: int = 300):
    page = doc.load_page(i)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes()))
    return pytesseract.image_to_string(img, config="--psm 4")

def parse_pdf_bytes(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Dict] = []
    ocr_pages = 0
    for i in range(len(doc)):
        if _page_needs_ocr(doc, i):
            text = _ocr_page(doc, i)
            ocr_pages += 1
        else:
            text = doc.load_page(i).get_text("text")
        pages.append({"page": i + 1, "text": text})
    return {"num_pages": len(doc), "ocr_pages": ocr_pages, "pages": pages}

def parse_any_bytes(filename: str, data: bytes):
    ext = pathlib.Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf_bytes(data)
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img, config="--psm 4")
        return {"num_pages": 1, "ocr_pages": 1, "pages": [{"page": 1, "text": text}]}
    raise ValueError(f"Unsupported file type: {ext}")
