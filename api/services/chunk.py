from typing import List, Dict

TARGET_TOKENS = 600
MIN_TOKENS = 200

def _approx_tokens(s: str):
    return max(1, len(s) // 4)

def _split_blocks(text: str) -> List[str]:
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    return blocks

def chunk_pages(pages: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    buf: List[str] = []
    tok = 0
    start_page = None
    start_doc = None

    for p in pages:
        blocks = _split_blocks(p["text"]) or [""]
        for b in blocks:
            bt = _approx_tokens(b)
            if start_page is None:
                start_page = p["page"]
                start_doc = p.get("doc_name")
            if tok + bt > TARGET_TOKENS and tok >= MIN_TOKENS:
                chunks.append({
                    "text": "\n\n".join(buf),
                    "page_start": start_page,
                    "page_end": p["page"],
                    "doc_name": start_doc,
                })
                buf, tok = [], 0
                start_page = p["page"]
                start_doc = p.get("doc_name")
            buf.append(b); tok += bt

        if tok >= TARGET_TOKENS:
            chunks.append({
                "text": "\n\n".join(buf),
                "page_start": start_page,
                "page_end": p["page"],
                "doc_name": start_doc,
            })
            buf, tok, start_page, start_doc = [], 0, None, None

    if buf:
        chunks.append({
            "text": "\n\n".join(buf),
            "page_start": start_page or 1,
            "page_end": pages[-1]["page"] if pages else 1,
            "doc_name": start_doc,
        })
    return chunks
