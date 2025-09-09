from typing import List, Dict

TARGET_TOKENS = 600
MIN_TOKENS = 200

def _approx_tokens(s: str):
    return max(1, len(s) // 4)

def _split_blocks(text: str):
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    return blocks

def chunk_pages(pages: List[Dict]):
    chunks: List[Dict] = []
    buf: List[str] = []
    tok = 0
    start_page = None

    for p in pages:
        blocks = _split_blocks(p["text"]) or [""]
        for b in blocks:
            bt = _approx_tokens(b)
            if start_page is None:
                start_page = p["page"]
            if tok + bt > TARGET_TOKENS and tok >= MIN_TOKENS:
                chunks.append({
                    "text": "\n\n".join(buf),
                    "page_start": start_page,
                    "page_end": p["page"],
                })
                buf, tok = [], 0
                start_page = p["page"]
            buf.append(b); tok += bt

        if tok >= TARGET_TOKENS:
            chunks.append({
                "text": "\n\n".join(buf),
                "page_start": start_page,
                "page_end": p["page"],
            })
            buf, tok, start_page = [], 0, None

    if buf:
        chunks.append({
            "text": "\n\n".join(buf),
            "page_start": start_page or 1,
            "page_end": pages[-1]["page"] if pages else 1,
        })
    return chunks
