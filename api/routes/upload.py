from fastapi import APIRouter, UploadFile, HTTPException
from api.services.parse import parse_pdf_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    pdf_bytes = await file.read()
    parsed = parse_pdf_bytes(pdf_bytes)
    chunks = chunk_pages(parsed["pages"])
    ids = add_chunks(chunks)
    return {
        "ok": True,
        "num_pages": parsed["num_pages"],
        "ocr_pages": parsed["ocr_pages"],
        "num_chunks": len(chunks),
        "ids_start": ids[0] if ids else None
    }
