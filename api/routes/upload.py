from fastapi import APIRouter, UploadFile, HTTPException, Depends
import pathlib
from api.services.parse import parse_pdf_bytes
from api.services.chunk import chunk_pages
from api.services.index import add_chunks
from api.auth.middleware import require_auth
from api.db.models import User

router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("/")
async def upload_pdf(file: UploadFile, user: User = Depends(require_auth)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    pdf_bytes = await file.read()
    try:
        safe_name = pathlib.Path(file.filename).name
        base = pathlib.Path("artifacts") / "docs" / str(user.id)
        base.mkdir(parents=True, exist_ok=True)
        (base / safe_name).write_bytes(pdf_bytes)
    except Exception:
        pass
    parsed = parse_pdf_bytes(pdf_bytes)
    chunks = chunk_pages(parsed["pages"])
    ids = add_chunks(chunks, user_id=str(user.id))
    return {
        "ok": True,
        "num_pages": parsed["num_pages"],
        "ocr_pages": parsed["ocr_pages"],
        "num_chunks": len(chunks),
        "ids_start": ids[0] if ids else None
    }
