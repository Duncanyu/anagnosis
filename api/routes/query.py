from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from api.services.index import search
from api.services.summarize import summarize
from api.auth.middleware import require_auth
from api.db.models import User

router = APIRouter(prefix="/query", tags=["query"])

class QueryRequest(BaseModel):
    question: str
    k: int = 5

@router.post("/")
async def query_documents(request: QueryRequest, user: User = Depends(require_auth)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    hits = search(request.question, k=request.k, user_id=str(user.id))
    if not hits:
        return {"answer": "No results found. Please upload documents first.", "citations": []}
    
    top_chunks = [h[1] for h in hits]
    result = summarize(request.question, top_chunks)
    
    return result
