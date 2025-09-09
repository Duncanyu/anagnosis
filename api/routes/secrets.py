from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from api.core.config import save_secret, present_keys, WANTED_KEYS

router = APIRouter(prefix="/secrets", tags=["secrets"])

class SecretIn(BaseModel):
    name: str = Field(..., description="e.g. OPENAI_API_KEY")
    value: str = Field(..., description="the secret value")
    storage: str = Field("keyring", description="keyring|file")

@router.get("/")
def get_status():
    return {"configured": present_keys()}

@router.post("/set")
def set_secret(secret: SecretIn):
    if secret.name not in WANTED_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown key: {secret.name}")
    save_secret(secret.name, secret.value, prefer=secret.storage)
    return {"ok": True}
