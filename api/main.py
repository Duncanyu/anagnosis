from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import upload, query, secrets

app = FastAPI(title="Reading Agent — Quick MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(secrets.router)
app.include_router(upload.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"ok": True, "service": "reading-agent-mvp"}
