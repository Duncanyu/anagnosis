from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import upload, query, secrets, auth
from api.db.database import init_db

app = FastAPI(title="Anagnosis — Multi-tenant Document Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://127.0.0.1:7860", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(auth.router)
app.include_router(secrets.router)
app.include_router(upload.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"ok": True, "service": "anagnosis", "version": "2.0-multitenancy"}
