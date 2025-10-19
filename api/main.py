from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import upload, query, secrets, auth
from api.routes import ingest as ingest_routes
from api.routes import library as library_routes
from api.routes import settings as settings_routes
from api.routes import chat as chat_routes
from api.routes import ask as ask_routes
from api.routes import admin as admin_routes
from api.db.database import init_db

app = FastAPI(title="Anagnosis — Multi-tenant Document Assistant")

app.add_middleware(
    CORSMiddleware,
    # Be explicit to avoid wildcard + credentials issues in browsers
    allow_origins=["http://localhost:7860", "http://127.0.0.1:7860"],
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
app.include_router(ingest_routes.router)
app.include_router(library_routes.router)
app.include_router(settings_routes.router)
app.include_router(chat_routes.router)
app.include_router(ask_routes.router)
app.include_router(admin_routes.router)

@app.get("/")
def root():
    return {"ok": True, "service": "anagnosis", "version": "2.0-multitenancy"}
