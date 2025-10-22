from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os, time, logging
import os
from api.routes import upload, query, secrets, auth
from api.routes import ingest as ingest_routes
from api.routes import library as library_routes
from api.routes import settings as settings_routes
from api.routes import chat as chat_routes
from api.routes import ask as ask_routes
from api.routes import admin as admin_routes
from api.db.database import init_db

def _setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "TRACE": logging.DEBUG,
    }
    log_level = level_map.get(level, logging.INFO)
    try:
        logging.getLogger().setLevel(log_level)
        logging.getLogger("uvicorn").setLevel(log_level)
        logging.getLogger("uvicorn.error").setLevel(log_level)
        logging.getLogger("uvicorn.access").setLevel(log_level)
    except Exception:
        pass


_setup_logging()

app = FastAPI(title="Anagnosis — Multi-tenant Document Assistant")

origins_env = os.getenv("ALLOW_ORIGINS")
if origins_env:
    allow_list = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    allow_list = [
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    # Sensible defaults for small single-instance deployments (Phase 4)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("ASK_TIME_BUDGET_SEC", "120")
    os.environ.setdefault("ASK_MAX_BATCHES", "6")
    os.environ.setdefault("ASK_CANDIDATES", "300")
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

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Alias for clients expecting an /api/ prefix
@app.get("/api/healthz")
def healthz_api():
    return {"status": "ok"}

"""Lightweight access logging (method path status latency).
Skips noisy endpoints such as streaming status polls and health checks.
"""
SKIP_LOG_PREFIXES = ["/api/ingest/status", "/api/ask/status", "/healthz", "/api/healthz"]

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    path = request.url.path
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    try:
        if not any(path.startswith(p) for p in SKIP_LOG_PREFIXES):
            code = getattr(resp, "status_code", None)
            logging.getLogger("uvicorn.access").info("%s %s %s %dms", request.method, path, code, dt_ms)
    except Exception:
        pass
    return resp
