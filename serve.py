"""
Unified ASGI entrypoint that serves both the API and Web UI on one process.

Start with:
  uvicorn serve:app --host 0.0.0.0 --port $PORT

On Render, set the Start Command to the above.
"""
# Load .env file before anything else
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

# Import the Web UI app
from web.app import app as web_app
from api.db.database import init_db
from api.routes import (
    upload as upload_routes,
    query as query_routes,
    secrets as secrets_routes,
    auth as auth_routes,
    ingest as ingest_routes,
    library as library_routes,
    settings as settings_routes,
    chat as chat_routes,
    ask as ask_routes,
    admin as admin_routes,
)

# Compose a single app: /api for the backend, / for the UI
app = FastAPI()

# Mount API first to ensure /api/* takes precedence over the root mount
# Register API routers at their native paths (which already include /api prefixes)
app.include_router(auth_routes.router)
app.include_router(secrets_routes.router)
app.include_router(upload_routes.router)
app.include_router(query_routes.router)
app.include_router(ingest_routes.router)
app.include_router(library_routes.router)
app.include_router(settings_routes.router)
app.include_router(chat_routes.router)
app.include_router(ask_routes.router)
app.include_router(admin_routes.router)

# Mount the web UI at root
app.mount("/", web_app)

@app.on_event("startup")
def _startup_once():
    # Ensure DB tables exist when running as a single process
    try:
        init_db()
    except Exception:
        pass
