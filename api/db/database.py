import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy.exc

DATABASE_URL = os.getenv("DATABASE_URL")

# Allow forcing SQLite for local single-process runs where a Docker-only
# DATABASE_URL might be present in the environment.
if os.getenv("FORCE_SQLITE", "").strip().lower() in {"1", "true", "yes", "on"}:
    DATABASE_URL = None

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./anagnosis.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # allow rebinding the module-level engine/session when falling back
    global engine, SessionLocal

    try:
        Base.metadata.create_all(bind=engine)
    except sqlalchemy.exc.OperationalError as exc:
        # Could not connect to the configured database (common when running
        # locally without Docker). Fall back to a local SQLite file so the
        # application can start for development/testing.
        logging.getLogger(__name__).warning(
            "Database connection failed (%s). Falling back to local SQLite.", exc
        )
        fallback_url = "sqlite:///./anagnosis.db"
        fallback_engine = create_engine(
            fallback_url, connect_args={"check_same_thread": False}
        )
        # Rebind module-level objects so get_db()/SessionLocal use the fallback engine
        engine = fallback_engine
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
    except Exception:
        # Re-raise unexpected exceptions
        raise
