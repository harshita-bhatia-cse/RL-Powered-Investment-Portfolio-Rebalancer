"""SQLAlchemy engine and request-scoped sessions for account data."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Generator

from fastapi import HTTPException, Request, status
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_PORTFOLIO_DATABASE_URL = "sqlite:///./data/portfolio_platform.db"


def _database_url() -> str:
    return os.environ.get(
        "PORTFOLIO_DATABASE_URL", DEFAULT_PORTFOLIO_DATABASE_URL
    ).strip()


def _make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    connect_args = {}
    if url.get_backend_name() == "sqlite":
        connect_args["check_same_thread"] = False
    created_engine = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    if url.get_backend_name() == "sqlite":
        @event.listens_for(created_engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return created_engine


engine = _make_engine(_database_url())
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def get_db(request: Request) -> Generator[Session, None, None]:
    """Yield and close one database session per request."""
    if not getattr(request.app.state, "account_database_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account persistence is unavailable",
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
