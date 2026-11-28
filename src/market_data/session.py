"""Dedicated market-data engine and lazy request-scoped sessions."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from pathlib import Path

from fastapi import HTTPException, Request, status
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_DATA_DATABASE_URL = "sqlite:///./data/market_data.db"


def _database_url() -> str:
    return os.environ.get(
        "MARKET_DATA_DATABASE_URL", DEFAULT_MARKET_DATA_DATABASE_URL
    ).strip()


def _make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    connect_args = {"check_same_thread": False} if url.get_backend_name() == "sqlite" else {}
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
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_market_db(request: Request) -> Generator[Session, None, None]:
    """Initialize the independent price schema on first use and yield a session."""
    if not getattr(request.app.state, "market_data_ready", False):
        from src.market_data.migrations import initialize_market_database

        try:
            initialize_market_database()
        except Exception as error:
            request.app.state.market_data_ready = False
            logging.getLogger(__name__).exception(
                "Market-data storage initialization failed"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Market-data storage is unavailable",
            ) from error
        request.app.state.market_data_ready = True
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
