"""Versioned, additive migrations for the dedicated market-data database."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, inspect, text

from src.market_data import models as _models  # noqa: F401
from src.market_data.base import MarketDataBase
from src.market_data.session import engine as default_engine

_REVISION = "0001_eod_prices"


def _create_eod_prices(connection) -> None:
    MarketDataBase.metadata.create_all(connection)


def initialize_market_database(target_engine: Engine | None = None) -> None:
    """Create the version table and apply pending additive revisions."""
    selected_engine = target_engine or default_engine
    if selected_engine.dialect.name not in {"sqlite", "postgresql"}:
        raise RuntimeError(
            f"Market-data migrations do not support {selected_engine.dialect.name!r}"
        )
    if selected_engine.dialect.name == "sqlite":
        database_path = selected_engine.url.database
        if database_path and database_path != ":memory:":
            Path(database_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )

    existing_tables = set(inspect(selected_engine).get_table_names())
    if (
        "market_data_schema_migrations" not in existing_tables
        and "eod_prices" in existing_tables
    ):
        raise RuntimeError(
            "Market-data table eod_prices exists without a recorded "
            "migration revision; review the database before migrating."
        )

    with selected_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS market_data_schema_migrations ("
                "revision VARCHAR(100) PRIMARY KEY, "
                "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        revisions = set(
            connection.execute(
                text("SELECT revision FROM market_data_schema_migrations")
            ).scalars()
        )
        unknown = revisions - {_REVISION}
        if unknown:
            raise RuntimeError(
                "Market-data database contains unknown schema revisions: "
                + ", ".join(sorted(unknown))
            )
        if _REVISION not in revisions:
            _create_eod_prices(connection)
            connection.execute(
                text(
                    "INSERT INTO market_data_schema_migrations (revision) "
                    "VALUES (:revision)"
                ),
                {"revision": _REVISION},
            )
        elif "eod_prices" not in inspect(connection).get_table_names():
            raise RuntimeError(
                "Market-data migration 0001 is recorded but eod_prices is "
                "missing; restore the market-data schema before retrying."
            )
