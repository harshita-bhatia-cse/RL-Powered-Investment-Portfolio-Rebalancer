"""Small versioned schema initializer for the account database.

Run ``python -m src.database.migrations`` to create/apply pending migrations.
Future schema changes should add a new revision rather than modify an applied
revision. The legacy research DataStore is intentionally not migrated here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sqlalchemy import Engine, inspect, text

from src.database import models as _models  # noqa: F401
from src.database.base import Base
from src.database.session import engine as default_engine


_TRANSACTION_FIELDS_VALID = (
    "("
    "transaction_type IN ('BUY', 'SELL') "
    "AND exchange IS NOT NULL AND exchange IN ('NSE', 'BSE') "
    "AND symbol IS NOT NULL "
    "AND quantity IS NOT NULL AND quantity > 0 "
    "AND unit_price IS NOT NULL AND unit_price > 0 "
    "AND cash_amount IS NULL"
    ") OR ("
    "transaction_type IN ('CASH_DEPOSIT', 'CASH_WITHDRAWAL') "
    "AND exchange IS NULL AND symbol IS NULL "
    "AND quantity IS NULL AND unit_price IS NULL "
    "AND cash_amount IS NOT NULL AND cash_amount > 0"
    ")"
)


def _create_pre_monitoring_account_schema(connection) -> None:
    monitoring_table = _models.PortfolioMonitoringReport.__table__
    Base.metadata.create_all(
        connection,
        tables=[
            table
            for table in Base.metadata.sorted_tables
            if table is not monitoring_table
        ],
    )


def _upgrade_account_foundation(connection) -> None:
    _create_pre_monitoring_account_schema(connection)


def _upgrade_transaction_constraints(connection) -> None:
    invalid_rows = connection.execute(
        text(
            "SELECT id FROM transactions "
            "WHERE CASE WHEN ("
            + _TRANSACTION_FIELDS_VALID
            + ") THEN 0 ELSE 1 END = 1 LIMIT 10"
        )
    ).scalars().all()
    if invalid_rows:
        raise RuntimeError(
            "Cannot apply account migration 0002: transactions contain rows "
            "that violate the stricter transaction-field rules. Review and "
            "correct these records through an approved data-repair process, "
            "then retry. Affected transaction IDs (up to 10): "
            + ", ".join(str(row_id) for row_id in invalid_rows)
        )

    dialect_name = connection.dialect.name
    if dialect_name == "sqlite":
        trigger_condition = _TRANSACTION_FIELDS_VALID
        for column in (
            "transaction_type",
            "exchange",
            "symbol",
            "quantity",
            "unit_price",
            "cash_amount",
        ):
            trigger_condition = re.sub(
                rf"\b{column}\b",
                f"NEW.{column}",
                trigger_condition,
            )
        for operation, trigger_name in (
            ("INSERT", "trg_transactions_validate_fields_insert"),
            (
                "UPDATE OF transaction_type, exchange, symbol, quantity, "
                "unit_price, cash_amount",
                "trg_transactions_validate_fields_update",
            ),
        ):
            connection.execute(
                text(
                    f"CREATE TRIGGER IF NOT EXISTS {trigger_name} "
                    f"BEFORE {operation} ON transactions "
                    "FOR EACH ROW WHEN CASE WHEN ("
                    + trigger_condition
                    + ") THEN 0 ELSE 1 END = 1 "
                    "BEGIN SELECT RAISE(ABORT, "
                    "'transaction fields do not match transaction type'); END"
                )
            )
    elif dialect_name == "postgresql":
        connection.execute(
            text(
                "ALTER TABLE transactions ADD CONSTRAINT "
                "ck_transaction_fields_match_type_v2 CHECK ("
                + _TRANSACTION_FIELDS_VALID
                + ")"
            )
        )
    else:
        raise RuntimeError(
            f"Account migration 0002 does not support database dialect "
            f"{dialect_name!r}"
        )


def _upgrade_portfolio_ledger_baseline(connection) -> None:
    required_tables = {"portfolios", "holdings", "transactions"}
    existing_tables = set(inspect(connection).get_table_names())
    missing_tables = required_tables - existing_tables
    if missing_tables:
        raise RuntimeError(
            "Cannot apply account migration 0003: required account tables are "
            "missing: "
            + ", ".join(sorted(missing_tables))
            + ". Restore the account schema before retrying."
        )

    invalid_cash_fees = connection.execute(
        text(
            "SELECT id FROM transactions "
            "WHERE transaction_type IN ('CASH_DEPOSIT', 'CASH_WITHDRAWAL') "
            "AND fees <> 0 LIMIT 10"
        )
    ).scalars().all()
    if invalid_cash_fees:
        raise RuntimeError(
            "Cannot apply account migration 0003: cash transactions with "
            "non-zero fees cannot be replayed unambiguously. Review these "
            "historical records and resolve them through an approved process; "
            "the migration will not rewrite them. Affected transaction IDs "
            "(up to 10): "
            + ", ".join(str(row_id) for row_id in invalid_cash_fees)
        )

    transaction_columns = {
        column["name"] for column in inspect(connection).get_columns("transactions")
    }
    if "posted_at" not in transaction_columns:
        if connection.dialect.name == "sqlite":
            connection.execute(
                text("ALTER TABLE transactions ADD COLUMN posted_at DATETIME")
            )
        elif connection.dialect.name == "postgresql":
            connection.execute(
                text(
                    "ALTER TABLE transactions ADD COLUMN posted_at "
                    "TIMESTAMP WITH TIME ZONE"
                )
            )
        else:
            raise RuntimeError(
                "Account migration 0003 supports SQLite and PostgreSQL only; "
                f"found {connection.dialect.name!r}"
            )

    if connection.dialect.name == "sqlite":
        for operation, trigger_name in (
            ("INSERT", "trg_transactions_validate_cash_fees_insert"),
            (
                "UPDATE OF transaction_type, fees",
                "trg_transactions_validate_cash_fees_update",
            ),
        ):
            connection.execute(
                text(
                    f"CREATE TRIGGER IF NOT EXISTS {trigger_name} "
                    f"BEFORE {operation} ON transactions "
                    "FOR EACH ROW WHEN NEW.transaction_type IN "
                    "('CASH_DEPOSIT', 'CASH_WITHDRAWAL') AND NEW.fees <> 0 "
                    "BEGIN SELECT RAISE(ABORT, "
                    "'cash transactions do not support fees'); END"
                )
            )
    elif connection.dialect.name == "postgresql":
        constraint_exists = any(
            constraint.get("name") == "ck_transaction_cash_fees_zero"
            for constraint in inspect(connection).get_check_constraints(
                "transactions"
            )
        )
        if not constraint_exists:
            connection.execute(
                text(
                    "ALTER TABLE transactions ADD CONSTRAINT "
                    "ck_transaction_cash_fees_zero CHECK ("
                    "transaction_type IN ('BUY', 'SELL') OR fees = 0)"
                )
            )
    else:
        raise RuntimeError(
            "Account migration 0003 supports SQLite and PostgreSQL only; "
            f"found {connection.dialect.name!r}"
        )

    _create_pre_monitoring_account_schema(connection)


def _upgrade_transaction_idempotency(connection) -> None:
    if "transactions" not in inspect(connection).get_table_names():
        raise RuntimeError(
            "Cannot apply account migration 0004: transactions table is missing. "
            "Restore the account schema before retrying."
        )
    columns = {
        column["name"] for column in inspect(connection).get_columns("transactions")
    }
    if "idempotency_key" not in columns:
        if connection.dialect.name == "sqlite":
            connection.execute(
                text("ALTER TABLE transactions ADD COLUMN idempotency_key VARCHAR(128)")
            )
        elif connection.dialect.name == "postgresql":
            connection.execute(
                text(
                    "ALTER TABLE transactions ADD COLUMN idempotency_key "
                    "VARCHAR(128)"
                )
            )
        else:
            raise RuntimeError(
                "Account migration 0004 supports SQLite and PostgreSQL only; "
                f"found {connection.dialect.name!r}"
            )

    duplicate_keys = connection.execute(
        text(
            "SELECT portfolio_id, idempotency_key, COUNT(*) "
            "FROM transactions "
            "WHERE idempotency_key IS NOT NULL "
            "GROUP BY portfolio_id, idempotency_key "
            "HAVING COUNT(*) > 1 LIMIT 10"
        )
    ).all()
    if duplicate_keys:
        raise RuntimeError(
            "Cannot apply account migration 0004: duplicate non-null "
            "idempotency keys already exist within a portfolio. Resolve the "
            "ambiguous keys through an approved process; no transactions were "
            "changed. Conflicts (portfolio_id, idempotency_key, count): "
            + "; ".join(
                f"({portfolio_id}, {key}, {count})"
                for portfolio_id, key, count in duplicate_keys
            )
        )

    existing_unique_constraints = inspect(connection).get_unique_constraints(
        "transactions"
    )
    existing_unique_indexes = inspect(connection).get_indexes("transactions")
    key_is_unique = any(
        constraint.get("column_names") == ["portfolio_id", "idempotency_key"]
        for constraint in existing_unique_constraints
    ) or any(
        index.get("unique")
        and index.get("column_names") == ["portfolio_id", "idempotency_key"]
        for index in existing_unique_indexes
    )
    if not key_is_unique:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX uq_transactions_portfolio_idempotency_key "
                "ON transactions (portfolio_id, idempotency_key)"
            )
        )


def _upgrade_portfolio_monitoring_reports(connection) -> None:
    if connection.dialect.name not in {"sqlite", "postgresql"}:
        raise RuntimeError(
            "Account migration 0005 supports SQLite and PostgreSQL only; "
            f"found {connection.dialect.name!r}"
        )
    if "portfolios" not in inspect(connection).get_table_names():
        raise RuntimeError(
            "Cannot apply account migration 0005: portfolios table is missing. "
            "Restore the account schema before retrying."
        )
    _models.PortfolioMonitoringReport.__table__.create(
        connection,
        checkfirst=True,
    )


def _upgrade_portfolio_exposure_risk(connection) -> None:
    if connection.dialect.name not in {"sqlite", "postgresql"}:
        raise RuntimeError(
            "Account migration 0006 supports SQLite and PostgreSQL only; "
            f"found {connection.dialect.name!r}"
        )
    if "portfolio_monitoring_reports" not in inspect(connection).get_table_names():
        raise RuntimeError(
            "Cannot apply account migration 0006: monitoring reports table is "
            "missing. Restore the account schema before retrying."
        )
    existing_columns = {
        column["name"]
        for column in inspect(connection).get_columns(
            "portfolio_monitoring_reports"
        )
    }
    if "exposure_risk_assessment" not in existing_columns:
        column_type = _models.PortfolioMonitoringReport.__table__.c[
            "exposure_risk_assessment"
        ].type.compile(dialect=connection.dialect)
        connection.execute(
            text(
                "ALTER TABLE portfolio_monitoring_reports "
                f"ADD COLUMN exposure_risk_assessment {column_type}"
            )
        )


def _upgrade_portfolio_issuer_concentration(connection) -> None:
    if connection.dialect.name not in {"sqlite", "postgresql"}:
        raise RuntimeError(
            "Account migration 0007 supports SQLite and PostgreSQL only; "
            f"found {connection.dialect.name!r}"
        )
    if "portfolio_monitoring_reports" not in inspect(connection).get_table_names():
        raise RuntimeError(
            "Cannot apply account migration 0007: monitoring reports table is "
            "missing. Restore the account schema before retrying."
        )
    existing_columns = {
        column["name"]
        for column in inspect(connection).get_columns(
            "portfolio_monitoring_reports"
        )
    }
    if "issuer_concentration_assessment" not in existing_columns:
        column_type = _models.PortfolioMonitoringReport.__table__.c[
            "issuer_concentration_assessment"
        ].type.compile(dialect=connection.dialect)
        connection.execute(
            text(
                "ALTER TABLE portfolio_monitoring_reports "
                f"ADD COLUMN issuer_concentration_assessment {column_type}"
            )
        )


def _upgrade_portfolio_sector_concentration(connection) -> None:
    if connection.dialect.name not in {"sqlite", "postgresql"}:
        raise RuntimeError(
            "Account migration 0008 supports SQLite and PostgreSQL only; "
            f"found {connection.dialect.name!r}"
        )
    if "portfolio_monitoring_reports" not in inspect(connection).get_table_names():
        raise RuntimeError(
            "Cannot apply account migration 0008: monitoring reports table is "
            "missing. Restore the account schema before retrying."
        )
    existing_columns = {
        column["name"]
        for column in inspect(connection).get_columns(
            "portfolio_monitoring_reports"
        )
    }
    if "sector_concentration_assessment" not in existing_columns:
        column_type = _models.PortfolioMonitoringReport.__table__.c[
            "sector_concentration_assessment"
        ].type.compile(dialect=connection.dialect)
        connection.execute(
            text(
                "ALTER TABLE portfolio_monitoring_reports "
                f"ADD COLUMN sector_concentration_assessment {column_type}"
            )
        )


SCHEMA_MIGRATIONS = (
    ("0001_account_foundation", _upgrade_account_foundation),
    ("0002_transaction_field_integrity", _upgrade_transaction_constraints),
    ("0003_portfolio_ledger_baseline", _upgrade_portfolio_ledger_baseline),
    ("0004_transaction_idempotency", _upgrade_transaction_idempotency),
    ("0005_portfolio_monitoring_reports", _upgrade_portfolio_monitoring_reports),
    ("0006_portfolio_exposure_risk", _upgrade_portfolio_exposure_risk),
    ("0007_portfolio_issuer_concentration", _upgrade_portfolio_issuer_concentration),
    ("0008_portfolio_sector_concentration", _upgrade_portfolio_sector_concentration),
)


def initialize_database(target_engine: Engine | None = None) -> None:
    """Apply pending account schema revisions to the selected database."""
    selected_engine = target_engine or default_engine
    if selected_engine.dialect.name == "sqlite":
        database_path = selected_engine.url.database
        if database_path and database_path != ":memory:":
            Path(database_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
    with selected_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "revision VARCHAR(100) PRIMARY KEY, "
                "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        applied = {
            row[0]
            for row in connection.execute(
                text("SELECT revision FROM schema_migrations")
            )
        }
        known_revisions = {revision for revision, _upgrade in SCHEMA_MIGRATIONS}
        unknown = applied - known_revisions
        if unknown:
            raise RuntimeError(
                "account database contains unknown schema revisions: "
                + ", ".join(sorted(unknown))
            )
        for revision, upgrade in SCHEMA_MIGRATIONS:
            if revision in applied:
                continue
            upgrade(connection)
            connection.execute(
                text(
                    "INSERT INTO schema_migrations (revision) "
                    "VALUES (:revision)"
                ),
                {"revision": revision},
            )


def main() -> None:
    argparse.ArgumentParser(
        description="Initialize or upgrade the account database schema"
    ).parse_args()
    initialize_database()


if __name__ == "__main__":
    main()
