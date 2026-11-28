import tempfile
import unittest
from pathlib import Path

from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import IntegrityError

from src.database.migrations import (
    _upgrade_portfolio_exposure_risk,
    _upgrade_portfolio_issuer_concentration,
    _upgrade_portfolio_sector_concentration,
    initialize_database,
)


_LEGACY_TRANSACTION_CHECK = (
    "("
    "transaction_type IN ('BUY', 'SELL') "
    "AND exchange IN ('NSE', 'BSE') "
    "AND symbol IS NOT NULL "
    "AND quantity > 0 "
    "AND unit_price > 0 "
    "AND cash_amount IS NULL"
    ") OR ("
    "transaction_type IN ('CASH_DEPOSIT', 'CASH_WITHDRAWAL') "
    "AND exchange IS NULL AND symbol IS NULL "
    "AND quantity IS NULL AND unit_price IS NULL "
    "AND cash_amount > 0"
    ")"
)


class AccountMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            URL.create(
                "sqlite",
                database=str(Path(self.temporary_directory.name) / "legacy.sqlite3"),
            )
        )
        self._create_revision_0001_database()

    def tearDown(self):
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def _create_revision_0001_database(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE schema_migrations ("
                    "revision VARCHAR(100) PRIMARY KEY, "
                    "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations (revision) "
                    "VALUES ('0001_account_foundation')"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE transactions ("
                    "id VARCHAR(36) PRIMARY KEY, "
                    "portfolio_id VARCHAR(36) NOT NULL DEFAULT 'legacy', "
                    "transaction_type VARCHAR(20) NOT NULL, "
                    "transaction_date DATE NOT NULL, "
                    "exchange VARCHAR(3), "
                    "symbol VARCHAR(32), "
                    "quantity NUMERIC(20, 6), "
                    "unit_price NUMERIC(20, 6), "
                    "cash_amount NUMERIC(20, 2), "
                    "fees NUMERIC(20, 2) NOT NULL DEFAULT 0, "
                    f"CONSTRAINT ck_transaction_fields_match_type "
                    f"CHECK ({_LEGACY_TRANSACTION_CHECK}))"
                )
            )
            connection.execute(
                text("CREATE TABLE portfolios (id VARCHAR(36) PRIMARY KEY)")
            )
            connection.execute(
                text("CREATE TABLE holdings (id VARCHAR(36) PRIMARY KEY)")
            )

    def test_revision_0001_migrates_without_recreating_or_losing_rows(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO transactions "
                    "(id, transaction_type, transaction_date, exchange, symbol, "
                    "quantity, unit_price, cash_amount, fees) "
                    "VALUES ('legacy-valid', 'BUY', '2025-01-03', 'NSE', "
                    "'TCS', 2, 100, NULL, 1)"
                )
            )

        initialize_database(self.engine)
        initialize_database(self.engine)

        with self.engine.connect() as connection:
            existing = connection.execute(
                text("SELECT id, quantity FROM transactions")
            ).one()
            revisions = connection.execute(
                text("SELECT revision FROM schema_migrations ORDER BY revision")
            ).scalars().all()
        self.assertEqual(existing.id, "legacy-valid")
        self.assertEqual(float(existing.quantity), 2.0)
        self.assertEqual(
            revisions,
            [
                "0001_account_foundation",
                "0002_transaction_field_integrity",
                "0003_portfolio_ledger_baseline",
                "0004_transaction_idempotency",
                "0005_portfolio_monitoring_reports",
                "0006_portfolio_exposure_risk",
                "0007_portfolio_issuer_concentration",
                "0008_portfolio_sector_concentration",
            ],
        )
        with self.engine.connect() as connection:
            columns = {
                row["name"] for row in connection.execute(text("PRAGMA table_info(transactions)")).mappings()
            }
            tables = set(
                connection.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                ).scalars()
            )
        self.assertIn("posted_at", columns)
        self.assertIn("portfolio_ledger_baselines", tables)
        self.assertIn("portfolio_ledger_baseline_holdings", tables)
        self.assertIn("idempotency_key", columns)

        with self.engine.begin() as connection:
            with self.assertRaises(IntegrityError):
                connection.execute(
                    text(
                        "INSERT INTO transactions "
                        "(id, transaction_type, transaction_date, exchange, "
                        "symbol, quantity, unit_price, cash_amount) "
                        "VALUES ('new-invalid', 'BUY', '2025-01-03', 'NSE', "
                        "'TCS', NULL, 100, NULL)"
                    )
                )
            with self.assertRaises(IntegrityError):
                connection.execute(
                    text(
                        "INSERT INTO transactions "
                        "(id, transaction_type, transaction_date, cash_amount, fees) "
                        "VALUES ('cash-fee-invalid', 'CASH_DEPOSIT', "
                        "'2025-01-03', 100, 1)"
                    )
                )

    def test_revision_0001_migration_stops_on_invalid_legacy_rows(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO transactions "
                    "(id, transaction_type, transaction_date, exchange, symbol, "
                    "quantity, unit_price, cash_amount, fees) "
                    "VALUES ('legacy-invalid', 'BUY', '2025-01-03', 'NSE', "
                    "'TCS', NULL, 100, NULL, 0)"
                )
            )

        with self.assertRaisesRegex(RuntimeError, "legacy-invalid"):
            initialize_database(self.engine)

        with self.engine.connect() as connection:
            revisions = connection.execute(
                text("SELECT revision FROM schema_migrations ORDER BY revision")
            ).scalars().all()
            existing = connection.execute(
                text("SELECT id, quantity FROM transactions")
            ).one()
        self.assertEqual(revisions, ["0001_account_foundation"])
        self.assertEqual(existing.id, "legacy-invalid")
        self.assertIsNone(existing.quantity)

    def test_revision_0006_adds_nullable_json_without_rewriting_reports(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE portfolio_monitoring_reports "
                    "(id VARCHAR(36) PRIMARY KEY, "
                    "portfolio_id VARCHAR(36) NOT NULL, "
                    "session_date DATE NOT NULL, "
                    "status VARCHAR(32) NOT NULL, "
                    "CONSTRAINT uq_portfolio_monitoring_portfolio_session "
                    "UNIQUE (portfolio_id, session_date))"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO portfolio_monitoring_reports "
                    "(id, portfolio_id, session_date, status) "
                    "VALUES ('legacy-report', 'portfolio-1', '2025-01-03', "
                    "'NO_ACTION_REQUIRED')"
                )
            )
            _upgrade_portfolio_exposure_risk(connection)
            _upgrade_portfolio_exposure_risk(connection)
            row = connection.execute(
                text(
                    "SELECT id, portfolio_id, session_date, status, "
                    "exposure_risk_assessment "
                    "FROM portfolio_monitoring_reports"
                )
            ).one()
            columns = {
                column["name"]
                for column in connection.exec_driver_sql(
                    "PRAGMA table_info(portfolio_monitoring_reports)"
                ).mappings()
            }
        self.assertEqual(
            (
                row.id,
                row.portfolio_id,
                row.session_date,
                row.status,
                row.exposure_risk_assessment,
            ),
            (
                "legacy-report",
                "portfolio-1",
                "2025-01-03",
                "NO_ACTION_REQUIRED",
                None,
            ),
        )
        self.assertIn("exposure_risk_assessment", columns)
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO portfolio_monitoring_reports "
                        "(id, portfolio_id, session_date, status) "
                        "VALUES ('duplicate-report', 'portfolio-1', "
                        "'2025-01-03', 'NO_ACTION_REQUIRED')"
                    )
                )

    def test_revision_0007_adds_nullable_issuer_json_without_rewriting_reports(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE portfolio_monitoring_reports "
                    "(id VARCHAR(36) PRIMARY KEY, "
                    "portfolio_id VARCHAR(36) NOT NULL, "
                    "session_date DATE NOT NULL, "
                    "status VARCHAR(32) NOT NULL, "
                    "exposure_risk_assessment JSON, "
                    "CONSTRAINT uq_portfolio_monitoring_portfolio_session "
                    "UNIQUE (portfolio_id, session_date))"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO portfolio_monitoring_reports "
                    "(id, portfolio_id, session_date, status, "
                    "exposure_risk_assessment) "
                    "VALUES ('legacy-report', 'portfolio-1', '2025-01-03', "
                    "'NO_ACTION_REQUIRED', '{\"legacy\": true}')"
                )
            )
            _upgrade_portfolio_issuer_concentration(connection)
            _upgrade_portfolio_issuer_concentration(connection)
            row = connection.execute(
                text(
                    "SELECT id, exposure_risk_assessment, "
                    "issuer_concentration_assessment "
                    "FROM portfolio_monitoring_reports"
                )
            ).one()
            columns = {
                column["name"]
                for column in connection.exec_driver_sql(
                    "PRAGMA table_info(portfolio_monitoring_reports)"
                ).mappings()
            }
        self.assertEqual(
            (
                row.id,
                row.exposure_risk_assessment,
                row.issuer_concentration_assessment,
            ),
            ("legacy-report", '{"legacy": true}', None),
        )
        self.assertIn("issuer_concentration_assessment", columns)

    def test_revision_0008_adds_nullable_sector_json_without_rewriting_reports(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE portfolio_monitoring_reports "
                    "(id VARCHAR(36) PRIMARY KEY, "
                    "portfolio_id VARCHAR(36) NOT NULL, "
                    "session_date DATE NOT NULL, "
                    "status VARCHAR(32) NOT NULL, "
                    "exposure_risk_assessment JSON, "
                    "issuer_concentration_assessment JSON, "
                    "CONSTRAINT uq_portfolio_monitoring_portfolio_session "
                    "UNIQUE (portfolio_id, session_date))"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO portfolio_monitoring_reports "
                    "(id, portfolio_id, session_date, status, "
                    "exposure_risk_assessment, issuer_concentration_assessment) "
                    "VALUES ('legacy-report', 'portfolio-1', '2025-01-03', "
                    "'NO_ACTION_REQUIRED', '{\"legacy\": true}', "
                    "'{\"issuer\": true}')"
                )
            )
            _upgrade_portfolio_sector_concentration(connection)
            _upgrade_portfolio_sector_concentration(connection)
            row = connection.execute(
                text(
                    "SELECT id, exposure_risk_assessment, "
                    "issuer_concentration_assessment, "
                    "sector_concentration_assessment "
                    "FROM portfolio_monitoring_reports"
                )
            ).one()
            columns = connection.exec_driver_sql(
                "PRAGMA table_info(portfolio_monitoring_reports)"
            ).mappings().all()
            indexes = connection.exec_driver_sql(
                "PRAGMA index_list(portfolio_monitoring_reports)"
            ).mappings().all()
            unique_columns = [
                tuple(
                    column["name"]
                    for column in connection.exec_driver_sql(
                        f"PRAGMA index_info('{index['name']}')"
                    ).mappings()
                )
                for index in indexes
                if index["unique"]
            ]
        self.assertEqual(
            (
                row.id,
                row.exposure_risk_assessment,
                row.issuer_concentration_assessment,
                row.sector_concentration_assessment,
            ),
            ("legacy-report", '{"legacy": true}', '{"issuer": true}', None),
        )
        sector_column = next(
            column
            for column in columns
            if column["name"] == "sector_concentration_assessment"
        )
        self.assertEqual(sector_column["notnull"], 0)
        self.assertIn(("portfolio_id", "session_date"), unique_columns)

    def test_revision_0002_adds_baseline_storage_and_post_marker_preserving_rows(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO schema_migrations (revision) "
                    "VALUES ('0002_transaction_field_integrity')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO transactions "
                    "(id, transaction_type, transaction_date, exchange, symbol, "
                    "quantity, unit_price, cash_amount, fees) "
                    "VALUES ('old-row', 'BUY', '2025-01-03', 'NSE', "
                    "'TCS', 2, 100, NULL, 1)"
                )
            )

        initialize_database(self.engine)
        initialize_database(self.engine)
        with self.engine.connect() as connection:
            existing = connection.execute(
                text(
                    "SELECT id, quantity, unit_price, posted_at, idempotency_key "
                    "FROM transactions WHERE id='old-row'"
                )
            ).one()
            revisions = connection.execute(
                text("SELECT revision FROM schema_migrations ORDER BY revision")
            ).scalars().all()
        self.assertEqual(existing.id, "old-row")
        self.assertEqual(float(existing.quantity), 2.0)
        self.assertEqual(float(existing.unit_price), 100.0)
        self.assertIsNone(existing.posted_at)
        self.assertIsNone(existing.idempotency_key)
        self.assertIn("0003_portfolio_ledger_baseline", revisions)
        self.assertIn("0004_transaction_idempotency", revisions)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO transactions "
                    "(id, transaction_type, transaction_date, cash_amount, fees, "
                    "idempotency_key) VALUES "
                    "('key-one', 'CASH_DEPOSIT', '2025-01-03', 1, 0, 'same-key')"
                )
            )
            with self.assertRaises(IntegrityError):
                connection.execute(
                    text(
                        "INSERT INTO transactions "
                        "(id, transaction_type, transaction_date, cash_amount, fees, "
                        "idempotency_key) VALUES "
                        "('key-two', 'CASH_DEPOSIT', '2025-01-03', 1, 0, 'same-key')"
                    )
                )

    def test_revision_0003_refuses_ambiguous_cash_fee_history_without_rewriting(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO transactions "
                    "(id, transaction_type, transaction_date, exchange, symbol, "
                    "quantity, unit_price, cash_amount, fees) "
                    "VALUES ('cash-fee-row', 'CASH_DEPOSIT', '2025-01-03', "
                    "NULL, NULL, NULL, NULL, 100, 1)"
                )
            )

        with self.assertRaisesRegex(RuntimeError, "cash-fee-row"):
            initialize_database(self.engine)
        with self.engine.connect() as connection:
            revisions = connection.execute(
                text("SELECT revision FROM schema_migrations ORDER BY revision")
            ).scalars().all()
            existing = connection.execute(
                text(
                    "SELECT id, cash_amount, fees "
                    "FROM transactions WHERE id='cash-fee-row'"
                )
            ).one()
            columns = {
                row["name"]
                for row in connection.execute(
                    text("PRAGMA table_info(transactions)")
                ).mappings()
            }
        self.assertEqual(revisions, ["0001_account_foundation"])
        self.assertEqual(existing.id, "cash-fee-row")
        self.assertEqual(float(existing.cash_amount), 100.0)
        self.assertEqual(float(existing.fees), 1.0)
        self.assertNotIn("posted_at", columns)

    def test_revision_0004_refuses_duplicate_existing_keys_without_rewriting(self):
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO schema_migrations (revision) VALUES "
                    "('0002_transaction_field_integrity'), "
                    "('0003_portfolio_ledger_baseline')"
                )
            )
            connection.execute(
                text("ALTER TABLE transactions ADD COLUMN idempotency_key VARCHAR(128)")
            )
            connection.execute(
                text(
                    "INSERT INTO transactions "
                    "(id, transaction_type, transaction_date, cash_amount, fees, "
                    "idempotency_key) VALUES "
                    "('duplicate-a', 'CASH_DEPOSIT', '2025-01-03', 1, 0, 'dup'), "
                    "('duplicate-b', 'CASH_DEPOSIT', '2025-01-03', 2, 0, 'dup')"
                )
            )

        with self.assertRaisesRegex(RuntimeError, "duplicate non-null"):
            initialize_database(self.engine)
        with self.engine.connect() as connection:
            revisions = connection.execute(
                text("SELECT revision FROM schema_migrations ORDER BY revision")
            ).scalars().all()
            rows = connection.execute(
                text(
                    "SELECT id, cash_amount, idempotency_key FROM transactions "
                    "ORDER BY id"
                )
            ).all()
            indexes = connection.execute(
                text("PRAGMA index_list(transactions)")
            ).all()
        self.assertNotIn("0004_transaction_idempotency", revisions)
        self.assertEqual(
            [(row.id, row.cash_amount, row.idempotency_key) for row in rows],
            [("duplicate-a", 1, "dup"), ("duplicate-b", 2, "dup")],
        )
        self.assertFalse(
            any(index.name == "uq_transactions_portfolio_idempotency_key" for index in indexes)
        )


if __name__ == "__main__":
    unittest.main()
