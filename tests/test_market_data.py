import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine, select, text
from sqlalchemy.orm import sessionmaker

from src.api.analytics_api import app
from src.database.migrations import initialize_database as initialize_account_database
from src.database.models import Holding, Portfolio, Transaction, User
from src.market_data.base import MarketDataBase
from src.market_data.migrations import initialize_market_database
from src.market_data.models import EODPrice
from src.market_data.providers import (
    EODPriceRecord,
    InvalidMarketDataError,
    MarketDataUnavailableError,
    MockEODProvider,
)
from src.market_data.service import (
    MarketDataConflictError,
    get_history,
    ingest_history,
    ingest_latest,
    value_portfolio,
)

_INR_NOW = datetime.now(timezone.utc)
_INDIA_TODAY = datetime.now(ZoneInfo("Asia/Kolkata")).date()


def price_record(
    *,
    exchange="NSE",
    symbol="TCS",
    market_date=None,
    open_price="100",
    high="110",
    low="95",
    close="105",
    provider="test-provider",
    source="licensed test source",
    permitted=True,
    fetched_at=_INR_NOW,
):
    return EODPriceRecord(
        exchange=exchange,
        symbol=symbol,
        market_date=market_date or _INDIA_TODAY,
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        provider=provider,
        source=source,
        fetched_at=fetched_at,
        volume=100,
        valuation_permitted=permitted,
    )


class MarketDataPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            URL.create(
                "sqlite",
                database=str(Path(self.temporary_directory.name) / "market.sqlite3"),
            )
        )
        initialize_market_database(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )

    def tearDown(self):
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def test_ingests_and_reads_valid_history(self):
        records = (
            price_record(market_date=_INDIA_TODAY - timedelta(days=2)),
            price_record(
                market_date=_INDIA_TODAY - timedelta(days=1),
                close="106",
            ),
        )
        with self.session_factory.begin() as db:
            rows = ingest_history(
                db,
                MockEODProvider(records),
                "NSE",
                "TCS",
                records[0].market_date,
                records[1].market_date,
            )
            self.assertEqual(len(rows), 2)
        with self.session_factory() as db:
            history = get_history(
                db, "NSE", "TCS", records[0].market_date, records[1].market_date
            )
        self.assertEqual([row.close for row in history], [Decimal("105.000000"), Decimal("106.000000")])
        self.assertEqual(history[0].currency, "INR")
        self.assertEqual(history[0].provider, "test-provider")
        self.assertTrue(history[0].valuation_permitted)

    def test_latest_fetch_persists_provider_metadata(self):
        record = price_record()
        with self.session_factory.begin() as db:
            saved = ingest_latest(db, MockEODProvider([record]), "NSE", "TCS")
            self.assertEqual(saved.market_date, record.market_date)
            self.assertEqual(saved.fetched_at, record.fetched_at)

    def test_duplicate_listing_date_is_rejected_without_overwrite(self):
        record = price_record()
        with self.session_factory.begin() as db:
            ingest_latest(db, MockEODProvider([record]), "NSE", "TCS")
        with self.session_factory() as db:
            with self.assertRaises(MarketDataConflictError):
                ingest_latest(db, MockEODProvider([record]), "NSE", "TCS")
        with self.session_factory() as db:
            self.assertEqual(
                len(db.scalars(select(EODPrice)).all()),
                1,
            )

    def test_invalid_prices_ohlc_future_dates_and_missing_provider_rows_rejected(self):
        invalid_records = (
            price_record(close="0"),
            price_record(open_price="120", high="110"),
            price_record(market_date=_INDIA_TODAY + timedelta(days=1)),
        )
        for record in invalid_records:
            with self.subTest(record=record):
                with self.session_factory.begin() as db:
                    with self.assertRaises(InvalidMarketDataError):
                        ingest_latest(db, MockEODProvider([record]), "NSE", "TCS")
        with self.session_factory.begin() as db:
            with self.assertRaises(MarketDataUnavailableError):
                ingest_latest(db, MockEODProvider(), "NSE", "TCS")
        with self.session_factory() as db:
            self.assertEqual(db.scalar(select(EODPrice.id)), None)

    def test_duplicate_rows_in_provider_history_are_rejected(self):
        record = price_record()
        with self.session_factory.begin() as db:
            with self.assertRaises(MarketDataConflictError):
                ingest_history(
                    db,
                    MockEODProvider([record, record]),
                    "NSE",
                    "TCS",
                    record.market_date,
                    record.market_date,
                )
        with self.session_factory() as db:
            self.assertEqual(db.scalar(select(EODPrice.id)), None)

    def test_versioned_migration_is_idempotent_and_preserves_existing_data(self):
        with self.engine.begin() as connection:
            connection.execute(
                text("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY, value TEXT)")
            )
            connection.execute(
                text("INSERT INTO legacy_marker (id, value) VALUES (1, 'preserve')")
            )
        initialize_market_database(self.engine)
        initialize_market_database(self.engine)
        with self.engine.connect() as connection:
            value = connection.execute(
                text("SELECT value FROM legacy_marker WHERE id=1")
            ).scalar_one()
            revisions = connection.execute(
                text(
                    "SELECT revision FROM market_data_schema_migrations"
                )
            ).scalars().all()
            tables = set(MarketDataBase.metadata.tables)
            db_tables = set(
                connection.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                ).scalars()
            )
        self.assertEqual(value, "preserve")
        self.assertEqual(revisions, ["0001_eod_prices"])
        self.assertIn("eod_prices", tables & db_tables)

    def test_migration_refuses_unversioned_price_table(self):
        conflict_engine = create_engine(
            URL.create(
                "sqlite",
                database=str(
                    Path(self.temporary_directory.name) / "conflicting.sqlite3"
                ),
            )
        )
        try:
            with conflict_engine.begin() as connection:
                connection.execute(
                    text("CREATE TABLE eod_prices (unrecognized_schema TEXT)")
                )
            with self.assertRaisesRegex(RuntimeError, "without a recorded"):
                initialize_market_database(conflict_engine)
            with conflict_engine.connect() as connection:
                columns = {
                    row["name"]
                    for row in connection.execute(
                        text("PRAGMA table_info(eod_prices)")
                    ).mappings()
                }
                version_table_exists = connection.execute(
                    text(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='market_data_schema_migrations'"
                    )
                ).first()
            self.assertEqual(columns, {"unrecognized_schema"})
            self.assertIsNone(version_table_exists)
        finally:
            conflict_engine.dispose()


class PortfolioValuationAPITests(unittest.TestCase):
    JWT_SECRET = "test-market-data-auth-secret-is-long-enough-123456"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.account_engine = create_engine(
            URL.create("sqlite", database=str(root / "accounts.sqlite3")),
            connect_args={"check_same_thread": False},
        )
        self.market_engine = create_engine(
            URL.create("sqlite", database=str(root / "market.sqlite3")),
            connect_args={"check_same_thread": False},
        )
        self.account_sessions = sessionmaker(
            bind=self.account_engine, autoflush=False, expire_on_commit=False
        )
        self.market_sessions = sessionmaker(
            bind=self.market_engine, autoflush=False, expire_on_commit=False
        )
        initialize_account_database(self.account_engine)
        initialize_market_database(self.market_engine)
        self.secret_patch = patch.dict(
            os.environ, {"AUTH_JWT_SECRET": self.JWT_SECRET}
        )
        self.secret_patch.start()
        self.account_session_patch = patch(
            "src.database.session.SessionLocal", self.account_sessions
        )
        self.account_engine_patch = patch(
            "src.database.migrations.default_engine", self.account_engine
        )
        self.market_session_patch = patch(
            "src.market_data.session.SessionLocal", self.market_sessions
        )
        self.market_engine_patch = patch(
            "src.market_data.migrations.default_engine", self.market_engine
        )
        for patcher in (
            self.account_session_patch,
            self.account_engine_patch,
            self.market_session_patch,
            self.market_engine_patch,
        ):
            patcher.start()
        app.state.market_data_ready = False
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        app.state.market_data_ready = False
        for patcher in (
            self.market_engine_patch,
            self.market_session_patch,
            self.account_engine_patch,
            self.account_session_patch,
        ):
            patcher.stop()
        self.secret_patch.stop()
        self.market_engine.dispose()
        self.account_engine.dispose()
        self.temporary_directory.cleanup()

    def _login(self, email):
        self.client.post(
            "/api/auth/register",
            json={"email": email, "password": "a-secure-password-123"},
        )
        response = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": "a-secure-password-123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return {
            "Authorization": f"Bearer {response.json()['access_token']}"
        }

    def _make_portfolio(self, headers, name="Valuation"):
        response = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": name, "cash_balance": "1000.00"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def _add_holding(self, headers, portfolio_id, symbol="TCS"):
        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/holdings",
            headers=headers,
            json={
                "exchange": "NSE",
                "symbol": symbol,
                "quantity": "2",
                "average_purchase_price": "100",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

    def _store_price(self, record):
        with self.market_sessions.begin() as db:
            db.add(
                EODPrice(
                    exchange=record.exchange,
                    symbol=record.symbol,
                    market_date=record.market_date,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    volume=record.volume,
                    currency=record.currency,
                    provider=record.provider,
                    source=record.source,
                    valuation_permitted=record.valuation_permitted,
                    fetched_at=record.fetched_at,
                )
            )

    def test_complete_portfolio_valuation_and_no_account_mutations(self):
        headers = self._login("value-complete@example.com")
        portfolio_id = self._make_portfolio(headers)
        self._add_holding(headers, portfolio_id)
        self._store_price(price_record())
        with self.account_sessions() as db:
            before_portfolio = db.get(Portfolio, portfolio_id)
            before_cash = before_portfolio.cash_balance
            before_holdings = db.scalars(
                select(Holding).where(Holding.portfolio_id == portfolio_id)
            ).all()
            before_transactions = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()

        response = self.client.get(
            f"/api/portfolios/{portfolio_id}/valuation", headers=headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["valuation_complete"])
        self.assertEqual(result["invested_cost_basis"], "200.00")
        self.assertEqual(result["market_value"], "210.00")
        self.assertEqual(result["unrealized_profit_loss"], "10.00")
        self.assertEqual(result["total_portfolio_value"], "1210.00")

        with self.account_sessions() as db:
            after_portfolio = db.get(Portfolio, portfolio_id)
            after_holdings = db.scalars(
                select(Holding).where(Holding.portfolio_id == portfolio_id)
            ).all()
            after_transactions = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
        self.assertEqual(after_portfolio.cash_balance, before_cash)
        self.assertEqual(
            [(item.id, item.quantity) for item in after_holdings],
            [(item.id, item.quantity) for item in before_holdings],
        )
        self.assertEqual(
            [item.id for item in after_transactions],
            [item.id for item in before_transactions],
        )

    def test_partial_valuation_missing_stale_and_unverified_prices_are_explicit(self):
        headers = self._login("value-partial@example.com")
        portfolio_id = self._make_portfolio(headers)
        self._add_holding(headers, portfolio_id, symbol="TCS")
        self._add_holding(headers, portfolio_id, symbol="INFY")
        self._add_holding(headers, portfolio_id, symbol="WIPRO")
        self._store_price(
            price_record(
                symbol="TCS",
                market_date=_INDIA_TODAY - timedelta(days=10),
            )
        )
        self._store_price(
            price_record(
                symbol="INFY",
                permitted=False,
            )
        )
        result = self.client.get(
            f"/api/portfolios/{portfolio_id}/valuation", headers=headers
        )
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertFalse(data["valuation_complete"])
        self.assertEqual(data["valuation_status"], "partial")
        self.assertIsNone(data["total_portfolio_value"])
        self.assertEqual(len(data["stale_prices"]), 1)
        self.assertEqual(len(data["unavailable_prices"]), 1)
        self.assertEqual(len(data["missing_prices"]), 1)
        self.assertEqual(data["market_value"], "0.00")

    def test_partial_total_reports_valued_subtotal_without_claiming_completeness(self):
        headers = self._login("value-subtotal@example.com")
        portfolio_id = self._make_portfolio(headers)
        self._add_holding(headers, portfolio_id, symbol="TCS")
        self._add_holding(headers, portfolio_id, symbol="INFY")
        self._store_price(price_record(symbol="TCS"))
        data = self.client.get(
            f"/api/portfolios/{portfolio_id}/valuation", headers=headers
        ).json()
        self.assertFalse(data["valuation_complete"])
        self.assertTrue(data["market_value_is_partial"])
        self.assertEqual(data["market_value"], "210.00")
        self.assertIsNone(data["total_portfolio_value"])

    def test_cross_user_valuation_is_not_found(self):
        owner_headers = self._login("value-owner@example.com")
        other_headers = self._login("value-other@example.com")
        portfolio_id = self._make_portfolio(owner_headers)
        response = self.client.get(
            f"/api/portfolios/{portfolio_id}/valuation",
            headers=other_headers,
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
