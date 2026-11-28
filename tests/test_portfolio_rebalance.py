import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine, select
from sqlalchemy.orm import sessionmaker

from src.api.analytics_api import app
from src.database.migrations import initialize_database
from src.database.models import Holding, Portfolio, Transaction
from src.market_data.migrations import initialize_market_database
from src.market_data.models import EODPrice

_INDIA_TODAY = datetime.now(ZoneInfo("Asia/Kolkata")).date()


class PortfolioRebalanceAPITests(unittest.TestCase):
    JWT_SECRET = "portfolio-rebalance-test-secret-long-enough-123"

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
        initialize_database(self.account_engine)
        initialize_market_database(self.market_engine)
        self.account_sessions = sessionmaker(
            bind=self.account_engine, autoflush=False, expire_on_commit=False
        )
        self.market_sessions = sessionmaker(
            bind=self.market_engine, autoflush=False, expire_on_commit=False
        )
        self.secret_patch = patch.dict(
            os.environ, {"AUTH_JWT_SECRET": self.JWT_SECRET}
        )
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
            self.secret_patch,
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
            self.secret_patch,
        ):
            patcher.stop()
        self.market_engine.dispose()
        self.account_engine.dispose()
        self.temporary_directory.cleanup()

    def _login(self, email="rebalance@example.com"):
        registered = self.client.post(
            "/api/auth/register",
            json={"email": email, "password": "a-secure-password-123"},
        )
        self.assertEqual(registered.status_code, 201, registered.text)
        response = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": "a-secure-password-123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    def _portfolio(self, headers, cash="1000.00"):
        response = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": "Preview", "cash_balance": cash},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def _holding(self, headers, portfolio_id, exchange, symbol, quantity="1"):
        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/holdings",
            headers=headers,
            json={
                "exchange": exchange,
                "symbol": symbol,
                "quantity": quantity,
                "average_purchase_price": "90.00",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

    def _price(
        self,
        exchange,
        symbol,
        *,
        close="100",
        market_date=None,
        permitted=True,
    ):
        with self.market_sessions.begin() as db:
            db.add(
                EODPrice(
                    exchange=exchange,
                    symbol=symbol,
                    market_date=market_date or _INDIA_TODAY,
                    open=Decimal(close),
                    high=Decimal(close),
                    low=Decimal(close),
                    close=Decimal(close),
                    currency="INR",
                    provider="test-provider",
                    source="temporary test data",
                    valuation_permitted=permitted,
                    fetched_at=datetime.now(timezone.utc),
                )
            )

    def _preview(self, headers, portfolio_id, body=None):
        return self.client.post(
            f"/api/portfolios/{portfolio_id}/rebalance-preview",
            headers=headers,
            json=body or {},
        )

    def test_equal_weight_baseline_and_exchange_qualified_collisions(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, "NSE", "TCS")
        self._holding(headers, portfolio_id, "BSE", "TCS")
        self._price("NSE", "TCS", close="100")
        self._price("BSE", "TCS", close="200")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "cash_reserve_pct": "0.10",
                "selected_holdings": ["NSE:TCS", "BSE:TCS"],
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["preview_available"])
        self.assertEqual(result["allocation_source"], "equal_weight_baseline")
        self.assertEqual(result["requested_target_weights"], {
            "NSE:TCS": 0.45,
            "BSE:TCS": 0.45,
        })
        self.assertEqual(result["target_cash_weight"], 0.1)
        self.assertEqual(
            {item["listing"] for item in result["market_prices"]},
            {"NSE:TCS", "BSE:TCS"},
        )

    def test_requested_and_adjusted_baseline_allocations_and_events_are_distinct(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="0.00")
        self._holding(headers, portfolio_id, "NSE", "CAPA")
        self._holding(headers, portfolio_id, "BSE", "CAPB")
        self._price("NSE", "CAPA")
        self._price("BSE", "CAPB")

        response = self._preview(
            headers,
            portfolio_id,
            {"max_weight_per_asset": "0.30"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["requested_target_weights"], {
            "NSE:CAPA": 0.5,
            "BSE:CAPB": 0.5,
        })
        self.assertEqual(result["requested_target_cash_weight"], 0.0)
        self.assertEqual(result["adjusted_target_weights"], {
            "NSE:CAPA": 0.3,
            "BSE:CAPB": 0.3,
        })
        self.assertAlmostEqual(result["adjusted_target_cash_weight"], 0.4)
        allocation_events = result["allocation_adjustment_events"]
        self.assertEqual(
            {event["listing"] for event in allocation_events},
            {"NSE:CAPA", "BSE:CAPB"},
        )
        self.assertEqual(result["final_simulated_weights"], {
            "NSE:CAPA": 0.5,
            "BSE:CAPB": 0.5,
        })
        self.assertEqual(result["final_simulated_cash_weight"], 0.0)

    def test_custom_user_weights_and_adjusted_hold_target_are_distinct(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="100.00")
        self._holding(headers, portfolio_id, "NSE", "CAPTARGET")
        self._price("NSE", "CAPTARGET")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:CAPTARGET": "0.80"},
                "target_cash_weight": "0.20",
                "max_weight_per_asset": "0.50",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["user_target_weights"], {"NSE:CAPTARGET": 0.8})
        self.assertEqual(result["user_target_cash_weight"], 0.2)
        self.assertEqual(result["requested_target_weights"], {"NSE:CAPTARGET": 0.8})
        self.assertEqual(result["requested_target_cash_weight"], 0.2)
        self.assertEqual(result["adjusted_target_weights"], {"NSE:CAPTARGET": 0.5})
        self.assertEqual(result["adjusted_target_cash_weight"], 0.5)
        self.assertTrue(
            any(
                event["code"] == "asset_cap"
                for event in result["allocation_adjustment_events"]
            )
        )
        recommendation = result["recommendations"][0]
        self.assertEqual(recommendation["action"], "HOLD")
        self.assertNotIn("asset_cap", recommendation["reason_codes"])

    def test_custom_weights_override_baseline_and_cash_reserve_is_applied(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, "NSE", "TCS")
        self._holding(headers, portfolio_id, "BSE", "TCS")
        self._price("NSE", "TCS")
        self._price("BSE", "TCS", close="200")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:TCS": "0.60"},
                "cash_reserve_pct": "0.20",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["allocation_source"], "custom")
        self.assertAlmostEqual(result["adjusted_target_weights"]["NSE:TCS"], 0.6)
        self.assertAlmostEqual(result["adjusted_target_weights"]["BSE:TCS"], 200 / 1300)
        self.assertAlmostEqual(result["target_cash_weight"], 1 - 0.6 - 200 / 1300)

    def test_excluded_holdings_are_preserved_by_baseline(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, "NSE", "SELECTED", quantity="1")
        self._holding(headers, portfolio_id, "BSE", "PRESERVE", quantity="2")
        self._price("NSE", "SELECTED", close="100")
        self._price("BSE", "PRESERVE", close="200")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "selected_holdings": ["NSE:SELECTED"],
                "cash_reserve_pct": "0.10",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertAlmostEqual(
            result["requested_target_weights"]["BSE:PRESERVE"],
            400 / 1500,
        )
        self.assertEqual(result["final_simulated_holdings"]["BSE:PRESERVE"], 2.0)
        self.assertFalse(
            any(
                trade["listing"] == "BSE:PRESERVE"
                for trade in result["proposed_trades"]
            )
        )
        recommendation = next(
            item for item in result["recommendations"]
            if item["listing"] == "BSE:PRESERVE"
        )
        self.assertEqual(recommendation["action"], "HOLD")
        self.assertEqual(recommendation["execution_status"], "NOT_APPLICABLE")

    def test_excluded_holding_value_is_preserved_with_additional_investment(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="10000.00")
        self._holding(headers, portfolio_id, "BSE", "PRESERVE", quantity="200")
        self._holding(headers, portfolio_id, "NSE", "SELECTED", quantity="100")
        self._price("BSE", "PRESERVE", close="50")
        self._price("NSE", "SELECTED", close="50")

        with self.account_sessions() as db:
            portfolio_before = db.get(Portfolio, portfolio_id)
            cash_before = portfolio_before.cash_balance
            holdings_before = db.scalars(
                select(Holding)
                .where(Holding.portfolio_id == portfolio_id)
                .order_by(Holding.exchange, Holding.symbol)
            ).all()
            transactions_before = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
            holdings_snapshot = [
                (item.id, item.exchange, item.symbol, item.quantity)
                for item in holdings_before
            ]
            transaction_ids_before = [item.id for item in transactions_before]

        response = self._preview(
            headers,
            portfolio_id,
            {
                "selected_holdings": ["NSE:SELECTED"],
                "additional_investment": "5000.00",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["preview_available"])
        self.assertEqual(result["additional_investment"], "5000.00")
        self.assertEqual(result["cash_available_for_preview"], "15000.00")
        self.assertEqual(result["initial_nav"], "30000.00")
        preserved_target = (
            Decimal(str(result["requested_target_weights"]["BSE:PRESERVE"]))
            * Decimal(result["initial_nav"])
        ).quantize(Decimal("0.01"))
        self.assertEqual(preserved_target, Decimal("10000.000000"))
        self.assertEqual(result["final_simulated_holdings"]["BSE:PRESERVE"], 200.0)
        self.assertFalse(
            any(
                trade["listing"] == "BSE:PRESERVE"
                for trade in result["proposed_trades"]
            )
        )

        with self.account_sessions() as db:
            portfolio_after = db.get(Portfolio, portfolio_id)
            holdings_after = db.scalars(
                select(Holding)
                .where(Holding.portfolio_id == portfolio_id)
                .order_by(Holding.exchange, Holding.symbol)
            ).all()
            transactions_after = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
        self.assertEqual(portfolio_after.cash_balance, cash_before)
        self.assertEqual(
            [
                (item.id, item.exchange, item.symbol, item.quantity)
                for item in holdings_after
            ],
            holdings_snapshot,
        )
        self.assertEqual([item.id for item in transactions_after], transaction_ids_before)

    def test_custom_weight_omission_preserves_other_current_holdings(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, "NSE", "TARGET", quantity="1")
        self._holding(headers, portfolio_id, "BSE", "KEEP", quantity="2")
        self._price("NSE", "TARGET")
        self._price("BSE", "KEEP", close="200")

        response = self._preview(
            headers,
            portfolio_id,
            {"target_weights": {"NSE:TARGET": "0.50"}},
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["final_simulated_holdings"]["BSE:KEEP"], 2.0)
        self.assertFalse(
            any(trade["listing"] == "BSE:KEEP" for trade in result["proposed_trades"])
        )

    def test_invalid_weight_total_is_rejected(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, "NSE", "TCS")
        self._price("NSE", "TCS")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:TCS": "0.90"},
                "target_cash_weight": "0.20",
            },
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("sum to 1.0", response.json()["detail"])

        invalid_reserve = self._preview(
            headers,
            portfolio_id,
            {"cash_reserve_pct": "1.01"},
        )
        self.assertEqual(invalid_reserve.status_code, 422, invalid_reserve.text)

        infeasible_reserve = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:TCS": "0.90"},
                "cash_reserve_pct": "0.20",
            },
        )
        self.assertEqual(infeasible_reserve.status_code, 422, infeasible_reserve.text)
        self.assertIn("insufficient cash", infeasible_reserve.json()["detail"])

    def test_incomplete_valuation_marks_preview_unavailable(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, "NSE", "STALE")
        self._holding(headers, portfolio_id, "NSE", "UNPERMITTED")
        self._holding(headers, portfolio_id, "BSE", "MISSING")
        self._price(
            "NSE",
            "STALE",
            market_date=_INDIA_TODAY - timedelta(days=10),
        )
        self._price("NSE", "UNPERMITTED", permitted=False)

        response = self._preview(headers, portfolio_id)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["preview_available"])
        self.assertEqual(result["proposed_trades"], [])
        self.assertEqual(
            {warning["category"] for warning in result["price_warnings"]},
            {"stale_prices", "unavailable_prices", "missing_prices"},
        )
        self.assertIsNone(result["initial_nav"])
        self.assertEqual(
            {item["action"] for item in result["recommendations"]},
            {"UNAVAILABLE"},
        )
        self.assertEqual(result["allocation_status"], "UNAVAILABLE")
        self.assertIsNotNone(result["allocation_unavailable_reason"])
        self.assertIsNone(result["requested_target_weights"])
        self.assertIsNone(result["requested_target_cash_weight"])
        self.assertIsNone(result["adjusted_target_weights"])
        self.assertIsNone(result["adjusted_target_cash_weight"])
        self.assertIsNone(result["final_simulated_weights"])
        self.assertIsNone(result["final_simulated_cash_weight"])
        self.assertEqual(result["allocation_adjustment_events"], [])
        self.assertTrue(
            all(item["requested_target_weight"] is None for item in result["recommendations"])
        )

    def test_incomplete_equal_weight_allocation_does_not_invent_excluded_weights(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="1000.00")
        self._holding(headers, portfolio_id, "NSE", "PICKED")
        self._holding(headers, portfolio_id, "BSE", "UNPRICED")
        self._price("NSE", "PICKED")

        with self.account_sessions() as db:
            portfolio_before = db.get(Portfolio, portfolio_id)
            cash_before = portfolio_before.cash_balance
            holdings_before = db.scalars(
                select(Holding)
                .where(Holding.portfolio_id == portfolio_id)
                .order_by(Holding.exchange, Holding.symbol)
            ).all()
            transactions_before = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
            holdings_snapshot = [
                (item.id, item.exchange, item.symbol, item.quantity)
                for item in holdings_before
            ]
            transaction_ids_before = [item.id for item in transactions_before]

        response = self._preview(
            headers,
            portfolio_id,
            {"selected_holdings": ["NSE:PICKED"]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["preview_available"])
        self.assertEqual(result["allocation_status"], "UNAVAILABLE")
        self.assertIn("complete permitted valuation", result["allocation_unavailable_reason"])
        self.assertIsNone(result["requested_target_weights"])
        self.assertIsNone(result["requested_target_cash_weight"])
        self.assertIsNone(result["adjusted_target_weights"])
        self.assertIsNone(result["adjusted_target_cash_weight"])
        self.assertIsNone(result["final_simulated_weights"])
        self.assertIsNone(result["final_simulated_cash_weight"])
        self.assertEqual(result["proposed_trades"], [])
        self.assertTrue(
            all(
                item["action"] == "UNAVAILABLE"
                and item["requested_target_weight"] is None
                and item["target_weight"] is None
                for item in result["recommendations"]
            )
        )
        with self.account_sessions() as db:
            portfolio_after = db.get(Portfolio, portfolio_id)
            holdings_after = db.scalars(
                select(Holding)
                .where(Holding.portfolio_id == portfolio_id)
                .order_by(Holding.exchange, Holding.symbol)
            ).all()
            transactions_after = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
        self.assertEqual(portfolio_after.cash_balance, cash_before)
        self.assertEqual(
            [
                (item.id, item.exchange, item.symbol, item.quantity)
                for item in holdings_after
            ],
            holdings_snapshot,
        )
        self.assertEqual(
            [item.id for item in transactions_after],
            transaction_ids_before,
        )

    def test_incomplete_custom_allocation_preserves_raw_weights_but_not_derived_weights(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="1000.00")
        self._holding(headers, portfolio_id, "NSE", "CUSTOM")
        self._holding(headers, portfolio_id, "BSE", "OMITTED")
        self._price("NSE", "CUSTOM")

        with self.account_sessions() as db:
            portfolio_before = db.get(Portfolio, portfolio_id)
            cash_before = portfolio_before.cash_balance
            holdings_before = db.scalars(
                select(Holding)
                .where(Holding.portfolio_id == portfolio_id)
                .order_by(Holding.exchange, Holding.symbol)
            ).all()
            transactions_before = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
            holdings_snapshot = [
                (item.id, item.exchange, item.symbol, item.quantity)
                for item in holdings_before
            ]
            transaction_ids_before = [item.id for item in transactions_before]

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:CUSTOM": "0.60"},
                "target_cash_weight": "0.40",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["preview_available"])
        self.assertEqual(result["allocation_status"], "UNAVAILABLE")
        self.assertEqual(result["user_target_weights"], {"NSE:CUSTOM": 0.6})
        self.assertEqual(result["user_target_cash_weight"], 0.4)
        self.assertIsNone(result["requested_target_weights"])
        self.assertIsNone(result["requested_target_cash_weight"])
        self.assertIsNone(result["adjusted_target_weights"])
        self.assertIsNone(result["adjusted_target_cash_weight"])
        self.assertIsNone(result["final_simulated_weights"])
        self.assertIsNone(result["final_simulated_cash_weight"])
        self.assertEqual(result["proposed_trades"], [])
        recommendations = {
            item["listing"]: item for item in result["recommendations"]
        }
        self.assertEqual(
            recommendations["NSE:CUSTOM"]["requested_target_weight"],
            0.6,
        )
        self.assertIsNone(
            recommendations["BSE:OMITTED"]["requested_target_weight"]
        )
        self.assertTrue(
            all(item["action"] == "UNAVAILABLE" for item in result["recommendations"])
        )
        with self.account_sessions() as db:
            portfolio_after = db.get(Portfolio, portfolio_id)
            holdings_after = db.scalars(
                select(Holding)
                .where(Holding.portfolio_id == portfolio_id)
                .order_by(Holding.exchange, Holding.symbol)
            ).all()
            transactions_after = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
        self.assertEqual(portfolio_after.cash_balance, cash_before)
        self.assertEqual(
            [
                (item.id, item.exchange, item.symbol, item.quantity)
                for item in holdings_after
            ],
            holdings_snapshot,
        )
        self.assertEqual(
            [item.id for item in transactions_after],
            transaction_ids_before,
        )

    def test_additional_investment_is_preview_only_and_does_not_mutate_account(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="1000.00")
        self._holding(headers, portfolio_id, "NSE", "TCS")
        self._price("NSE", "TCS")

        response = self._preview(
            headers, portfolio_id, {"additional_investment": "250.00"}
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["cash_available_for_preview"], "1250.00")
        self.assertEqual(result["initial_nav"], "1350.00")
        with self.account_sessions() as db:
            portfolio = db.get(Portfolio, portfolio_id)
            holdings = db.scalars(
                select(Holding).where(Holding.portfolio_id == portfolio_id)
            ).all()
            transactions = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
        self.assertEqual(portfolio.cash_balance, Decimal("1000.00"))
        self.assertEqual([(h.exchange, h.symbol, h.quantity) for h in holdings], [
            ("NSE", "TCS", Decimal("1.000000"))
        ])
        self.assertEqual(transactions, [])

    def test_empty_portfolio_with_positive_cash_is_cash_only_no_trade_preview(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="750.00")

        response = self._preview(headers, portfolio_id)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["preview_available"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["initial_nav"], "750.00")
        self.assertEqual(result["final_simulated_cash_balance"], "750.00")
        self.assertEqual(result["final_simulated_holdings"], {})
        self.assertEqual(result["proposed_trades"], [])

    def test_empty_portfolio_with_zero_nav_is_unavailable_not_an_error(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="0.00")

        response = self._preview(headers, portfolio_id)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["preview_available"])
        self.assertEqual(
            result["rejection_reasons"],
            ["Preview unavailable because total NAV is zero"],
        )

    def test_monetary_response_values_are_rounded_to_paise(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="1000.00")
        self._holding(headers, portfolio_id, "NSE", "ROUND", quantity="1")
        self._price("NSE", "ROUND", close="100.005")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:ROUND": "0.50"},
                "target_cash_weight": "0.50",
                "transaction_cost_rate": "0.003333",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["market_prices"][0]["close"], "100.01")
        for field in ("initial_nav", "final_nav", "final_simulated_cash_balance"):
            self.assertRegex(result[field], r"^-?\d+\.\d{2}$")
        for amount in result["estimated_transaction_costs"].values():
            self.assertRegex(amount, r"^-?\d+\.\d{2}$")
        for trade in result["proposed_trades"]:
            for field in (
                "reference_price",
                "execution_price",
                "reference_notional",
                "execution_notional",
                "fee",
                "slippage_cost",
            ):
                self.assertRegex(trade[field], r"^-?\d+\.\d{2}$")
        self.assertTrue(result["proposed_trades"])
        self.assertGreater(
            Decimal(result["estimated_transaction_costs"]["fees"]),
            Decimal("0.00"),
        )

    def test_multi_trade_displayed_costs_and_turnover_reconcile(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="4.00")
        self._holding(headers, portfolio_id, "NSE", "ROUNDA")
        self._holding(headers, portfolio_id, "BSE", "ROUNDB")
        self._price("NSE", "ROUNDA", close="1.00")
        self._price("BSE", "ROUNDB", close="2.00")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {
                    "NSE:ROUNDA": "0.285714285714285714",
                    "BSE:ROUNDB": "0.428571428571428571",
                },
                "target_cash_weight": "0.285714285714285715",
                "transaction_cost_rate": "0.0049",
                "slippage_bps": "60",
                "allow_fractional_shares": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        trades = result["proposed_trades"]
        self.assertEqual(len(trades), 2)
        costs = result["estimated_transaction_costs"]
        self.assertEqual(
            Decimal(costs["fees"]),
            sum((Decimal(trade["fee"]) for trade in trades), Decimal("0.00")),
        )
        self.assertEqual(
            Decimal(costs["slippage"]),
            sum(
                (Decimal(trade["slippage_cost"]) for trade in trades),
                Decimal("0.00"),
            ),
        )
        self.assertEqual(
            Decimal(costs["total"]),
            Decimal(costs["fees"]) + Decimal(costs["slippage"]),
        )
        self.assertEqual(
            Decimal(result["gross_turnover"]),
            sum(
                (Decimal(trade["reference_notional"]) for trade in trades),
                Decimal("0.00"),
            ),
        )
        self.assertEqual(costs["fees"], "0.00")
        self.assertEqual(costs["slippage"], "0.02")
        self.assertEqual(result["gross_turnover"], "2.00")
        self.assertAlmostEqual(result["turnover_ratio"], 2 / 7)
        recommendations = {
            item["listing"]: item for item in result["recommendations"]
        }
        for trade in trades:
            recommendation = recommendations[trade["listing"]]
            self.assertEqual(recommendation["estimated_fees"], trade["fee"])
            self.assertEqual(
                recommendation["estimated_slippage"],
                trade["slippage_cost"],
            )

    def test_cash_affordability_and_share_rounding_are_audited(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="100.00")
        self._holding(headers, portfolio_id, "NSE", "TCS", quantity="2")
        self._price("NSE", "TCS")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:TCS": "1.0"},
                "transaction_cost_rate": "0.10",
                "allow_fractional_shares": False,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "partially_constrained")
        self.assertEqual(result["proposed_trades"], [])
        self.assertEqual(result["final_simulated_cash_balance"], "100.00")
        self.assertTrue(
            any(
                event["code"] == "cash_affordability"
                for event in result["constraint_audit_events"]
            )
        )
        recommendation = result["recommendations"][0]
        self.assertEqual(recommendation["action"], "BUY")
        self.assertEqual(recommendation["execution_status"], "NOT_EXECUTED")
        self.assertIn("cash_affordability", recommendation["reason_codes"])

        rounded_response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:TCS": "0.5"},
                "target_cash_weight": "0.5",
                "allow_fractional_shares": False,
            },
        )
        self.assertEqual(rounded_response.status_code, 200, rounded_response.text)
        rounded = rounded_response.json()
        self.assertTrue(
            any(
                event["code"] == "share_rounding"
                for event in rounded["constraint_audit_events"]
            )
        )

    def test_global_cash_constraint_is_not_attributed_to_unaffected_hold(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="300.00")
        self._holding(headers, portfolio_id, "NSE", "CASHHOLD")
        self._holding(headers, portfolio_id, "BSE", "CASHBUY")
        self._price("NSE", "CASHHOLD")
        self._price("BSE", "CASHBUY")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {
                    "NSE:CASHHOLD": "0.20",
                    "BSE:CASHBUY": "0.80",
                },
                "target_cash_weight": "0.00",
                "transaction_cost_rate": "0.10",
                "allow_fractional_shares": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        recommendations = {
            item["listing"]: item for item in result["recommendations"]
        }
        hold = recommendations["NSE:CASHHOLD"]
        buy = recommendations["BSE:CASHBUY"]
        self.assertEqual(hold["action"], "HOLD")
        self.assertNotIn("cash_affordability", hold["reason_codes"])
        self.assertEqual(buy["action"], "BUY")
        self.assertEqual(buy["execution_status"], "PARTIALLY_EXECUTED")
        self.assertIn("cash_affordability", buy["reason_codes"])
        self.assertTrue(
            any(
                event["code"] == "cash_affordability"
                for event in result["portfolio_constraint_events"]
            )
        )
        self.assertTrue(
            any(
                event["listing"] == "BSE:CASHBUY"
                and event["code"] == "cash_affordability"
                for event in result["constraint_audit_events"]
            )
        )

    def test_buy_recommendation_and_execution_explanation(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="1000.00")
        self._holding(headers, portfolio_id, "NSE", "BUYME", quantity="1")
        self._price("NSE", "BUYME", close="100")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:BUYME": "0.50"},
                "target_cash_weight": "0.50",
                "allow_fractional_shares": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        recommendation = response.json()["recommendations"][0]
        self.assertEqual(recommendation["action"], "BUY")
        self.assertEqual(recommendation["execution_status"], "EXECUTED")
        self.assertIn("target_buy", recommendation["reason_codes"])
        self.assertIn("simulated_trade", recommendation["reason_codes"])

    def test_sell_recommendation(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="0.00")
        self._holding(headers, portfolio_id, "NSE", "SELLME", quantity="10")
        self._price("NSE", "SELLME", close="100")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:SELLME": "0.50"},
                "target_cash_weight": "0.50",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        recommendation = response.json()["recommendations"][0]
        self.assertEqual(recommendation["action"], "SELL")
        self.assertEqual(recommendation["execution_status"], "EXECUTED")
        self.assertIn("target_sell", recommendation["reason_codes"])

    def test_genuine_hold_recommendation(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="100.00")
        self._holding(headers, portfolio_id, "NSE", "HOLDME", quantity="1")
        self._price("NSE", "HOLDME", close="100")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:HOLDME": "0.50"},
                "target_cash_weight": "0.50",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        recommendation = response.json()["recommendations"][0]
        self.assertEqual(recommendation["action"], "HOLD")
        self.assertEqual(recommendation["execution_status"], "NOT_APPLICABLE")
        self.assertIn("target_already_matched", recommendation["reason_codes"])

    def test_turnover_limited_buy_is_partially_executed_and_explained(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="900.00")
        self._holding(headers, portfolio_id, "NSE", "LIMITBUY", quantity="1")
        self._price("NSE", "LIMITBUY", close="100")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:LIMITBUY": "1.0"},
                "target_cash_weight": "0.0",
                "max_turnover": "0.45",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        recommendation = result["recommendations"][0]
        self.assertEqual(recommendation["action"], "BUY")
        self.assertEqual(recommendation["execution_status"], "PARTIALLY_EXECUTED")
        self.assertIn("turnover_limit", recommendation["reason_codes"])
        self.assertEqual(result["gross_turnover"], "400.00")
        self.assertAlmostEqual(result["turnover_ratio"], 0.4)

    def test_rejected_plan_marks_all_holdings_unavailable(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, cash="0.00")
        self._holding(headers, portfolio_id, "NSE", "REJECTED", quantity="10")
        self._price("NSE", "REJECTED", close="100")

        response = self._preview(
            headers,
            portfolio_id,
            {
                "target_weights": {"NSE:REJECTED": "0.0"},
                "target_cash_weight": "1.0",
                "cash_reserve_pct": "0.50",
                "max_turnover": "0.0",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "rejected")
        recommendation = result["recommendations"][0]
        self.assertEqual(recommendation["action"], "UNAVAILABLE")
        self.assertEqual(recommendation["execution_status"], "NOT_APPLICABLE")
        self.assertIn("plan_rejected", recommendation["reason_codes"])

    def test_cross_user_preview_is_not_found(self):
        owner_headers = self._login("owner-preview@example.com")
        portfolio_id = self._portfolio(owner_headers)
        other_headers = self._login("other-preview@example.com")

        response = self._preview(other_headers, portfolio_id)

        self.assertEqual(response.status_code, 404, response.text)


if __name__ == "__main__":
    unittest.main()
