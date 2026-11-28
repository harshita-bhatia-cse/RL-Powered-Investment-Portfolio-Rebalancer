import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine, select
from sqlalchemy.orm import sessionmaker

from src.api.analytics_api import app
from src.database.migrations import initialize_database
from src.database.models import (
    Holding,
    Portfolio,
    PortfolioLedgerBaseline,
    PortfolioLedgerBaselineHolding,
    Transaction,
)


class TransactionReconciliationTests(unittest.TestCase):
    JWT_SECRET = "test-ledger-auth-secret-that-is-long-enough-123456"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            URL.create(
                "sqlite",
                database=str(Path(self.temporary_directory.name) / "ledger.sqlite3"),
            ),
            connect_args={"check_same_thread": False},
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.secret_patch = patch.dict(
            os.environ, {"AUTH_JWT_SECRET": self.JWT_SECRET}
        )
        self.secret_patch.start()
        self.session_patch = patch(
            "src.database.session.SessionLocal", self.session_factory
        )
        self.migration_engine_patch = patch(
            "src.database.migrations.default_engine", self.engine
        )
        self.session_patch.start()
        self.migration_engine_patch.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.migration_engine_patch.stop()
        self.session_patch.stop()
        self.secret_patch.stop()
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def _login(self, email="ledger@example.com"):
        self.client.post(
            "/api/auth/register",
            json={"email": email, "password": "a-secure-password-123"},
        )
        response = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": "a-secure-password-123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    def _portfolio(self, headers, cash="0.00"):
        response = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": "Ledger", "cash_balance": cash},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def _post(self, headers, portfolio_id, transaction_type, **fields):
        payload = {
            "transaction_type": transaction_type,
            "transaction_date": "2025-01-03",
            **fields,
        }
        return self.client.post(
            f"/api/portfolios/{portfolio_id}/transactions",
            headers=headers,
            json=payload,
        )

    def _holding(self, headers, portfolio_id, quantity="1", average="100"):
        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/holdings",
            headers=headers,
            json={
                "exchange": "NSE",
                "symbol": "TCS",
                "quantity": quantity,
                "average_purchase_price": average,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def _state(self, portfolio_id):
        with self.session_factory() as db:
            portfolio = db.get(Portfolio, portfolio_id)
            holdings = db.scalars(
                select(Holding).where(Holding.portfolio_id == portfolio_id)
            ).all()
            transactions = db.scalars(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            ).all()
            return portfolio.cash_balance, holdings, transactions

    def test_buy_updates_cash_and_holdings(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        response = self._post(
            headers,
            portfolio_id,
            "BUY",
            exchange="NSE",
            symbol="TCS",
            quantity="2",
            unit_price="100",
            fees="3.00",
        )
        self.assertEqual(response.status_code, 201, response.text)
        cash, holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("797.00"))
        self.assertEqual(holdings[0].quantity, Decimal("2.000000"))
        self.assertEqual(holdings[0].average_purchase_price, Decimal("101.500000"))
        self.assertEqual(len(transactions), 1)

    def test_buy_weighted_average_includes_fees(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "5000")
        self._holding(headers, portfolio_id, quantity="2", average="100")
        response = self._post(
            headers,
            portfolio_id,
            "BUY",
            exchange="NSE",
            symbol="TCS",
            quantity="2",
            unit_price="200",
            fees="4.00",
        )
        self.assertEqual(response.status_code, 201, response.text)
        _cash, holdings, _transactions = self._state(portfolio_id)
        self.assertEqual(holdings[0].quantity, Decimal("4.000000"))
        self.assertEqual(holdings[0].average_purchase_price, Decimal("151.000000"))

    def test_buy_rejects_insufficient_cash_without_partial_writes(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "10")
        response = self._post(
            headers,
            portfolio_id,
            "BUY",
            exchange="NSE",
            symbol="TCS",
            quantity="1",
            unit_price="11",
        )
        self.assertEqual(response.status_code, 409)
        cash, holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("10.00"))
        self.assertEqual(holdings, [])
        self.assertEqual(transactions, [])
        with self.session_factory() as db:
            self.assertIsNone(db.get(PortfolioLedgerBaseline, portfolio_id))

    def test_sell_updates_cash_and_preserves_remaining_average(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "100")
        self._holding(headers, portfolio_id, quantity="5", average="120")
        response = self._post(
            headers,
            portfolio_id,
            "SELL",
            exchange="NSE",
            symbol="TCS",
            quantity="2",
            unit_price="150",
            fees="5",
        )
        self.assertEqual(response.status_code, 201, response.text)
        cash, holdings, _transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("395.00"))
        self.assertEqual(holdings[0].quantity, Decimal("3.000000"))
        self.assertEqual(holdings[0].average_purchase_price, Decimal("120.000000"))

    def test_sell_rejects_excess_quantity_and_excessive_fees(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "100")
        self._holding(headers, portfolio_id, quantity="1", average="100")
        too_many = self._post(
            headers,
            portfolio_id,
            "SELL",
            exchange="NSE",
            symbol="TCS",
            quantity="2",
            unit_price="100",
        )
        excessive_fees = self._post(
            headers,
            portfolio_id,
            "SELL",
            exchange="NSE",
            symbol="TCS",
            quantity="1",
            unit_price="100",
            fees="100.01",
        )
        self.assertEqual(too_many.status_code, 409)
        self.assertEqual(excessive_fees.status_code, 409)
        cash, holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("100.00"))
        self.assertEqual(holdings[0].quantity, Decimal("1.000000"))
        self.assertEqual(transactions, [])

    def test_sell_removes_fully_closed_holding_but_keeps_transaction(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id, quantity="1", average="100")
        response = self._post(
            headers,
            portfolio_id,
            "SELL",
            exchange="NSE",
            symbol="TCS",
            quantity="1",
            unit_price="100",
        )
        self.assertEqual(response.status_code, 201, response.text)
        cash, holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("100.00"))
        self.assertEqual(holdings, [])
        self.assertEqual(len(transactions), 1)

    def test_cash_deposit_withdrawal_and_insufficient_withdrawal(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "100")
        ambiguous_fee = self._post(
            headers,
            portfolio_id,
            "CASH_DEPOSIT",
            cash_amount="5",
            fees="1",
        )
        deposited = self._post(
            headers, portfolio_id, "CASH_DEPOSIT", cash_amount="50"
        )
        withdrawn = self._post(
            headers, portfolio_id, "CASH_WITHDRAWAL", cash_amount="25"
        )
        rejected = self._post(
            headers, portfolio_id, "CASH_WITHDRAWAL", cash_amount="126"
        )
        self.assertEqual(ambiguous_fee.status_code, 422)
        self.assertEqual(deposited.status_code, 201, deposited.text)
        self.assertEqual(withdrawn.status_code, 201, withdrawn.text)
        self.assertEqual(rejected.status_code, 409)
        cash, _holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("125.00"))
        self.assertEqual(len(transactions), 2)

    def test_baseline_captures_manual_state_and_ignores_historical_rows(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        self._holding(headers, portfolio_id, quantity="3", average="125")
        with self.session_factory.begin() as db:
            db.add(
                Transaction(
                    portfolio_id=portfolio_id,
                    transaction_type="CASH_DEPOSIT",
                    transaction_date=date(2024, 1, 1),
                    cash_amount=Decimal("9999.00"),
                    fees=Decimal("0.00"),
                )
            )
        response = self._post(
            headers, portfolio_id, "CASH_DEPOSIT", cash_amount="50"
        )
        self.assertEqual(response.status_code, 201, response.text)
        with self.session_factory() as db:
            baseline = db.get(PortfolioLedgerBaseline, portfolio_id)
            self.assertEqual(baseline.cash_balance, Decimal("1000.00"))
            self.assertEqual(len(baseline.holdings), 1)
            self.assertEqual(
                baseline.holdings[0].quantity, Decimal("3.000000")
            )
            report = self.client.get(
                f"/api/portfolios/{portfolio_id}/reconciliation",
                headers=headers,
            )
        self.assertEqual(report.status_code, 200, report.text)
        self.assertTrue(report.json()["is_reconciled"])
        self.assertEqual(report.json()["expected_cash"], "1050.00")

    def test_reconciliation_detects_manual_holding_and_cash_changes(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        holding = self._holding(headers, portfolio_id, quantity="3", average="125")
        self.assertEqual(
            self._post(headers, portfolio_id, "CASH_DEPOSIT", cash_amount="10").status_code,
            201,
        )
        self.client.patch(
            f"/api/portfolios/{portfolio_id}/holdings/{holding['id']}",
            headers=headers,
            json={"quantity": "4", "average_purchase_price": "130"},
        )
        self.client.patch(
            f"/api/portfolios/{portfolio_id}",
            headers=headers,
            json={"cash_balance": "900"},
        )
        report = self.client.get(
            f"/api/portfolios/{portfolio_id}/reconciliation", headers=headers
        )
        self.assertEqual(report.status_code, 200, report.text)
        data = report.json()
        self.assertFalse(data["is_reconciled"])
        self.assertEqual(data["cash_difference"], "-110.00")
        self.assertEqual(data["holding_discrepancies"][0]["quantity_difference"], "1.000000")
        self.assertEqual(
            data["holding_discrepancies"][0][
                "average_purchase_price_difference"
            ],
            "5.000000",
        )

    def test_reconciliation_is_read_only(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "500")
        self._holding(headers, portfolio_id)
        self._post(headers, portfolio_id, "CASH_DEPOSIT", cash_amount="25")
        before = self._state(portfolio_id)
        self.client.get(
            f"/api/portfolios/{portfolio_id}/reconciliation", headers=headers
        )
        after = self._state(portfolio_id)
        self.assertEqual(before[0], after[0])
        self.assertEqual(
            [(h.quantity, h.average_purchase_price) for h in before[1]],
            [(h.quantity, h.average_purchase_price) for h in after[1]],
        )
        self.assertEqual(
            [t.id for t in before[2]],
            [t.id for t in after[2]],
        )

    def test_cross_user_transaction_and_reconciliation_access_returns_404(self):
        owner_headers = self._login("ledger-owner@example.com")
        other_headers = self._login("ledger-other@example.com")
        portfolio_id = self._portfolio(owner_headers, "1000")
        transaction = self._post(
            owner_headers, portfolio_id, "CASH_DEPOSIT", cash_amount="1"
        )
        self.assertEqual(transaction.status_code, 201, transaction.text)
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio_id}/reconciliation",
                headers=other_headers,
            ).status_code,
            404,
        )
        denied = self._post(
            other_headers, portfolio_id, "CASH_DEPOSIT", cash_amount="10"
        )
        self.assertEqual(denied.status_code, 404)

    def test_repeated_posts_are_each_applied_once_on_sqlite(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        for _ in range(2):
            response = self._post(
                headers, portfolio_id, "CASH_DEPOSIT", cash_amount="10"
            )
            self.assertEqual(response.status_code, 201, response.text)
        cash, _holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("1020.00"))
        self.assertEqual(len(transactions), 2)
        report = self.client.get(
            f"/api/portfolios/{portfolio_id}/reconciliation", headers=headers
        )
        self.assertTrue(report.json()["is_reconciled"])

    def test_same_idempotency_key_and_payload_returns_original_without_reposting(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        payload = {
            "idempotency_key": "client-request-123",
            "cash_amount": "10",
        }
        first = self._post(headers, portfolio_id, "CASH_DEPOSIT", **payload)
        replay = self._post(headers, portfolio_id, "CASH_DEPOSIT", **payload)
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(first.json()["id"], replay.json()["id"])
        self.assertEqual(replay.json()["idempotency_key"], "client-request-123")
        cash, _holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("1010.00"))
        self.assertEqual(len(transactions), 1)

    def test_reusing_idempotency_key_with_different_payload_returns_conflict(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        first = self._post(
            headers,
            portfolio_id,
            "CASH_DEPOSIT",
            idempotency_key="same-key",
            cash_amount="10",
        )
        conflict = self._post(
            headers,
            portfolio_id,
            "CASH_DEPOSIT",
            idempotency_key="same-key",
            cash_amount="11",
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(conflict.status_code, 409, conflict.text)
        cash, _holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("1010.00"))
        self.assertEqual(len(transactions), 1)

    def test_same_idempotency_key_is_unique_per_portfolio(self):
        headers = self._login()
        first_portfolio = self._portfolio(headers, "1000")
        second_portfolio = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": "Second", "cash_balance": "1000"},
        ).json()["id"]
        first = self._post(
            headers,
            first_portfolio,
            "CASH_DEPOSIT",
            idempotency_key="reusable-per-portfolio",
            cash_amount="10",
        )
        second = self._post(
            headers,
            second_portfolio,
            "CASH_DEPOSIT",
            idempotency_key="reusable-per-portfolio",
            cash_amount="10",
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)
        self.assertNotEqual(first.json()["id"], second.json()["id"])

    def test_concurrent_duplicate_posts_create_at_most_one_ledger_entry(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")

        def send_duplicate():
            return self._post(
                headers,
                portfolio_id,
                "CASH_DEPOSIT",
                idempotency_key="concurrent-key",
                cash_amount="10",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda _index: send_duplicate(), range(2)))
        self.assertTrue(
            all(response.status_code in {200, 201} for response in responses),
            [response.text for response in responses],
        )
        self.assertEqual(len({response.json()["id"] for response in responses}), 1)
        cash, _holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("1010.00"))
        self.assertEqual(len(transactions), 1)

    def test_legacy_requests_without_idempotency_key_remain_supported(self):
        headers = self._login()
        portfolio_id = self._portfolio(headers, "1000")
        first = self._post(
            headers, portfolio_id, "CASH_DEPOSIT", cash_amount="10"
        )
        second = self._post(
            headers, portfolio_id, "CASH_DEPOSIT", cash_amount="10"
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)
        self.assertIsNone(first.json()["idempotency_key"])
        cash, _holdings, transactions = self._state(portfolio_id)
        self.assertEqual(cash, Decimal("1020.00"))
        self.assertEqual(len(transactions), 2)


if __name__ == "__main__":
    unittest.main()
