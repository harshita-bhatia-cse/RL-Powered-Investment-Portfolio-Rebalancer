import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SQLAlchemySession, sessionmaker

from src.api.analytics_api import app
from src.database.models import Holding


class PortfolioImportAPITests(unittest.TestCase):
    JWT_SECRET = "test-auth-secret-that-is-long-enough-123456"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_url = URL.create(
            "sqlite",
            database=str(Path(self.temporary_directory.name) / "accounts.sqlite3"),
        )
        self.engine = create_engine(
            self.database_url,
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
        self.session_patch = patch(
            "src.database.session.SessionLocal", self.session_factory
        )
        self.migration_patch = patch(
            "src.database.migrations.default_engine", self.engine
        )
        self.secret_patch.start()
        self.session_patch.start()
        self.migration_patch.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.migration_patch.stop()
        self.session_patch.stop()
        self.secret_patch.stop()
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def _create_user(self, email: str) -> tuple[str, dict[str, str]]:
        password = "a-secure-password-123"
        response = self.client.post(
            "/api/auth/register",
            json={"email": email, "password": password},
        )
        self.assertEqual(response.status_code, 201, response.text)
        login = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
        self.assertEqual(login.status_code, 200, login.text)
        return login.json()["user"]["id"], {
            "Authorization": f"Bearer {login.json()['access_token']}"
        }

    def _create_portfolio(
        self,
        headers: dict[str, str],
        *,
        name: str = "India holdings",
        cash_balance: str = "0.00",
    ) -> dict:
        response = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": name, "cash_balance": cash_balance},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def _upload(
        self,
        portfolio_id: str,
        headers: dict[str, str],
        content: bytes,
        *,
        filename: str = "holdings.csv",
        content_type: str = "text/csv",
    ):
        return self.client.post(
            f"/api/portfolios/{portfolio_id}/holdings/import",
            headers=headers,
            files={"file": (filename, content, content_type)},
        )

    def test_imports_all_valid_csv_holdings(self):
        _user_id, headers = self._create_user("csv-success@example.com")
        portfolio = self._create_portfolio(headers)
        content = (
            b"exchange,symbol,quantity,average_price\n"
            b"NSE,RELIANCE,5,1250.50\n"
            b"NSE,TCS,2,3400.00\n"
            b"BSE,500325,3,1500.00\n"
        )
        response = self._upload(portfolio["id"], headers, content)

        self.assertEqual(response.status_code, 201, response.text)
        result = response.json()
        self.assertEqual(result["imported_count"], 3)
        self.assertEqual(
            {(item["exchange"], item["symbol"]) for item in result["holdings"]},
            {("NSE", "RELIANCE"), ("NSE", "TCS"), ("BSE", "500325")},
        )
        self.assertEqual(
            Decimal(result["holdings"][0]["average_purchase_price"]),
            Decimal("1250.500000"),
        )
        with self.session_factory() as db:
            count = db.scalar(
                select(Holding.id).where(
                    Holding.portfolio_id == portfolio["id"]
                )
            )
            self.assertIsNotNone(count)

    def test_missing_csv_columns_are_reported_without_importing(self):
        _user_id, headers = self._create_user("csv-columns@example.com")
        portfolio = self._create_portfolio(headers)
        response = self._upload(
            portfolio["id"],
            headers,
            b"exchange,symbol,quantity\nNSE,TCS,2\n",
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("average_price", response.json()["detail"]["missing_columns"])
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=headers,
            ).json(),
            [],
        )

    def test_invalid_exchange_quantity_and_average_price_report_row_numbers(self):
        _user_id, headers = self._create_user("csv-invalid@example.com")
        portfolio = self._create_portfolio(headers)
        content = (
            b"exchange,symbol,quantity,average_price\n"
            b"NYSE,ABC,1,10\n"
            b"NSE,DEF,0,10\n"
            b"BSE,GHI,1,-1\n"
        )
        response = self._upload(portfolio["id"], headers, content)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            [error["row"] for error in response.json()["detail"]["errors"]],
            [2, 3, 4],
        )
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=headers,
            ).json(),
            [],
        )

    def test_duplicate_rows_reject_the_entire_csv(self):
        _user_id, headers = self._create_user("csv-duplicate@example.com")
        portfolio = self._create_portfolio(headers)
        response = self._upload(
            portfolio["id"],
            headers,
            (
                b"exchange,symbol,quantity,average_price\n"
                b"NSE,TCS,1,100\n"
                b"NSE,tcs,2,200\n"
            ),
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["errors"][0]["row"], 3)
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=headers,
            ).json(),
            [],
        )

    def test_existing_listing_conflict_does_not_import_other_rows(self):
        _user_id, headers = self._create_user("csv-conflict@example.com")
        portfolio = self._create_portfolio(headers)
        created = self.client.post(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
            json={
                "exchange": "NSE",
                "symbol": "TCS",
                "quantity": "1",
                "average_purchase_price": "100",
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        response = self._upload(
            portfolio["id"],
            headers,
            (
                b"exchange,symbol,quantity,average_price\n"
                b"NSE,INFY,2,200\n"
                b"NSE,TCS,3,300\n"
            ),
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["errors"][0]["row"], 3)
        holdings = self.client.get(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
        ).json()
        self.assertEqual([(item["symbol"], item["quantity"]) for item in holdings], [("TCS", "1.000000")])

    def test_one_invalid_row_causes_atomic_no_import(self):
        _user_id, headers = self._create_user("csv-atomic@example.com")
        portfolio = self._create_portfolio(headers)
        response = self._upload(
            portfolio["id"],
            headers,
            (
                b"exchange,symbol,quantity,average_price\n"
                b"NSE,RELIANCE,5,1250.50\n"
                b"NSE,TCS,not-a-number,3400.00\n"
            ),
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["errors"][0]["row"], 3)
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=headers,
            ).json(),
            [],
        )

    def test_database_integrity_failure_rolls_back_the_whole_import(self):
        _user_id, headers = self._create_user("csv-db-rollback@example.com")
        portfolio = self._create_portfolio(headers)

        class ConflictingFlushSession(SQLAlchemySession):
            target_portfolio_id = ""
            inject_conflict = [True]

            def flush(self, objects=None):
                if self.inject_conflict[0]:
                    self.inject_conflict[0] = False
                    self.add(
                        Holding(
                            portfolio_id=self.target_portfolio_id,
                            exchange="NSE",
                            symbol="TCS",
                            quantity=Decimal("1"),
                            average_purchase_price=Decimal("100"),
                        )
                    )
                super().flush(objects)

        ConflictingFlushSession.target_portfolio_id = portfolio["id"]
        conflict_factory = sessionmaker(
            bind=self.engine,
            class_=ConflictingFlushSession,
            autoflush=False,
            expire_on_commit=False,
        )
        with patch("src.database.session.SessionLocal", conflict_factory):
            response = self._upload(
                portfolio["id"],
                headers,
                (
                    b"exchange,symbol,quantity,average_price\n"
                    b"NSE,RELIANCE,5,1250.50\n"
                    b"NSE,TCS,2,3400.00\n"
                ),
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=headers,
            ).json(),
            [],
        )

    def test_import_into_another_users_portfolio_is_rejected(self):
        _owner_id, owner_headers = self._create_user("csv-owner@example.com")
        _other_id, other_headers = self._create_user("csv-other@example.com")
        portfolio = self._create_portfolio(owner_headers)
        response = self._upload(
            portfolio["id"],
            other_headers,
            b"exchange,symbol,quantity,average_price\nNSE,TCS,1,100\n",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=owner_headers,
            ).json(),
            [],
        )

    def test_csv_file_type_and_size_are_validated(self):
        _user_id, headers = self._create_user("csv-filecheck@example.com")
        portfolio = self._create_portfolio(headers)
        wrong_type = self._upload(
            portfolio["id"],
            headers,
            b"not csv",
            filename="holdings.txt",
            content_type="text/plain",
        )
        self.assertEqual(wrong_type.status_code, 415)

        oversized = self._upload(
            portfolio["id"],
            headers,
            b"x" * (1024 * 1024 + 1),
        )
        self.assertEqual(oversized.status_code, 413)

    def test_portfolio_summary_reports_cost_basis_without_market_valuation(self):
        _user_id, headers = self._create_user("summary@example.com")
        portfolio = self._create_portfolio(headers, cash_balance="2500.75")
        imported = self._upload(
            portfolio["id"],
            headers,
            (
                b"exchange,symbol,quantity,average_price\n"
                b"NSE,RELIANCE,5,1250.50\n"
                b"NSE,TCS,2,3400.00\n"
                b"BSE,500325,3,1500.00\n"
            ),
        )
        self.assertEqual(imported.status_code, 201, imported.text)
        response = self.client.get(
            f"/api/portfolios/{portfolio['id']}/summary",
            headers=headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        summary = response.json()
        self.assertEqual(summary["holdings_count"], 3)
        self.assertEqual(
            Decimal(summary["total_invested_amount"]),
            Decimal("17552.500000"),
        )
        self.assertEqual(Decimal(summary["cash_balance"]), Decimal("2500.75"))
        self.assertEqual(summary["holdings_by_exchange"], {"NSE": 2, "BSE": 1})
        self.assertIn("current market value", summary["valuation_note"])
        self.assertNotIn("market_value", summary)

    def test_existing_manual_holding_crud_and_initial_cash_remain_available(self):
        _user_id, headers = self._create_user("manual-crud@example.com")
        portfolio = self._create_portfolio(headers, cash_balance="1000.00")
        self.assertEqual(Decimal(portfolio["cash_balance"]), Decimal("1000.00"))
        response = self.client.post(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
            json={
                "exchange": "BSE",
                "symbol": "500325",
                "quantity": "2",
                "average_purchase_price": "1500",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        holding_id = response.json()["id"]
        listed = self.client.get(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["id"], holding_id)
        updated = self.client.patch(
            f"/api/portfolios/{portfolio['id']}/holdings/{holding_id}",
            headers=headers,
            json={"quantity": "4"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(Decimal(updated.json()["quantity"]), Decimal("4.000000"))
        deleted = self.client.delete(
            f"/api/portfolios/{portfolio['id']}/holdings/{holding_id}",
            headers=headers,
        )
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=headers,
            ).json(),
            [],
        )


if __name__ == "__main__":
    unittest.main()
