import base64
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SQLAlchemySession, sessionmaker

from src.api.analytics_api import app
from src.auth.security import (
    ACCESS_TOKEN_SECONDS,
    InvalidAccessToken,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.database.migrations import initialize_database
from src.database.models import (
    Holding,
    Portfolio,
    PortfolioLedgerBaseline,
    Transaction,
    User,
)


class AccountAPITests(unittest.TestCase):
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
            os.environ,
            {"AUTH_JWT_SECRET": self.JWT_SECRET},
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

    def register_and_login(self, email: str) -> tuple[str, str]:
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
        data = response.json()
        return data["user"]["id"], data["access_token"]

    def auth(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _signed_token(header: dict, payload: dict) -> str:
        def encode_json(value: dict) -> str:
            encoded = json.dumps(
                value, separators=(",", ":"), sort_keys=True
            ).encode()
            return base64.urlsafe_b64encode(encoded).decode().rstrip("=")

        signing_input = f"{encode_json(header)}.{encode_json(payload)}"
        signature = hmac.new(
            AccountAPITests.JWT_SECRET.encode(),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
        return f"{signing_input}.{encoded_signature}"

    def test_registration_hashes_password_and_login_returns_bearer_token(self):
        password = "a-secure-password-123"
        response = self.client.post(
            "/api/auth/register",
            json={"email": "Person@Example.com", "password": password},
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["email"], "person@example.com")

        with self.session_factory() as db:
            user = db.scalar(select(User).where(User.email == "person@example.com"))
            self.assertIsNotNone(user)
            self.assertNotEqual(user.password_hash, password)
            self.assertTrue(verify_password(password, user.password_hash))

        login = self.client.post(
            "/api/auth/login",
            json={"email": "PERSON@example.com", "password": password},
        )
        self.assertEqual(login.status_code, 200, login.text)
        self.assertEqual(login.json()["token_type"], "bearer")
        self.assertGreater(login.json()["expires_in"], 0)
        self.assertEqual(
            self.client.get(
                "/api/auth/me",
                headers=self.auth(login.json()["access_token"]),
            ).status_code,
            200,
        )

    def test_password_hashes_use_unique_salts_and_reject_wrong_password(self):
        first = hash_password("a-secure-password-123")
        second = hash_password("a-secure-password-123")
        self.assertNotEqual(first, second)
        self.assertTrue(verify_password("a-secure-password-123", first))
        self.assertFalse(verify_password("wrong-password", first))

    def test_login_rejects_invalid_credentials_and_duplicate_registration(self):
        self.register_and_login("duplicate@example.com")
        duplicate = self.client.post(
            "/api/auth/register",
            json={
                "email": "DUPLICATE@example.com",
                "password": "another-secure-password",
            },
        )
        self.assertEqual(duplicate.status_code, 409)

        invalid = self.client.post(
            "/api/auth/login",
            json={
                "email": "duplicate@example.com",
                "password": "not-the-password",
            },
        )
        self.assertEqual(invalid.status_code, 401)

    def test_jwt_signature_tampering_is_rejected(self):
        token, _lifetime = create_access_token("user-id")
        header, payload, signature = token.split(".")
        tampered_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
        with self.assertRaises(InvalidAccessToken):
            decode_access_token(f"{header}.{payload}.{tampered_signature}")

    def test_jwt_expiry_is_enforced_at_expiration_boundary(self):
        issued_at = 1_000
        token, lifetime = create_access_token("user-id", now=issued_at)
        expires_at = issued_at + lifetime
        self.assertEqual(
            decode_access_token(token, now=expires_at - 1),
            "user-id",
        )
        with self.assertRaisesRegex(InvalidAccessToken, "expired"):
            decode_access_token(token, now=expires_at)

    def test_malformed_jwt_claims_and_unsupported_algorithms_are_rejected(self):
        valid_header = {"alg": "HS256", "typ": "JWT"}
        expiry = ACCESS_TOKEN_SECONDS + 100
        malformed_subject = self._signed_token(
            valid_header,
            {"sub": 123, "iat": 1, "exp": expiry},
        )
        with self.assertRaisesRegex(InvalidAccessToken, "subject"):
            decode_access_token(malformed_subject, now=100)

        malformed_expiry = self._signed_token(
            valid_header,
            {"sub": "user-id", "iat": 1, "exp": True},
        )
        with self.assertRaisesRegex(InvalidAccessToken, "expiry"):
            decode_access_token(malformed_expiry, now=100)

        unsupported_algorithm = self._signed_token(
            {"alg": "none", "typ": "JWT"},
            {"sub": "user-id", "iat": 1, "exp": expiry},
        )
        with self.assertRaisesRegex(InvalidAccessToken, "algorithm"):
            decode_access_token(unsupported_algorithm, now=100)

    def test_registration_and_login_responses_never_expose_password_fields(self):
        password = "a-secure-password-123"
        registration = self.client.post(
            "/api/auth/register",
            json={"email": "nohash@example.com", "password": password},
        )
        self.assertEqual(registration.status_code, 201, registration.text)
        login = self.client.post(
            "/api/auth/login",
            json={"email": "nohash@example.com", "password": password},
        )
        self.assertEqual(login.status_code, 200, login.text)
        for response in (registration, login):
            response_text = response.text.lower()
            self.assertNotIn(password.lower(), response_text)
            self.assertNotIn("password_hash", response_text)
            self.assertNotIn("scrypt$", response_text)

    def test_missing_or_short_jwt_secret_does_not_issue_a_token(self):
        password = "a-secure-password-123"
        self.client.post(
            "/api/auth/register",
            json={"email": "secretcheck@example.com", "password": password},
        )
        for secret in ("", "too-short"):
            with self.subTest(secret_length=len(secret)):
                with patch.dict(os.environ, {"AUTH_JWT_SECRET": secret}):
                    response = self.client.post(
                        "/api/auth/login",
                        json={
                            "email": "secretcheck@example.com",
                            "password": password,
                        },
                    )
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("access_token", response.json())

    def test_direct_asgi_import_uses_environment_without_creating_database(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "not-created" / "accounts.db"
            environment = os.environ.copy()
            environment["PORTFOLIO_DATABASE_URL"] = (
                f"sqlite:///{database_path.as_posix()}"
            )
            environment["AUTH_JWT_SECRET"] = self.JWT_SECRET
            code = (
                "from pathlib import Path; "
                "from src.api.analytics_api import app; "
                "from src.database.session import engine; "
                "from src.auth.security import create_access_token; "
                "assert app.title == 'Portfolio Analytics API'; "
                "assert Path(engine.url.database).name == 'accounts.db'; "
                "assert not Path(engine.url.database).parent.exists(); "
                "create_access_token('asgi-import-check'); "
                "print('direct ASGI configuration OK')"
            )
            result = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("direct ASGI configuration OK", result.stdout)
            self.assertFalse(database_path.parent.exists())

    def test_account_initialization_failure_preserves_research_routes(self):
        with patch(
            "src.api.analytics_api.initialize_database",
            side_effect=RuntimeError("simulated database initialization failure"),
        ):
            with TestClient(app) as client:
                status_response = client.get("/api/status")
                account_response = client.post(
                    "/api/auth/register",
                    json={
                        "email": "unavailable@example.com",
                        "password": "a-secure-password-123",
                    },
                )
                dashboard_response = client.get("/")

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(account_response.status_code, 503)
        self.assertEqual(
            account_response.json(),
            {"detail": "Account persistence is unavailable"},
        )
        self.assertEqual(dashboard_response.status_code, 200)

    def test_transaction_integrity_failure_rolls_back_and_returns_conflict(self):
        _user_id, token = self.register_and_login("transaction-error@example.com")
        portfolio = self.client.post(
            "/api/portfolios",
            headers=self.auth(token),
            json={"name": "Commit failure"},
        ).json()

        class CommitFailureSession(SQLAlchemySession):
            fail_next_commit = [False]
            rollback_calls = []

            def commit(self):
                if self.fail_next_commit[0]:
                    self.fail_next_commit[0] = False
                    raise IntegrityError(
                        "simulated transaction constraint failure",
                        {},
                        RuntimeError("constraint failure"),
                    )
                super().commit()

            def rollback(self):
                self.rollback_calls.append(True)
                super().rollback()

        failing_factory = sessionmaker(
            bind=self.engine,
            class_=CommitFailureSession,
            autoflush=False,
            expire_on_commit=False,
        )
        CommitFailureSession.fail_next_commit[0] = True
        with patch("src.database.session.SessionLocal", failing_factory):
            response = self.client.post(
                f"/api/portfolios/{portfolio['id']}/transactions",
                headers=self.auth(token),
                json={
                    "transaction_type": "CASH_DEPOSIT",
                    "transaction_date": "2025-01-03",
                    "cash_amount": "100",
                    "idempotency_key": "commit-retry-key",
                },
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertGreaterEqual(len(CommitFailureSession.rollback_calls), 1)
        with self.session_factory() as db:
            saved_portfolio = db.get(Portfolio, portfolio["id"])
            self.assertEqual(saved_portfolio.cash_balance, Decimal("0.00"))
            self.assertIsNone(db.get(PortfolioLedgerBaseline, portfolio["id"]))
            self.assertEqual(
                db.scalars(
                    select(Transaction).where(
                        Transaction.portfolio_id == portfolio["id"]
                    )
                ).all(),
                [],
            )
        retry = self.client.post(
            f"/api/portfolios/{portfolio['id']}/transactions",
            headers=self.auth(token),
            json={
                "transaction_type": "CASH_DEPOSIT",
                "transaction_date": "2025-01-03",
                "cash_amount": "100",
                "idempotency_key": "commit-retry-key",
            },
        )
        self.assertEqual(retry.status_code, 201, retry.text)
        with self.session_factory() as db:
            saved_portfolio = db.get(Portfolio, portfolio["id"])
            self.assertEqual(saved_portfolio.cash_balance, Decimal("100.00"))

    def test_portfolio_routes_require_authentication(self):
        response = self.client.get("/api/portfolios")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(
            self.client.get(
                "/api/portfolios",
                headers=self.auth("not-a-valid-token"),
            ).status_code,
            401,
        )

    def test_user_can_create_and_retrieve_persisted_portfolio_and_holdings(self):
        _user_id, token = self.register_and_login("owner@example.com")
        headers = self.auth(token)
        created = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": "Long-term"},
        )
        self.assertEqual(created.status_code, 201, created.text)
        portfolio = created.json()
        self.assertEqual(portfolio["base_currency"], "INR")
        self.assertEqual(Decimal(portfolio["cash_balance"]), Decimal("0.00"))

        holding_response = self.client.post(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
            json={
                "exchange": "NSE",
                "symbol": "reliance",
                "quantity": "10",
                "average_purchase_price": "1250.500000",
            },
        )
        self.assertEqual(holding_response.status_code, 201, holding_response.text)
        self.assertEqual(holding_response.json()["symbol"], "RELIANCE")

        # A separate request/session reads rows committed by the prior requests.
        listed = self.client.get("/api/portfolios", headers=headers)
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual([item["id"] for item in listed.json()], [portfolio["id"]])
        holdings = self.client.get(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
        )
        self.assertEqual(holdings.status_code, 200, holdings.text)
        self.assertEqual(len(holdings.json()), 1)

        with self.session_factory() as db:
            saved_portfolio = db.get(Portfolio, portfolio["id"])
            saved_holding = db.get(Holding, holding_response.json()["id"])
            self.assertEqual(saved_portfolio.name, "Long-term")
            self.assertEqual(saved_holding.quantity, Decimal("10.000000"))
            self.assertEqual(
                saved_holding.average_purchase_price,
                Decimal("1250.500000"),
            )

        reopened_engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False},
        )
        try:
            reopened_sessions = sessionmaker(bind=reopened_engine)
            with reopened_sessions() as db:
                self.assertIsNotNone(db.get(Portfolio, portfolio["id"]))
                self.assertIsNotNone(
                    db.get(Holding, holding_response.json()["id"])
                )
        finally:
            reopened_engine.dispose()

    def test_holding_identity_requires_exchange_and_rejects_duplicate_listing(self):
        _user_id, token = self.register_and_login("identity@example.com")
        headers = self.auth(token)
        portfolio = self.client.post(
            "/api/portfolios", headers=headers, json={"name": "Equities"}
        ).json()
        path = f"/api/portfolios/{portfolio['id']}/holdings"
        payload = {
            "exchange": "BSE",
            "symbol": "500325",
            "quantity": "2",
            "average_purchase_price": "3000",
        }
        self.assertEqual(
            self.client.post(path, headers=headers, json=payload).status_code,
            201,
        )
        duplicate = self.client.post(path, headers=headers, json=payload)
        self.assertEqual(duplicate.status_code, 409)
        other_exchange = dict(payload, exchange="NSE")
        self.assertEqual(
            self.client.post(path, headers=headers, json=other_exchange).status_code,
            201,
        )

    def test_transaction_history_is_separate_and_owned_by_portfolio(self):
        _user_id, token = self.register_and_login("history@example.com")
        headers = self.auth(token)
        portfolio = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": "History", "cash_balance": "20000.00"},
        ).json()
        response = self.client.post(
            f"/api/portfolios/{portfolio['id']}/transactions",
            headers=headers,
            json={
                "transaction_type": "BUY",
                "transaction_date": "2025-01-03",
                "exchange": "NSE",
                "symbol": "TCS",
                "quantity": "3",
                "unit_price": "4000.25",
                "fees": "2.50",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        transactions = self.client.get(
            f"/api/portfolios/{portfolio['id']}/transactions",
            headers=headers,
        )
        self.assertEqual(transactions.status_code, 200, transactions.text)
        self.assertEqual(len(transactions.json()), 1)
        self.assertEqual(transactions.json()[0]["transaction_type"], "BUY")
        holdings = self.client.get(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=headers,
        ).json()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["quantity"], "3.000000")
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}",
                headers=headers,
            ).json()["cash_balance"],
            "7996.75",
        )
        with self.session_factory() as db:
            transaction = db.get(Transaction, response.json()["id"])
            self.assertEqual(transaction.unit_price, Decimal("4000.250000"))

    def test_portfolio_and_nested_resources_are_isolated_between_users(self):
        _owner_id, owner_token = self.register_and_login("owner2@example.com")
        _other_id, other_token = self.register_and_login("other@example.com")
        owner_headers = self.auth(owner_token)
        other_headers = self.auth(other_token)
        portfolio = self.client.post(
            "/api/portfolios",
            headers=owner_headers,
            json={"name": "Private"},
        ).json()
        holding = self.client.post(
            f"/api/portfolios/{portfolio['id']}/holdings",
            headers=owner_headers,
            json={
                "exchange": "NSE",
                "symbol": "INFY",
                "quantity": "1",
                "average_purchase_price": "1500",
            },
        ).json()
        transaction = self.client.post(
            f"/api/portfolios/{portfolio['id']}/transactions",
            headers=owner_headers,
            json={
                "transaction_type": "CASH_DEPOSIT",
                "transaction_date": "2025-01-03",
                "cash_amount": "5000.00",
            },
        ).json()

        self.assertEqual(self.client.get("/api/portfolios", headers=other_headers).json(), [])
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}",
                headers=other_headers,
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/portfolios/{portfolio['id']}",
                headers=other_headers,
                json={"name": "Stolen"},
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=other_headers,
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/portfolios/{portfolio['id']}/holdings/{holding['id']}",
                headers=other_headers,
                json={"quantity": "99"},
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                f"/api/portfolios/{portfolio['id']}/holdings",
                headers=other_headers,
                json={
                    "exchange": "NSE",
                    "symbol": "TCS",
                    "quantity": "1",
                    "average_purchase_price": "3000",
                },
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(
                f"/api/portfolios/{portfolio['id']}/holdings/{holding['id']}",
                headers=other_headers,
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                f"/api/portfolios/{portfolio['id']}/transactions",
                headers=other_headers,
                json={
                    "transaction_type": "CASH_DEPOSIT",
                    "transaction_date": "2025-01-03",
                    "cash_amount": "10",
                },
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/transactions",
                headers=other_headers,
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(
                f"/api/portfolios/{portfolio['id']}",
                headers=other_headers,
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio['id']}/transactions/{transaction['id']}",
                headers=other_headers,
            ).status_code,
            404,
        )

    def test_mass_assignment_cannot_change_resource_identity_or_ownership(self):
        owner_id, owner_token = self.register_and_login("mass-owner@example.com")
        other_id, other_token = self.register_and_login("mass-other@example.com")
        owner_headers = self.auth(owner_token)
        other_headers = self.auth(other_token)

        created = self.client.post(
            "/api/portfolios",
            headers=owner_headers,
            json={
                "id": "attacker-controlled-portfolio-id",
                "user_id": other_id,
                "name": "Owned",
                "cash_balance": "10000.00",
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        portfolio_id = created.json()["id"]
        self.assertNotEqual(portfolio_id, "attacker-controlled-portfolio-id")
        patched = self.client.patch(
            f"/api/portfolios/{portfolio_id}",
            headers=owner_headers,
            json={
                "id": "replacement-id",
                "user_id": other_id,
                "name": "Renamed",
            },
        )
        self.assertEqual(patched.status_code, 200, patched.text)

        holding_response = self.client.post(
            f"/api/portfolios/{portfolio_id}/holdings",
            headers=owner_headers,
            json={
                "id": "attacker-controlled-holding-id",
                "user_id": other_id,
                "portfolio_id": "attacker-controlled-portfolio-id",
                "exchange": "NSE",
                "symbol": "INFY",
                "quantity": "2",
                "average_purchase_price": "1400",
            },
        )
        self.assertEqual(holding_response.status_code, 201, holding_response.text)
        holding_id = holding_response.json()["id"]
        self.assertNotEqual(holding_id, "attacker-controlled-holding-id")
        patched_holding = self.client.patch(
            f"/api/portfolios/{portfolio_id}/holdings/{holding_id}",
            headers=owner_headers,
            json={
                "id": "replacement-holding-id",
                "user_id": other_id,
                "portfolio_id": "attacker-controlled-portfolio-id",
                "quantity": "3",
            },
        )
        self.assertEqual(patched_holding.status_code, 200, patched_holding.text)

        transaction_response = self.client.post(
            f"/api/portfolios/{portfolio_id}/transactions",
            headers=owner_headers,
            json={
                "id": "attacker-controlled-transaction-id",
                "user_id": other_id,
                "portfolio_id": "attacker-controlled-portfolio-id",
                "transaction_type": "BUY",
                "transaction_date": "2025-01-03",
                "exchange": "NSE",
                "symbol": "INFY",
                "quantity": "1",
                "unit_price": "1400",
            },
        )
        self.assertEqual(
            transaction_response.status_code, 201, transaction_response.text
        )
        transaction_id = transaction_response.json()["id"]
        self.assertNotEqual(transaction_id, "attacker-controlled-transaction-id")

        with self.session_factory() as db:
            saved_portfolio = db.get(Portfolio, portfolio_id)
            saved_holding = db.get(Holding, holding_id)
            saved_transaction = db.get(Transaction, transaction_id)
            self.assertEqual(saved_portfolio.user_id, owner_id)
            self.assertEqual(saved_portfolio.name, "Renamed")
            self.assertEqual(saved_holding.portfolio_id, portfolio_id)
            self.assertEqual(saved_holding.quantity, Decimal("4.000000"))
            self.assertEqual(saved_transaction.portfolio_id, portfolio_id)

        self.assertEqual(
            self.client.get(
                f"/api/portfolios/{portfolio_id}",
                headers=other_headers,
            ).status_code,
            404,
        )

    def test_database_rejects_buy_sell_rows_with_null_required_fields(self):
        _user_id, token = self.register_and_login("constraint-test@example.com")
        portfolio = self.client.post(
            "/api/portfolios",
            headers=self.auth(token),
            json={"name": "Constraints"},
        ).json()
        required_fields = ("exchange", "symbol", "quantity", "unit_price")
        for transaction_type in ("BUY", "SELL"):
            for missing_field in required_fields:
                with self.subTest(
                    transaction_type=transaction_type,
                    missing_field=missing_field,
                ):
                    values = {
                        "exchange": "NSE",
                        "symbol": "TCS",
                        "quantity": Decimal("1"),
                        "unit_price": Decimal("100"),
                        "cash_amount": None,
                    }
                    values[missing_field] = None
                    with self.session_factory() as db:
                        db.add(
                            Transaction(
                                portfolio_id=portfolio["id"],
                                transaction_type=transaction_type,
                                transaction_date=date(2025, 1, 3),
                                **values,
                            )
                        )
                        with self.assertRaises(IntegrityError):
                            db.flush()
                        db.rollback()

    def test_database_rejects_invalid_cash_transaction_field_combinations(self):
        _user_id, token = self.register_and_login("cash-constraint@example.com")
        portfolio = self.client.post(
            "/api/portfolios",
            headers=self.auth(token),
            json={"name": "Cash constraints"},
        ).json()
        invalid_cases = (
            {"transaction_type": "CASH_DEPOSIT", "cash_amount": None},
            {
                "transaction_type": "CASH_WITHDRAWAL",
                "cash_amount": Decimal("10"),
                "exchange": "NSE",
            },
            {
                "transaction_type": "CASH_DEPOSIT",
                "cash_amount": Decimal("10"),
                "symbol": "TCS",
            },
            {
                "transaction_type": "CASH_WITHDRAWAL",
                "cash_amount": Decimal("10"),
                "quantity": Decimal("1"),
            },
            {
                "transaction_type": "CASH_DEPOSIT",
                "cash_amount": Decimal("10"),
                "unit_price": Decimal("100"),
            },
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.session_factory() as db:
                    db.add(
                        Transaction(
                            portfolio_id=portfolio["id"],
                            transaction_date=date(2025, 1, 3),
                            **case,
                        )
                    )
                    with self.assertRaises(IntegrityError):
                        db.flush()
                    db.rollback()

    def test_initialize_database_is_idempotent(self):
        migration_engine = create_engine(
            URL.create(
                "sqlite",
                database=str(
                    Path(self.temporary_directory.name) / "migration-test.sqlite3"
                ),
            )
        )
        try:
            initialize_database(migration_engine)
            initialize_database(migration_engine)
            with migration_engine.connect() as connection:
                revisions = connection.execute(
                    text("SELECT revision FROM schema_migrations")
                ).scalars().all()
                report_columns = connection.exec_driver_sql(
                    "PRAGMA table_info(portfolio_monitoring_reports)"
                ).mappings().all()
                report_indexes = connection.exec_driver_sql(
                    "PRAGMA index_list(portfolio_monitoring_reports)"
                ).mappings().all()
                unique_report_indexes = [
                    tuple(
                        column["name"]
                        for column in connection.exec_driver_sql(
                            f"PRAGMA index_info('{index['name']}')"
                        ).mappings()
                    )
                    for index in report_indexes
                    if index["unique"]
                ]
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
            risk_column = next(
                column
                for column in report_columns
                if column["name"] == "exposure_risk_assessment"
            )
            self.assertEqual(risk_column["notnull"], 0)
            issuer_column = next(
                column
                for column in report_columns
                if column["name"] == "issuer_concentration_assessment"
            )
            self.assertEqual(issuer_column["notnull"], 0)
            sector_column = next(
                column
                for column in report_columns
                if column["name"] == "sector_concentration_assessment"
            )
            self.assertEqual(sector_column["notnull"], 0)
            self.assertIn(
                ("portfolio_id", "session_date"),
                unique_report_indexes,
            )
        finally:
            migration_engine.dispose()
