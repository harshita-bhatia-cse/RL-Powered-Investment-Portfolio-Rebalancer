import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine, select, text
from sqlalchemy.orm import sessionmaker

from src.api.analytics_api import app
from src.database.migrations import initialize_database
from src.database.models import (
    Holding,
    Portfolio,
    PortfolioLedgerBaseline,
    PortfolioLedgerBaselineHolding,
    PortfolioMonitoringReport,
    Transaction,
)
from src.market_data.migrations import initialize_market_database
from src.market_data.models import EODPrice
from src.market_data.service import (
    PortfolioNotFoundError,
    _latest_permitted,
    value_portfolio,
)
from src.services.portfolio_exposure_risk import assess_portfolio_exposure
from src.services.issuer_concentration import IssuerReferenceDataError
from src.services.portfolio_ledger import LedgerError
from src.services.portfolio_monitor import monitor_portfolio

_SESSION_DATE = datetime.now(ZoneInfo("Asia/Kolkata")).date()


class PortfolioMonitoringTests(unittest.TestCase):
    JWT_SECRET = "portfolio-monitor-test-secret-long-enough-123"

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
        initialize_database(self.account_engine)
        initialize_market_database(self.market_engine)

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
        app.state.account_database_ready = False
        app.state.market_data_ready = False
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        app.state.account_database_ready = False
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

    def _login(self, email):
        registered = self.client.post(
            "/api/auth/register",
            json={"email": email, "password": "a-secure-password-123"},
        )
        self.assertEqual(registered.status_code, 201, registered.text)
        login = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": "a-secure-password-123"},
        )
        self.assertEqual(login.status_code, 200, login.text)
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    def _portfolio(self, headers, name="Monitor", cash="1000.00"):
        response = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": name, "cash_balance": cash},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def _holding(self, headers, portfolio_id, *, exchange="NSE", symbol="TCS"):
        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/holdings",
            headers=headers,
            json={
                "exchange": exchange,
                "symbol": symbol,
                "quantity": "2",
                "average_purchase_price": "90.00",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

    def _price(
        self,
        *,
        exchange="NSE",
        symbol="TCS",
        market_date=None,
        permitted=True,
        close="100",
    ):
        with self.market_sessions.begin() as db:
            db.add(
                EODPrice(
                    exchange=exchange,
                    symbol=symbol,
                    market_date=market_date or _SESSION_DATE,
                    open=Decimal(close),
                    high=Decimal(close),
                    low=Decimal(close),
                    close=Decimal(close),
                    currency="INR",
                    provider="temporary-test-provider",
                    source="temporary test source",
                    valuation_permitted=permitted,
                    fetched_at=datetime.now(timezone.utc),
                )
            )

    def _capture_baseline(self, portfolio_id, *, cash="1000.00", holdings=()):
        with self.account_sessions.begin() as db:
            baseline = PortfolioLedgerBaseline(
                portfolio_id=portfolio_id,
                captured_at=datetime.now(timezone.utc),
                cash_balance=Decimal(cash),
            )
            baseline.holdings = [
                PortfolioLedgerBaselineHolding(
                    exchange=exchange,
                    symbol=symbol,
                    quantity=Decimal(quantity),
                    average_purchase_price=Decimal(average),
                )
                for exchange, symbol, quantity, average in holdings
            ]
            db.add(baseline)

    def _assert_account_state(
        self,
        portfolio_id,
        expected_cash,
        expected_holding_count,
        expected_transaction_count=0,
    ):
        with self.account_sessions() as db:
            portfolio = db.get(Portfolio, portfolio_id)
            self.assertEqual(portfolio.cash_balance, Decimal(expected_cash))
            self.assertEqual(
                len(
                    db.scalars(
                        select(Holding).where(
                            Holding.portfolio_id == portfolio_id
                        )
                    ).all()
                ),
                expected_holding_count,
            )
            self.assertEqual(
                len(
                    db.scalars(
                        select(Transaction).where(
                            Transaction.portfolio_id == portfolio_id
                        )
                    ).all()
                ),
                expected_transaction_count,
            )

    def _add_unposted_transaction(self, portfolio_id):
        with self.account_sessions.begin() as db:
            db.add(
                Transaction(
                    portfolio_id=portfolio_id,
                    transaction_type="CASH_DEPOSIT",
                    transaction_date=_SESSION_DATE,
                    cash_amount=Decimal("50.00"),
                    fees=Decimal("0.00"),
                    posted_at=None,
                )
            )

    def test_complete_valuation_and_clean_ledger_need_no_action(self):
        headers = self._login("clean@example.com")
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id)
        self._price()
        self._capture_baseline(
            portfolio_id,
            holdings=(("NSE", "TCS", "2", "90.00"),),
        )

        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["status"], "NO_ACTION_REQUIRED")
        self.assertTrue(report["valuation_complete"])
        self.assertEqual(Decimal(report["portfolio_nav"]), Decimal("1200.00"))
        self.assertEqual(report["evaluated_holdings_count"], 1)
        self.assertTrue(report["ledger_reconciliation"]["is_reconciled"])
        self.assertEqual(report["source_metadata"][0]["provider"], "temporary-test-provider")
        self.assertEqual(report["policy_version"], "6.1")
        risk = report["exposure_risk_assessment"]
        self.assertEqual(risk["assessment_status"], "EVALUATED")
        self.assertEqual(risk["policy_version"], "exposure-risk-v1")
        self.assertEqual(risk["listing_exposures"][0]["listing"], "NSE:TCS")
        self.assertEqual(
            [alert["code"] for alert in risk["alerts"]],
            ["EQUITY_HHI_HIGH"],
        )

    def test_exposure_uses_the_single_valuation_snapshot_without_extra_price_lookups(self):
        headers = self._login("snapshot@example.com")
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id)
        self._price()
        self._capture_baseline(
            portfolio_id,
            holdings=(("NSE", "TCS", "2", "90.00"),),
        )

        with (
            patch(
                "src.services.portfolio_monitor.value_portfolio",
                wraps=value_portfolio,
            ) as valuation_call,
            patch(
                "src.services.portfolio_monitor.assess_portfolio_exposure",
                wraps=assess_portfolio_exposure,
            ) as exposure_call,
            patch(
                "src.market_data.service._latest_permitted",
                wraps=_latest_permitted,
            ) as price_lookup,
        ):
            response = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(valuation_call.call_count, 1)
        self.assertEqual(price_lookup.call_count, 1)
        self.assertEqual(exposure_call.call_count, 1)
        exposure_inputs = exposure_call.call_args.kwargs
        self.assertEqual(
            exposure_inputs["portfolio_nav"],
            Decimal(report["portfolio_nav"]),
        )
        self.assertEqual(
            exposure_inputs["cash_balance"],
            Decimal(report["exposure_risk_assessment"]["cash_balance"]),
        )
        self.assertEqual(exposure_inputs["holdings"], report["evaluated_holdings"])

    def test_ledger_discrepancy_requires_action(self):
        headers = self._login("discrepancy@example.com")
        portfolio_id = self._portfolio(headers)
        self._capture_baseline(portfolio_id)
        with self.account_sessions.begin() as db:
            db.get(Portfolio, portfolio_id).cash_balance = Decimal("900.00")

        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "ACTION_REQUIRED")
        self.assertIn("ledger_discrepancy", response.json()["reason_codes"])
        self.assertIn(
            "ledger_discrepancy",
            [
                warning["code"]
                for warning in response.json()["data_quality_warnings"]
            ],
        )
        self.assertEqual(
            response.json()["exposure_risk_assessment"]["assessment_status"],
            "EVALUATED",
        )

    def test_missing_stale_and_unpermitted_prices_are_unavailable(self):
        for name, price_options in (
            ("missing", None),
            (
                "stale",
                {
                    "market_date": _SESSION_DATE - timedelta(days=5),
                },
            ),
            ("unpermitted", {"permitted": False}),
        ):
            with self.subTest(name=name):
                headers = self._login(f"{name}@example.com")
                portfolio_id = self._portfolio(headers, name=name)
                self._holding(headers, portfolio_id)
                self._capture_baseline(
                    portfolio_id,
                    holdings=(("NSE", "TCS", "2", "90.00"),),
                )
                if price_options is not None:
                    self._price(**price_options)

                response = self.client.post(
                    f"/api/portfolios/{portfolio_id}/monitor",
                    headers=headers,
                    json={"session_date": _SESSION_DATE.isoformat()},
                )

                self.assertEqual(response.status_code, 200, response.text)
                report = response.json()
                self.assertEqual(report["status"], "DATA_UNAVAILABLE")
                self.assertFalse(report["valuation_complete"])
                self.assertIsNone(report["portfolio_nav"])
                self.assertIn("required_price_unavailable", report["reason_codes"])
                self.assertEqual(
                    report["exposure_risk_assessment"]["assessment_status"],
                    "NOT_EVALUATED",
                )
                self.assertEqual(report["exposure_risk_assessment"]["alerts"], [])
                self.assertIsNone(
                    report["exposure_risk_assessment"]["portfolio_nav"]
                )
                self.assertEqual(
                    report["exposure_risk_assessment"]["unavailable_listings"][0][
                        "listing"
                    ],
                    "NSE:TCS",
                )
                self._assert_account_state(portfolio_id, "1000.00", 1)

    def test_cash_only_and_zero_nav_portfolios_are_valid_no_action_evaluations(self):
        headers = self._login("cash-only@example.com")
        cash_portfolio = self._portfolio(headers, name="Cash", cash="250.00")
        zero_portfolio = self._portfolio(headers, name="Zero", cash="0.00")
        self._capture_baseline(cash_portfolio, cash="250.00")
        self._capture_baseline(zero_portfolio, cash="0.00")

        cash_result = self.client.post(
            f"/api/portfolios/{cash_portfolio}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        zero_result = self.client.post(
            f"/api/portfolios/{zero_portfolio}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )

        self.assertEqual(cash_result.status_code, 200, cash_result.text)
        self.assertEqual(zero_result.status_code, 200, zero_result.text)
        self.assertEqual(cash_result.json()["status"], "NO_ACTION_REQUIRED")
        self.assertEqual(Decimal(cash_result.json()["portfolio_nav"]), Decimal("250.00"))
        self.assertEqual(zero_result.json()["status"], "NO_ACTION_REQUIRED")
        self.assertEqual(Decimal(zero_result.json()["portfolio_nav"]), Decimal("0.00"))
        self.assertEqual(cash_result.json()["evaluated_holdings_count"], 0)
        self.assertEqual(zero_result.json()["evaluated_holdings_count"], 0)
        cash_risk = cash_result.json()["exposure_risk_assessment"]
        zero_risk = zero_result.json()["exposure_risk_assessment"]
        self.assertEqual(cash_risk["assessment_status"], "EVALUATED")
        self.assertEqual(cash_risk["cash_weight_pct"], "100")
        self.assertIsNone(cash_risk["equity_hhi"])
        self.assertEqual(cash_risk["alerts"], [])
        self.assertEqual(zero_risk["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(zero_risk["reason_codes"], ["portfolio_nav_not_positive"])
        self.assertIsNone(zero_risk["portfolio_nav"])

    def test_duplicate_session_request_returns_original_report_and_does_not_mutate_account(self):
        headers = self._login("idempotent@example.com")
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id)
        self._price()
        self._capture_baseline(
            portfolio_id,
            holdings=(("NSE", "TCS", "2", "90.00"),),
        )
        self._add_unposted_transaction(portfolio_id)
        with self.account_sessions() as db:
            transaction_before = db.scalar(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            )
            before_transaction = (
                transaction_before.id,
                transaction_before.transaction_type,
                transaction_before.transaction_date,
                transaction_before.cash_amount,
                transaction_before.fees,
                transaction_before.posted_at,
            )

        first = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        with (
            patch(
                "src.services.portfolio_monitor.value_portfolio",
                side_effect=AssertionError("replay must not revalue"),
            ) as valuation_call,
            patch(
                "src.services.portfolio_monitor.assess_portfolio_exposure",
                side_effect=AssertionError("replay must not reassess exposure"),
            ) as exposure_call,
        ):
            second = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        valuation_call.assert_not_called()
        exposure_call.assert_not_called()
        self.assertEqual(first.json(), second.json())
        self.assertEqual(
            first.json()["exposure_risk_assessment"],
            second.json()["exposure_risk_assessment"],
        )
        with self.account_sessions() as db:
            reports = db.scalars(
                select(PortfolioMonitoringReport).where(
                    PortfolioMonitoringReport.portfolio_id == portfolio_id
                )
            ).all()
            transaction_after = db.scalar(
                select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            )
            after_transaction = (
                transaction_after.id,
                transaction_after.transaction_type,
                transaction_after.transaction_date,
                transaction_after.cash_amount,
                transaction_after.fees,
                transaction_after.posted_at,
            )
        self.assertEqual(len(reports), 1)
        self.assertEqual(before_transaction, after_transaction)
        self._assert_account_state(portfolio_id, "1000.00", 1, 1)

    def test_issuer_assessment_is_persisted_and_replayed_after_reference_change(self):
        headers = self._login("issuer-replay@example.com")
        portfolio_id = self._portfolio(headers, name="Issuer replay", cash="0.00")
        self._holding(headers, portfolio_id, exchange="NSE", symbol="TCS")
        self._holding(headers, portfolio_id, exchange="BSE", symbol="TCS")
        self._price(exchange="NSE", symbol="TCS", close="100")
        self._price(exchange="BSE", symbol="TCS", close="100")
        self._capture_baseline(
            portfolio_id,
            cash="0.00",
            holdings=(
                ("NSE", "TCS", "2", "90.00"),
                ("BSE", "TCS", "2", "90.00"),
            ),
        )
        reference_v1 = {
            "schema_version": 1,
            "reference_data_version": "test-issuer-reference-v1",
            "records": [
                {
                    "exchange": exchange,
                    "symbol": "TCS",
                    "instrument_identifier": f"TEST-{exchange}-INSTRUMENT",
                    "instrument_identifier_type": "OTHER_VERIFIED",
                    "issuer_id": "TEST-ISSUER-1",
                    "issuer_id_type": "OTHER_VERIFIED",
                    "effective_from": "2000-01-01",
                    "effective_to": None,
                    "source": "test fixture only",
                    "source_reference": "test fixture document 1",
                    "verified_on": _SESSION_DATE.isoformat(),
                    "reference_data_version": "test-issuer-reference-v1",
                }
                for exchange in ("NSE", "BSE")
            ],
        }
        reference_v2 = {
            **reference_v1,
            "reference_data_version": "test-issuer-reference-v2",
            "records": [
                {
                    **record,
                    "issuer_id": "TEST-ISSUER-2",
                    "reference_data_version": "test-issuer-reference-v2",
                }
                for record in reference_v1["records"]
            ],
        }

        with patch(
            "src.services.portfolio_monitor.load_issuer_reference_data",
            return_value=reference_v1,
        ) as reference_loader:
            first = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )
        self.assertEqual(first.status_code, 200, first.text)
        first_assessment = first.json()["issuer_concentration_assessment"]
        self.assertEqual(first_assessment["assessment_status"], "EVALUATED")
        self.assertEqual(
            first_assessment["reference_data_version"],
            "test-issuer-reference-v1",
        )
        self.assertEqual(len(first_assessment["issuer_exposures"]), 1)
        self.assertEqual(
            first_assessment["issuer_exposures"][0]["market_value"],
            "400.00",
        )
        reference_loader.assert_called_once()

        with patch(
            "src.services.portfolio_monitor.load_issuer_reference_data",
            return_value=reference_v2,
        ) as changed_reference_loader:
            second = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            second.json()["issuer_concentration_assessment"],
            first_assessment,
        )
        changed_reference_loader.assert_not_called()
        with self.account_sessions() as db:
            persisted = db.scalar(
                select(PortfolioMonitoringReport).where(
                    PortfolioMonitoringReport.portfolio_id == portfolio_id
                )
            )
            persisted_assessment = persisted.issuer_concentration_assessment
            persisted_time = datetime.fromisoformat(
                persisted_assessment["evaluated_at"]
            )
            response_time = datetime.fromisoformat(first_assessment["evaluated_at"])
            self.assertEqual(
                {
                    key: value
                    for key, value in persisted_assessment.items()
                    if key != "evaluated_at"
                },
                {
                    key: value
                    for key, value in first_assessment.items()
                    if key != "evaluated_at"
                },
            )
            self.assertEqual(persisted_time, response_time)

    def test_sector_assessment_is_persisted_and_replayed_without_reloading_reference(self):
        headers = self._login("sector-replay@example.com")
        portfolio_id = self._portfolio(headers, name="Sector replay", cash="0.00")
        self._holding(headers, portfolio_id, exchange="NSE", symbol="TCS")
        self._holding(headers, portfolio_id, exchange="BSE", symbol="TCS")
        self._price(exchange="NSE", symbol="TCS", close="100")
        self._price(exchange="BSE", symbol="TCS", close="100")
        self._capture_baseline(
            portfolio_id,
            cash="0.00",
            holdings=(
                ("NSE", "TCS", "2", "90.00"),
                ("BSE", "TCS", "2", "90.00"),
            ),
        )
        issuer_reference = {
            "schema_version": 1,
            "reference_data_version": "test-issuer-reference-v1",
            "records": [
                {
                    "exchange": exchange,
                    "symbol": "TCS",
                    "instrument_identifier": f"TEST-{exchange}-INSTRUMENT",
                    "instrument_identifier_type": "OTHER_VERIFIED",
                    "issuer_id": "TEST-ISSUER-1",
                    "issuer_id_type": "OTHER_VERIFIED",
                    "effective_from": "2000-01-01",
                    "effective_to": None,
                    "source": "test fixture only",
                    "source_reference": "test fixture document 1",
                    "verified_on": _SESSION_DATE.isoformat(),
                    "reference_data_version": "test-issuer-reference-v1",
                }
                for exchange in ("NSE", "BSE")
            ],
        }
        reference_v1 = {
            "schema_version": 1,
            "reference_data_version": "test-sector-reference-v1",
            "taxonomy": {
                "name": "project-india-equity-sectors",
                "version": "1.0",
                "sectors": [
                    {"code": code, "name": name}
                    for code, name in sorted(
                        {
                            "BANKING_FINANCE": "Banking and Financial Services",
                            "CONSUMER": "Consumer Products and Services",
                            "ENERGY_UTILITIES": "Energy and Utilities",
                            "HEALTHCARE": "Healthcare and Life Sciences",
                            "INDUSTRIALS": "Industrial Products and Services",
                            "MATERIALS": "Materials and Natural Resources",
                            "TECHNOLOGY": "Technology and Digital Services",
                            "TELECOM_MEDIA": "Telecommunications and Media",
                            "TRANSPORT_LOGISTICS": "Transportation and Logistics",
                            "DIVERSIFIED": "Diversified",
                        }.items()
                    )
                ],
            },
            "records": [
                {
                    "issuer_id_type": "OTHER_VERIFIED",
                    "issuer_id": "TEST-ISSUER-1",
                    "sector_code": "TECHNOLOGY",
                    "sector_name": "Technology and Digital Services",
                    "effective_from": "2000-01-01",
                    "effective_to": None,
                    "source": "test fixture only",
                    "source_reference": (
                        "Annual report, page 17 https://example.org/report.pdf"
                    ),
                    "source_published_on": "2020-01-01",
                    "verified_on": "2025-01-01",
                    "reference_data_version": "test-sector-reference-v1",
                }
            ],
        }
        reference_v2 = {
            **reference_v1,
            "reference_data_version": "test-sector-reference-v2",
            "records": [
                {
                    **reference_v1["records"][0],
                    "sector_code": "CONSUMER",
                    "sector_name": "Consumer Products and Services",
                    "reference_data_version": "test-sector-reference-v2",
                }
            ],
        }
        with patch(
            "src.services.portfolio_monitor.load_sector_reference_data",
            return_value=reference_v1,
        ) as reference_loader:
            with patch(
                "src.services.portfolio_monitor.load_issuer_reference_data",
                return_value=issuer_reference,
            ):
                first = self.client.post(
                    f"/api/portfolios/{portfolio_id}/monitor",
                    headers=headers,
                    json={"session_date": _SESSION_DATE.isoformat()},
                )
        self.assertEqual(first.status_code, 200, first.text)
        first_assessment = first.json()["sector_concentration_assessment"]
        self.assertEqual(first_assessment["assessment_status"], "EVALUATED")
        self.assertEqual(
            first_assessment["reference_data_version"],
            "test-sector-reference-v1",
        )
        self.assertEqual(len(first_assessment["sector_exposures"]), 1)
        self.assertEqual(
            first_assessment["sector_exposures"][0]["market_value"],
            "400.00",
        )
        self.assertEqual(len(first_assessment["issuer_contributions"]), 1)
        self.assertRegex(first_assessment["reference_snapshot_sha256"], r"^[a-f0-9]{64}$")
        reference_loader.assert_called_once()

        with (
            patch(
                "src.services.portfolio_monitor.load_sector_reference_data",
                return_value=reference_v2,
            ) as changed_reference_loader,
            patch(
                "src.services.portfolio_monitor.assess_portfolio_sector_concentration",
                side_effect=AssertionError("replay must not reassess sectors"),
            ) as sector_assessor,
            patch(
                "src.services.portfolio_monitor.load_issuer_reference_data",
                side_effect=AssertionError("replay must not reload issuer data"),
            ) as issuer_reference_loader,
        ):
            second = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(
            second.json()["sector_concentration_assessment"],
            first_assessment,
        )
        changed_reference_loader.assert_not_called()
        issuer_reference_loader.assert_not_called()
        sector_assessor.assert_not_called()
        history = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        detail = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{first.json()['id']}",
            headers=headers,
        )
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(
            history.json()[0]["sector_concentration_assessment"],
            first_assessment,
        )
        self.assertEqual(
            detail.json()["sector_concentration_assessment"],
            first_assessment,
        )
        with self.account_sessions() as db:
            persisted = db.scalar(
                select(PortfolioMonitoringReport).where(
                    PortfolioMonitoringReport.portfolio_id == portfolio_id
                )
            )
            self.assertEqual(
                persisted.sector_concentration_assessment["reference_data_version"],
                "test-sector-reference-v1",
            )

    def test_malformed_sector_assessment_is_null_on_replay_history_and_detail(self):
        headers = self._login("malformed-sector@example.com")
        portfolio_id = self._portfolio(
            headers,
            name="Malformed sector",
            cash="250.00",
        )
        self._capture_baseline(portfolio_id, cash="250.00")
        created = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        self.assertEqual(created.status_code, 200, created.text)
        report_id = created.json()["id"]
        malformed = {
            **created.json()["sector_concentration_assessment"],
            "portfolio_nav": "0.00",
            "total_equity_market_value": "1.00",
            "issuer_contributions": [
                {
                    "issuer_id_type": "OTHER_VERIFIED",
                    "issuer_id": "TEST-ISSUER",
                    "market_value": "1.00",
                    "sector_code": "TECHNOLOGY",
                    "sector_name": "Technology and Digital Services",
                    "mapping": {},
                }
            ],
        }
        with self.account_sessions.begin() as db:
            report = db.get(PortfolioMonitoringReport, report_id)
            report.sector_concentration_assessment = malformed

        replay = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        history = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        detail = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=headers,
        )
        for response in (replay, history, detail):
            self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(replay.json()["sector_concentration_assessment"])
        self.assertIsNone(history.json()[0]["sector_concentration_assessment"])
        self.assertIsNone(detail.json()["sector_concentration_assessment"])
        with self.account_sessions() as db:
            report = db.get(PortfolioMonitoringReport, report_id)
            self.assertEqual(report.sector_concentration_assessment, malformed)

    def test_issuer_assessment_is_not_evaluated_when_ledger_is_unavailable(self):
        headers = self._login("issuer-ledger-unavailable@example.com")
        portfolio_id = self._portfolio(
            headers,
            name="Issuer ledger unavailable",
            cash="250.00",
        )
        self._capture_baseline(portfolio_id, cash="250.00")

        with patch(
            "src.services.portfolio_monitor.reconcile_portfolio",
            side_effect=LedgerError("ledger reconciliation unavailable"),
        ):
            response = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["status"], "ACTION_REQUIRED")
        assessment = report["issuer_concentration_assessment"]
        self.assertEqual(assessment["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(
            assessment["reason_codes"],
            ["ledger_reconciliation_unavailable"],
        )
        self.assertIsNone(assessment["portfolio_nav"])
        self.assertEqual(assessment["issuer_exposures"], [])
        self.assertEqual(assessment["alerts"], [])

    def test_invalid_reference_file_does_not_fail_monitoring_evaluation(self):
        headers = self._login("issuer-invalid-reference@example.com")
        portfolio_id = self._portfolio(
            headers,
            name="Issuer invalid reference",
            cash="250.00",
        )
        self._capture_baseline(portfolio_id, cash="250.00")

        with patch(
            "src.services.portfolio_monitor.load_issuer_reference_data",
            side_effect=IssuerReferenceDataError(
                "local issuer reference data is unavailable"
            ),
        ):
            response = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["status"], "NO_ACTION_REQUIRED")
        assessment = report["issuer_concentration_assessment"]
        self.assertEqual(assessment["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(assessment["reason_codes"], ["issuer_reference_invalid"])
        self.assertEqual(assessment["issuer_exposures"], [])
        self.assertEqual(assessment["alerts"], [])

    def test_issuer_assessment_is_not_evaluated_on_ledger_discrepancy(self):
        headers = self._login("issuer-ledger-discrepancy@example.com")
        portfolio_id = self._portfolio(
            headers,
            name="Issuer ledger discrepancy",
            cash="250.00",
        )
        self._capture_baseline(portfolio_id, cash="250.00")
        discrepant_reconciliation = SimpleNamespace(
            is_reconciled=False,
            baseline_timestamp=datetime.now(timezone.utc),
            expected_cash=Decimal("250.00"),
            actual_cash=Decimal("249.00"),
            cash_difference=Decimal("-1.00"),
            expected_holdings=[],
            actual_holdings=[],
            holding_discrepancies=[],
        )

        with patch(
            "src.services.portfolio_monitor.reconcile_portfolio",
            return_value=discrepant_reconciliation,
        ):
            response = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["status"], "ACTION_REQUIRED")
        assessment = report["issuer_concentration_assessment"]
        self.assertEqual(assessment["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(assessment["reason_codes"], ["ledger_discrepancy"])
        self.assertIsNone(assessment["portfolio_nav"])
        self.assertEqual(assessment["issuer_exposures"], [])
        self.assertEqual(assessment["alerts"], [])

    def test_malformed_legacy_issuer_assessment_is_null_without_rewriting(self):
        headers = self._login("malformed-issuer@example.com")
        portfolio_id = self._portfolio(
            headers,
            name="Malformed issuer",
            cash="250.00",
        )
        self._capture_baseline(portfolio_id, cash="250.00")
        created = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        self.assertEqual(created.status_code, 200, created.text)
        report_id = created.json()["id"]
        malformed = {
            **created.json()["issuer_concentration_assessment"],
            "issuer_exposures": [{"issuer_id": []}],
        }
        with self.account_sessions.begin() as db:
            report = db.get(PortfolioMonitoringReport, report_id)
            report.issuer_concentration_assessment = malformed

        replay = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        history = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        detail = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=headers,
        )

        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertIsNone(replay.json()["issuer_concentration_assessment"])
        self.assertIsNone(history.json()[0]["issuer_concentration_assessment"])
        self.assertIsNone(detail.json()["issuer_concentration_assessment"])
        with self.account_sessions() as db:
            report = db.get(PortfolioMonitoringReport, report_id)
            self.assertEqual(report.issuer_concentration_assessment, malformed)

    def test_zero_nav_legacy_issuer_entry_is_null_without_rewriting(self):
        headers = self._login("zero-nav-issuer@example.com")
        portfolio_id = self._portfolio(
            headers,
            name="Zero NAV issuer assessment",
            cash="250.00",
        )
        self._capture_baseline(portfolio_id, cash="250.00")
        created = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        self.assertEqual(created.status_code, 200, created.text)
        report_id = created.json()["id"]
        malformed = {
            **created.json()["issuer_concentration_assessment"],
            "portfolio_nav": "0.00",
            "total_equity_market_value": "1.00",
            "issuer_exposures": [
                {
                    "issuer_id": "TEST-ISSUER-1",
                    "issuer_id_type": "OTHER_VERIFIED",
                    "market_value": "1.00",
                    "portfolio_weight_pct": "100",
                    "listings": [],
                }
            ],
        }
        with self.account_sessions.begin() as db:
            report = db.get(PortfolioMonitoringReport, report_id)
            report.issuer_concentration_assessment = malformed

        replay = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        history = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        detail = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=headers,
        )

        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertIsNone(replay.json()["issuer_concentration_assessment"])
        self.assertIsNone(history.json()[0]["issuer_concentration_assessment"])
        self.assertIsNone(detail.json()["issuer_concentration_assessment"])
        with self.account_sessions() as db:
            report = db.get(PortfolioMonitoringReport, report_id)
            self.assertEqual(report.issuer_concentration_assessment, malformed)

    def test_invalid_ownership_and_cross_owner_report_reads_return_404(self):
        owner_headers = self._login("owner@example.com")
        other_headers = self._login("other@example.com")
        portfolio_id = self._portfolio(owner_headers)
        self._capture_baseline(portfolio_id)
        report_response = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=owner_headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        self.assertEqual(report_response.status_code, 200, report_response.text)
        report_id = report_response.json()["id"]
        owner_list = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=owner_headers,
        )
        owner_detail = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=owner_headers,
        )
        self.assertEqual(owner_list.json()[0]["exposure_risk_assessment"], report_response.json()["exposure_risk_assessment"])
        self.assertEqual(owner_detail.json()["exposure_risk_assessment"], report_response.json()["exposure_risk_assessment"])

        invalid_portfolio = self.client.post(
            "/api/portfolios/not-a-portfolio/monitor",
            headers=owner_headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        cross_owner_post = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=other_headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        cross_owner_list = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=other_headers,
        )
        cross_owner_get = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=other_headers,
        )

        self.assertEqual(invalid_portfolio.status_code, 404)
        self.assertEqual(owner_list.status_code, 200, owner_list.text)
        self.assertEqual(owner_detail.status_code, 200, owner_detail.text)
        self.assertEqual(cross_owner_post.status_code, 404)
        self.assertEqual(cross_owner_list.status_code, 404)
        self.assertEqual(cross_owner_get.status_code, 404)

    def test_unexpected_error_is_persisted_as_evaluation_failed(self):
        headers = self._login("failure@example.com")
        portfolio_id = self._portfolio(headers)
        self._capture_baseline(portfolio_id)

        with self.assertLogs(
            "src.services.portfolio_monitor", level="ERROR"
        ) as captured_logs:
            with patch(
                "src.services.portfolio_monitor.value_portfolio",
                side_effect=RuntimeError("test evaluation failure"),
            ):
                response = self.client.post(
                    f"/api/portfolios/{portfolio_id}/monitor",
                    headers=headers,
                    json={"session_date": _SESSION_DATE.isoformat()},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("test evaluation failure", "\n".join(captured_logs.output))
        report = response.json()
        self.assertEqual(report["status"], "EVALUATION_FAILED")
        self.assertEqual(
            report["error_details"]["code"],
            "monitoring_evaluation_failed",
        )
        self.assertEqual(
            report["error_details"]["message"],
            "Monitoring evaluation failed unexpectedly.",
        )
        self.assertNotIn("test evaluation failure", response.text)

        report_id = report["id"]
        listed = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        retrieved = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=headers,
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(retrieved.status_code, 200, retrieved.text)
        self.assertNotIn("test evaluation failure", listed.text)
        self.assertNotIn("test evaluation failure", retrieved.text)

    def test_exposure_assessment_failure_is_persisted_as_evaluation_failed(self):
        headers = self._login("exposure-failure@example.com")
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id)
        self._price()
        self._capture_baseline(
            portfolio_id,
            holdings=(("NSE", "TCS", "2", "90.00"),),
        )

        with patch(
            "src.services.portfolio_monitor.assess_portfolio_exposure",
            side_effect=RuntimeError("assessment computation failed"),
        ):
            response = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": _SESSION_DATE.isoformat()},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "EVALUATION_FAILED")
        self.assertIsNone(response.json()["exposure_risk_assessment"])
        self.assertNotIn("assessment computation failed", response.text)
        with self.account_sessions() as db:
            persisted = db.scalar(
                select(PortfolioMonitoringReport).where(
                    PortfolioMonitoringReport.portfolio_id == portfolio_id,
                    PortfolioMonitoringReport.session_date == _SESSION_DATE,
                )
            )
            self.assertEqual(persisted.status, "EVALUATION_FAILED")
            self.assertIsNone(persisted.exposure_risk_assessment)

    def test_logging_failure_does_not_prevent_failed_report_persistence(self):
        headers = self._login("logging-failure@example.com")
        portfolio_id = self._portfolio(headers)
        self._capture_baseline(portfolio_id)

        with (
            patch(
                "src.services.portfolio_monitor.value_portfolio",
                side_effect=RuntimeError("private diagnostic"),
            ),
            patch(
                "src.services.portfolio_monitor.logger.exception",
                side_effect=RuntimeError("logging handler failure"),
            ) as log_exception,
        ):
            response = self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={
                    "session_date": (
                        _SESSION_DATE + timedelta(days=1)
                    ).isoformat()
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "EVALUATION_FAILED")
        self.assertEqual(log_exception.call_count, 1)
        self.assertNotIn("private diagnostic", response.text)

    def test_legacy_error_details_are_sanitized_for_replay_list_and_detail(self):
        headers = self._login("legacy-error@example.com")
        portfolio_id = self._portfolio(headers)
        raw_details = {
            "code": "monitoring_evaluation_failed",
            "message": "legacy private exception message",
            "traceback": "C:\\internal\\app\\monitor.py:123",
            "unexpected": "sensitive internal field",
        }
        with self.account_sessions.begin() as db:
            legacy_report = PortfolioMonitoringReport(
                portfolio_id=portfolio_id,
                session_date=_SESSION_DATE,
                evaluated_at=datetime.now(timezone.utc),
                status="EVALUATION_FAILED",
                valuation_complete=False,
                portfolio_nav=None,
                evaluated_holdings_count=0,
                evaluated_holdings=[],
                data_quality_warnings=[],
                ledger_reconciliation=None,
                reason_codes=["evaluation_failed"],
                explanation="Monitoring evaluation failed unexpectedly.",
                policy_version="6.1",
                source_metadata=[],
                error_details=raw_details,
            )
            db.add(legacy_report)
            db.flush()
            report_id = legacy_report.id

        replay = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        listed = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        retrieved = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=headers,
        )

        expected_details = {
            "code": "monitoring_evaluation_failed",
            "message": "Monitoring evaluation failed. Please try again later.",
        }
        for response in (replay, listed, retrieved):
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("legacy private exception message", response.text)
            self.assertNotIn("C:\\internal\\app\\monitor.py:123", response.text)
            self.assertNotIn("sensitive internal field", response.text)
        self.assertEqual(replay.json()["error_details"], expected_details)
        self.assertEqual(listed.json()[0]["error_details"], expected_details)
        self.assertEqual(retrieved.json()["error_details"], expected_details)

        with self.account_sessions() as db:
            stored_report = db.get(PortfolioMonitoringReport, report_id)
            self.assertEqual(stored_report.error_details, raw_details)

    def test_malformed_and_missing_error_details_use_fallback_on_all_report_routes(self):
        headers = self._login("malformed-error@example.com")
        portfolio_id = self._portfolio(headers)
        unsafe_values = [
            {"code": "monitoring_evaluation_failed", "message": {"detail": "nested"}},
            {"code": 42, "message": "Monitoring evaluation failed unexpectedly."},
            {"code": "unexpected_error_code", "message": "Monitoring evaluation failed unexpectedly."},
            {"code": "monitoring_evaluation_failed", "message": "private unexpected message"},
            {"code": "monitoring_evaluation_failed"},
            17,
            None,
        ]
        session_dates = [
            _SESSION_DATE + timedelta(days=index)
            for index in range(len(unsafe_values))
        ]
        report_ids = []
        with self.account_sessions.begin() as db:
            for session_date, error_details in zip(session_dates, unsafe_values):
                report = PortfolioMonitoringReport(
                    portfolio_id=portfolio_id,
                    session_date=session_date,
                    evaluated_at=datetime.now(timezone.utc),
                    status="EVALUATION_FAILED",
                    valuation_complete=False,
                    portfolio_nav=None,
                    evaluated_holdings_count=0,
                    evaluated_holdings=[],
                    data_quality_warnings=[],
                    ledger_reconciliation=None,
                    reason_codes=["evaluation_failed"],
                    explanation="Monitoring evaluation failed unexpectedly.",
                    policy_version="6.1",
                    source_metadata=[],
                    error_details=error_details,
                )
                db.add(report)
                db.flush()
                report_ids.append(report.id)

        fallback = {
            "code": "monitoring_evaluation_failed",
            "message": "Monitoring evaluation failed. Please try again later.",
        }
        replay_responses = [
            self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": session_date.isoformat()},
            )
            for session_date in session_dates
        ]
        listed = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
            params={"limit": 100},
        )
        detail_responses = [
            self.client.get(
                f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
                headers=headers,
            )
            for report_id in report_ids
        ]
        self.assertEqual(listed.status_code, 200, listed.text)
        listed_by_date = {
            item["session_date"]: item for item in listed.json()
        }

        for index, (session_date, replay, detail) in enumerate(
            zip(session_dates, replay_responses, detail_responses)
        ):
            with self.subTest(error_details=unsafe_values[index]):
                self.assertEqual(replay.status_code, 200, replay.text)
                self.assertEqual(detail.status_code, 200, detail.text)
                self.assertEqual(replay.json()["error_details"], fallback)
                self.assertEqual(
                    listed_by_date[session_date.isoformat()]["error_details"],
                    fallback,
                )
                self.assertEqual(detail.json()["error_details"], fallback)
                for response in (replay, detail):
                    self.assertNotIn("nested", response.text)
                    self.assertNotIn("private unexpected message", response.text)
                self.assertNotIn("nested", listed.text)
                self.assertNotIn("private unexpected message", listed.text)

        with self.account_sessions() as db:
            stored_reports = db.scalars(
                select(PortfolioMonitoringReport)
                .where(PortfolioMonitoringReport.id.in_(report_ids))
                .order_by(PortfolioMonitoringReport.session_date)
            ).all()
            self.assertEqual(
                [report.error_details for report in stored_reports],
                unsafe_values,
            )

    def test_malformed_legacy_risk_assessment_is_null_on_all_monitoring_paths(self):
        headers = self._login("malformed-risk@example.com")
        portfolio_id = self._portfolio(headers, cash="250.00")
        self._capture_baseline(portfolio_id, cash="250.00")
        created = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": _SESSION_DATE.isoformat()},
        )
        self.assertEqual(created.status_code, 200, created.text)
        base_risk = created.json()["exposure_risk_assessment"]
        unsafe_values = []
        unsafe_values.append(None)
        unsafe_values.append([])
        unsafe_status = dict(base_risk)
        unsafe_status["assessment_status"] = []
        unsafe_values.append(unsafe_status)
        unsafe_number = dict(base_risk)
        unsafe_number["portfolio_nav"] = {"nested": "private legacy value"}
        unsafe_values.append(unsafe_number)
        unsafe_nonfinite_number = dict(base_risk)
        unsafe_nonfinite_number["portfolio_nav"] = "NaN"
        unsafe_values.append(unsafe_nonfinite_number)
        unsafe_integer_number = dict(base_risk)
        unsafe_integer_number["portfolio_nav"] = 250
        unsafe_values.append(unsafe_integer_number)
        unsafe_negative_number = dict(base_risk)
        unsafe_negative_number["portfolio_nav"] = "-1.00"
        unsafe_values.append(unsafe_negative_number)
        unsafe_alert = dict(base_risk)
        unsafe_alert["alerts"] = [{"code": []}]
        unsafe_values.append(unsafe_alert)
        unsafe_malformed_alert = dict(base_risk)
        unsafe_malformed_alert["alerts"] = [[]]
        unsafe_values.append(unsafe_malformed_alert)
        unsafe_listing = dict(base_risk)
        unsafe_listing["listing_exposures"] = [{"exchange": []}]
        unsafe_values.append(unsafe_listing)
        unsafe_duplicate_listings = dict(base_risk)
        duplicate_listing = {
            "exchange": "NSE",
            "symbol": "TCS",
            "quantity": "1",
            "market_value": "0.00",
            "portfolio_weight_pct": "0",
        }
        unsafe_duplicate_listings["listing_exposures"] = [
            duplicate_listing,
            dict(duplicate_listing),
        ]
        unsafe_values.append(unsafe_duplicate_listings)
        unsafe_inconsistent_alerts = dict(base_risk)
        unsafe_inconsistent_alerts["alerts"] = [
            {
                "code": "CASH_ALLOCATION_LOW",
                "severity": "MEDIUM",
                "metric_name": "cash_weight_pct",
                "affected_listing": None,
                "observed_value": "4.99",
                "threshold": "5",
                "policy_version": base_risk["policy_version"],
            }
        ]
        unsafe_values.append(unsafe_inconsistent_alerts)
        unsafe_unsupported_status = dict(base_risk)
        unsafe_unsupported_status["assessment_status"] = "UNKNOWN"
        unsafe_values.append(unsafe_unsupported_status)
        safe_with_unexpected_field = dict(base_risk)
        safe_with_unexpected_field["internal_diagnostic"] = "must not be returned"
        unsafe_values.append(safe_with_unexpected_field)
        session_dates = [
            _SESSION_DATE + timedelta(days=index + 1)
            for index in range(len(unsafe_values))
        ]
        report_ids = []
        with self.account_sessions.begin() as db:
            for session_date, risk_data in zip(session_dates, unsafe_values):
                if isinstance(risk_data, dict):
                    risk_data["session_date"] = session_date.isoformat()
                report = PortfolioMonitoringReport(
                    portfolio_id=portfolio_id,
                    session_date=session_date,
                    evaluated_at=datetime.now(timezone.utc),
                    status="NO_ACTION_REQUIRED",
                    valuation_complete=True,
                    portfolio_nav=Decimal("250.00"),
                    evaluated_holdings_count=0,
                    evaluated_holdings=[],
                    data_quality_warnings=[],
                    ledger_reconciliation={"available": True, "is_reconciled": True},
                    reason_codes=["monitoring_checks_clean"],
                    explanation="Historical report",
                    policy_version="6.1",
                    source_metadata=[],
                    error_details=None,
                    exposure_risk_assessment=risk_data,
                )
                db.add(report)
                db.flush()
                report_ids.append(report.id)

        replay_responses = [
            self.client.post(
                f"/api/portfolios/{portfolio_id}/monitor",
                headers=headers,
                json={"session_date": session_date.isoformat()},
            )
            for session_date in session_dates
        ]
        listed = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
            params={"limit": 100},
        )
        detail_responses = [
            self.client.get(
                f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
                headers=headers,
            )
            for report_id in report_ids
        ]
        listed_by_date = {
            report["session_date"]: report for report in listed.json()
        }
        for session_date, replay, detail, risk_data in zip(
            session_dates, replay_responses, detail_responses, unsafe_values
        ):
            with self.subTest(session_date=session_date):
                for response in (replay, listed, detail):
                    self.assertEqual(response.status_code, 200, response.text)
                    response_report = (
                        response.json()
                        if response is not listed
                        else listed_by_date[session_date.isoformat()]
                    )
                    self.assertEqual(
                        response_report["status"],
                        "NO_ACTION_REQUIRED",
                    )
                    self.assertEqual(
                        response_report["explanation"],
                        "Historical report",
                    )
                risk_responses = (
                    replay.json()["exposure_risk_assessment"],
                    listed_by_date[session_date.isoformat()][
                        "exposure_risk_assessment"
                    ],
                    detail.json()["exposure_risk_assessment"],
                )
                if risk_data is safe_with_unexpected_field:
                    expected_safe_risk = dict(base_risk)
                    expected_safe_risk["session_date"] = session_date.isoformat()
                    for risk_response in risk_responses:
                        self.assertEqual(risk_response, expected_safe_risk)
                        self.assertNotIn(
                            "internal_diagnostic",
                            risk_response,
                        )
                else:
                    self.assertEqual(risk_responses, (None, None, None))
        with self.account_sessions() as db:
            persisted = db.scalars(
                select(PortfolioMonitoringReport)
                .where(PortfolioMonitoringReport.id.in_(report_ids))
                .order_by(PortfolioMonitoringReport.session_date)
            ).all()
            self.assertEqual(
                [report.exposure_risk_assessment for report in persisted],
                unsafe_values,
            )

    def test_reordered_legacy_listings_cannot_forge_top_five_exposure(self):
        headers = self._login("reordered-risk@example.com")
        portfolio_id = self._portfolio(headers, cash="0.00")
        session_date = _SESSION_DATE + timedelta(days=30)
        holdings = [
            {
                "exchange": "NSE",
                "symbol": f"STOCK{index}",
                "quantity": "1",
                "market_value": str(value),
                "valuation_status": "VALUED",
                "market_date": session_date,
            }
            for index, value in enumerate(("30.00", "20.00", "15.00", "13.00", "12.00", "10.00"))
        ]
        malformed_assessment = assess_portfolio_exposure(
            session_date=session_date,
            valuation_complete=True,
            portfolio_nav=Decimal("100.00"),
            cash_balance=Decimal("0.00"),
            holdings=holdings,
            evaluated_at=datetime.now(timezone.utc),
        )
        malformed_assessment["listing_exposures"].reverse()
        malformed_assessment["top_five_weight_pct"] = "70.00"
        malformed_assessment["alerts"] = [
            alert
            for alert in malformed_assessment["alerts"]
            if alert["code"] != "TOP_FIVE_CONCENTRATION_HIGH"
        ]
        with self.account_sessions.begin() as db:
            report = PortfolioMonitoringReport(
                portfolio_id=portfolio_id,
                session_date=session_date,
                evaluated_at=datetime.now(timezone.utc),
                status="NO_ACTION_REQUIRED",
                valuation_complete=True,
                portfolio_nav=Decimal("100.00"),
                evaluated_holdings_count=len(holdings),
                evaluated_holdings=[],
                data_quality_warnings=[],
                ledger_reconciliation={"available": True, "is_reconciled": True},
                reason_codes=["monitoring_checks_clean"],
                explanation="Historical report",
                policy_version="6.1",
                source_metadata=[],
                error_details=None,
                exposure_risk_assessment=malformed_assessment,
            )
            db.add(report)
            db.flush()
            report_id = report.id

        replay = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": session_date.isoformat()},
        )
        listed = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports",
            headers=headers,
        )
        detail = self.client.get(
            f"/api/portfolios/{portfolio_id}/monitor/reports/{report_id}",
            headers=headers,
        )
        for response in (replay, listed, detail):
            self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(replay.json()["exposure_risk_assessment"])
        self.assertIsNone(listed.json()[0]["exposure_risk_assessment"])
        self.assertIsNone(detail.json()["exposure_risk_assessment"])
        with self.account_sessions() as db:
            stored = db.get(PortfolioMonitoringReport, report_id)
            self.assertEqual(
                stored.exposure_risk_assessment,
                malformed_assessment,
            )

    def test_historical_monitoring_rejects_stored_prices_after_session(self):
        headers = self._login("historical@example.com")
        portfolio_id = self._portfolio(headers)
        self._holding(headers, portfolio_id)
        self._price(market_date=_SESSION_DATE + timedelta(days=1))
        self._capture_baseline(
            portfolio_id,
            holdings=(("NSE", "TCS", "2", "90.00"),),
        )
        historical_date = _SESSION_DATE - timedelta(days=1)

        response = self.client.post(
            f"/api/portfolios/{portfolio_id}/monitor",
            headers=headers,
            json={"session_date": historical_date.isoformat()},
        )

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["status"], "DATA_UNAVAILABLE")
        self.assertIsNone(report["portfolio_nav"])
        self.assertEqual(
            report["exposure_risk_assessment"]["assessment_status"],
            "NOT_EVALUATED",
        )
        self.assertEqual(report["exposure_risk_assessment"]["alerts"], [])
        self.assertIn("price_after_session", [
            warning["code"] for warning in report["data_quality_warnings"]
        ])

    def test_migration_is_versioned_and_monitoring_table_is_separate(self):
        with self.account_engine.connect() as connection:
            revisions = connection.execute(
                text("SELECT revision FROM schema_migrations ORDER BY revision")
            ).scalars().all()
            tables = set(
                connection.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='portfolio_monitoring_reports'"
                    )
                ).scalars()
            )
        self.assertIn("0005_portfolio_monitoring_reports", revisions)
        self.assertIn("0006_portfolio_exposure_risk", revisions)
        self.assertIn("portfolio_monitoring_reports", tables)

    def test_service_rejects_unowned_portfolio(self):
        headers = self._login("service-owner@example.com")
        portfolio_id = self._portfolio(headers)
        with self.account_sessions() as account_db, self.market_sessions() as market_db:
            with self.assertRaises(PortfolioNotFoundError):
                monitor_portfolio(
                    account_db,
                    market_db,
                    "different-user-id",
                    portfolio_id,
                    _SESSION_DATE,
                )
