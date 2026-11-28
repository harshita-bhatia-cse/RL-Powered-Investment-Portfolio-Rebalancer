"""Read-only portfolio data-integrity monitoring for explicit EOD sessions."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.models import (
    Holding,
    Portfolio,
    PortfolioMonitoringReport,
)
from src.market_data.service import PortfolioNotFoundError, value_portfolio
from src.services.portfolio_ledger import LedgerError, reconcile_portfolio
from src.services.portfolio_exposure_risk import assess_portfolio_exposure
from src.services.issuer_concentration import (
    IssuerReferenceDataError,
    assess_portfolio_issuer_concentration,
    load_issuer_reference_data,
)
from src.services.sector_concentration import (
    SectorReferenceDataError,
    assess_portfolio_sector_concentration,
    load_sector_reference_data,
)

MonitoringStatus = Literal[
    "NO_ACTION_REQUIRED",
    "ACTION_REQUIRED",
    "DATA_UNAVAILABLE",
    "EVALUATION_FAILED",
]
MONITORING_POLICY_VERSION = "6.1"
logger = logging.getLogger(__name__)


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _existing_report(
    db: Session,
    portfolio_id: str,
    session_date: date,
) -> PortfolioMonitoringReport | None:
    return db.scalar(
        select(PortfolioMonitoringReport).where(
            PortfolioMonitoringReport.portfolio_id == portfolio_id,
            PortfolioMonitoringReport.session_date == session_date,
        )
    )


def _new_report(
    portfolio_id: str,
    session_date: date,
    *,
    status: MonitoringStatus,
    valuation_complete: bool,
    portfolio_nav: Decimal | None,
    evaluated_holdings: list[dict],
    warnings: list[dict],
    reconciliation: dict | None,
    reason_codes: list[str],
    explanation: str,
    sources: list[dict],
    error_details: dict | None = None,
    exposure_risk_assessment: dict | None = None,
    issuer_concentration_assessment: dict | None = None,
    sector_concentration_assessment: dict | None = None,
) -> PortfolioMonitoringReport:
    return PortfolioMonitoringReport(
        portfolio_id=portfolio_id,
        session_date=session_date,
        evaluated_at=datetime.now(timezone.utc),
        status=status,
        valuation_complete=valuation_complete,
        portfolio_nav=portfolio_nav,
        evaluated_holdings_count=len(evaluated_holdings),
        evaluated_holdings=_json_value(evaluated_holdings),
        data_quality_warnings=_json_value(warnings),
        ledger_reconciliation=_json_value(reconciliation),
        reason_codes=reason_codes,
        explanation=explanation,
        policy_version=MONITORING_POLICY_VERSION,
        source_metadata=_json_value(sources),
        error_details=_json_value(error_details),
        exposure_risk_assessment=_json_value(exposure_risk_assessment),
        issuer_concentration_assessment=_json_value(
            issuer_concentration_assessment
        ),
        sector_concentration_assessment=_json_value(
            sector_concentration_assessment
        ),
    )


def _evaluate(
    account_db: Session,
    market_db: Session,
    user_id: str,
    portfolio_id: str,
    session_date: date,
) -> PortfolioMonitoringReport:
    portfolio = account_db.scalar(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id,
        )
    )
    if portfolio is None:
        raise PortfolioNotFoundError("Portfolio not found")

    holdings = account_db.scalars(
        select(Holding)
        .where(Holding.portfolio_id == portfolio_id)
        .order_by(Holding.exchange, Holding.symbol)
    ).all()
    valuation = value_portfolio(
        account_db,
        market_db,
        user_id,
        portfolio_id,
        today=session_date,
    )

    warnings: list[dict] = []
    category_codes = {
        "missing_prices": "missing_price",
        "stale_prices": "stale_price",
        "unavailable_prices": "unpermitted_price",
    }
    for category, code in category_codes.items():
        for item in valuation[category]:
            warnings.append(
                _json_value(
                    {
                        "code": code,
                        "exchange": item["exchange"],
                        "symbol": item["symbol"],
                        "reason": item.get("reason")
                        or f"Price is not usable because it is {code}.",
                        "market_date": item.get("market_date"),
                        "provider": item.get("provider"),
                        "source": item.get("source"),
                        "fetched_at": item.get("fetched_at"),
                    }
                )
            )

    valued_by_listing = {
        (item["exchange"], item["symbol"]): item
        for item in valuation["valued_holdings"]
    }
    future_price_listings: set[tuple[str, str]] = set()
    sources_by_key: dict[tuple[str, str, str], dict] = {}
    evaluated_holdings = []
    for holding in holdings:
        listing = (holding.exchange, holding.symbol)
        valued = valued_by_listing.get(listing)
        if valued is not None and valued["market_date"] > session_date:
            future_price_listings.add(listing)
            warnings.append(
                {
                    "code": "price_after_session",
                    "exchange": holding.exchange,
                    "symbol": holding.symbol,
                    "reason": (
                        "The latest permitted stored price is later than the "
                        "requested monitoring session."
                    ),
                    "market_date": valued["market_date"].isoformat(),
                    "provider": valued["provider"],
                    "source": valued["source"],
                    "fetched_at": valued["fetched_at"].isoformat(),
                }
            )
            valued = None

        row = {
            "exchange": holding.exchange,
            "symbol": holding.symbol,
            "listing": f"{holding.exchange}:{holding.symbol}",
            "quantity": str(holding.quantity),
            "valuation_status": "VALUED" if valued is not None else "UNAVAILABLE",
            "market_date": None,
            "close": None,
            "market_value": None,
            "provider": None,
            "source": None,
            "fetched_at": None,
        }
        if valued is not None:
            row.update(
                {
                    "market_date": valued["market_date"].isoformat(),
                    "close": str(valued["close"]),
                    "market_value": str(valued["market_value"]),
                    "provider": valued["provider"],
                    "source": valued["source"],
                    "fetched_at": valued["fetched_at"].isoformat(),
                }
            )
            source_key = (
                valued["provider"],
                valued["source"],
                valued["market_date"].isoformat(),
            )
            sources_by_key[source_key] = {
                "provider": valued["provider"],
                "source": valued["source"],
                "market_date": valued["market_date"].isoformat(),
            }
        evaluated_holdings.append(row)

    valuation_complete = bool(valuation["valuation_complete"]) and not future_price_listings
    nav = valuation["total_portfolio_value"] if valuation_complete else None

    reconciliation_data = None
    reason_codes: list[str] = []
    reconciliation_clean = False
    reconciliation_available = False
    try:
        reconciliation = reconcile_portfolio(account_db, portfolio_id)
    except LedgerError as error:
        reconciliation_data = {
            "available": False,
            "is_reconciled": None,
            "reason": str(error),
        }
        reason_codes.append("ledger_reconciliation_unavailable")
        warnings.append(
            {
                "code": "ledger_reconciliation_unavailable",
                "reason": str(error),
            }
        )
    else:
        reconciliation_available = True
        reconciliation_data = {
            "available": True,
            "is_reconciled": reconciliation.is_reconciled,
            "baseline_timestamp": reconciliation.baseline_timestamp,
            "expected_cash": reconciliation.expected_cash,
            "actual_cash": reconciliation.actual_cash,
            "cash_difference": reconciliation.cash_difference,
            "expected_holdings": reconciliation.expected_holdings,
            "actual_holdings": reconciliation.actual_holdings,
            "holding_discrepancies": reconciliation.holding_discrepancies,
        }
        reconciliation_clean = reconciliation.is_reconciled
        if not reconciliation_clean:
            reason_codes.append("ledger_discrepancy")
            warnings.append(
                {
                    "code": "ledger_discrepancy",
                    "reason": (
                        "The opening ledger baseline and subsequent posted "
                        "transactions do not match current account state."
                    ),
                    "cash_difference": str(reconciliation.cash_difference),
                    "holding_discrepancies": _json_value(
                        reconciliation.holding_discrepancies
                    ),
                }
            )

    if not valuation_complete:
        status: MonitoringStatus = "DATA_UNAVAILABLE"
        reason_codes.insert(0, "required_price_unavailable")
        explanation = (
            "Monitoring could not complete because one or more holdings lack a "
            "fresh, permitted price on or before the requested session date."
        )
    elif not reconciliation_clean:
        status = "ACTION_REQUIRED"
        if not reason_codes:
            reason_codes.append("ledger_discrepancy")
        explanation = (
            "Valuation completed, but ledger reconciliation is unavailable or "
            "found a discrepancy that requires data-integrity follow-up."
        )
    else:
        status = "NO_ACTION_REQUIRED"
        reason_codes.append("monitoring_checks_clean")
        explanation = (
            "Configured data-quality and ledger checks completed without a "
            "detected issue. This is not an investment recommendation."
        )

    warnings_by_listing = {
        (warning.get("exchange"), warning.get("symbol")): warning
        for warning in warnings
        if warning.get("exchange") and warning.get("symbol")
    }
    unavailable_listings = []
    for row in evaluated_holdings:
        if row["valuation_status"] != "UNAVAILABLE":
            continue
        warning = warnings_by_listing.get((row["exchange"], row["symbol"]), {})
        unavailable_listings.append(
            {
                "exchange": row["exchange"],
                "symbol": row["symbol"],
                "reason": warning.get("reason")
                or "No valid fresh, permitted stored EOD price is available.",
            }
        )

    exposure_risk_assessment = assess_portfolio_exposure(
        session_date=session_date,
        valuation_complete=valuation_complete,
        portfolio_nav=nav,
        cash_balance=valuation["cash_balance"],
        holdings=evaluated_holdings,
        unavailable_listings=unavailable_listings,
        data_quality_warnings=warnings,
    )
    try:
        issuer_reference_data = load_issuer_reference_data()
    except IssuerReferenceDataError:
        issuer_reference_data = None
    issuer_concentration_assessment = assess_portfolio_issuer_concentration(
        session_date=session_date,
        valuation_complete=valuation_complete,
        portfolio_nav=nav,
        cash_balance=valuation["cash_balance"],
        holdings=evaluated_holdings,
        reference_data=issuer_reference_data,
        ledger_reconciliation_available=reconciliation_available,
        ledger_reconciled=reconciliation_clean,
    )
    try:
        sector_reference_data = load_sector_reference_data()
    except SectorReferenceDataError:
        sector_reference_data = None
    sector_concentration_assessment = assess_portfolio_sector_concentration(
        session_date=session_date,
        issuer_assessment=issuer_concentration_assessment,
        reference_data=sector_reference_data,
    )

    return _new_report(
        portfolio_id,
        session_date,
        status=status,
        valuation_complete=valuation_complete,
        portfolio_nav=nav,
        evaluated_holdings=evaluated_holdings,
        warnings=warnings,
        reconciliation=reconciliation_data,
        reason_codes=reason_codes,
        explanation=explanation,
        sources=list(sources_by_key.values()),
        exposure_risk_assessment=exposure_risk_assessment,
        issuer_concentration_assessment=issuer_concentration_assessment,
        sector_concentration_assessment=sector_concentration_assessment,
    )


def monitor_portfolio(
    account_db: Session,
    market_db: Session,
    user_id: str,
    portfolio_id: str,
    session_date: date,
) -> PortfolioMonitoringReport:
    """Evaluate one portfolio once for an explicit exchange session date.

    A unique portfolio/session key makes retries return the original persisted
    report. Only the monitoring report table is written; account ledger state
    and stored market data are read-only.
    """
    if not isinstance(session_date, date) or isinstance(session_date, datetime):
        raise ValueError("session_date must be an explicit calendar date")
    portfolio = account_db.scalar(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id,
        )
    )
    if portfolio is None:
        raise PortfolioNotFoundError("Portfolio not found")

    existing = _existing_report(account_db, portfolio_id, session_date)
    if existing is not None:
        return existing

    try:
        report = _evaluate(
            account_db,
            market_db,
            user_id,
            portfolio_id,
            session_date,
        )
    except Exception:
        try:
            logger.exception(
                "Portfolio monitoring evaluation failed",
                extra={
                    "portfolio_id": portfolio_id,
                    "session_date": session_date.isoformat(),
                },
            )
        except Exception:
            # Logging must not prevent persisting the safe failure report.
            pass
        report = _new_report(
            portfolio_id,
            session_date,
            status="EVALUATION_FAILED",
            valuation_complete=False,
            portfolio_nav=None,
            evaluated_holdings=[],
            warnings=[],
            reconciliation=None,
            reason_codes=["evaluation_failed"],
            explanation="Monitoring evaluation failed unexpectedly.",
            sources=[],
            error_details={
                "code": "monitoring_evaluation_failed",
                "message": "Monitoring evaluation failed unexpectedly.",
            },
        )

    account_db.add(report)
    try:
        account_db.commit()
    except IntegrityError:
        account_db.rollback()
        existing = _existing_report(account_db, portfolio_id, session_date)
        if existing is not None:
            return existing
        raise
    account_db.refresh(report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run read-only portfolio monitoring for one NSE/BSE session"
    )
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument(
        "--session-date",
        required=True,
        type=date.fromisoformat,
        help="Explicit exchange session date in YYYY-MM-DD format",
    )
    args = parser.parse_args(argv)

    from src.database.migrations import initialize_database
    from src.database.models import Portfolio
    from src.database.session import SessionLocal, engine
    from src.market_data.migrations import initialize_market_database
    from src.market_data.session import SessionLocal as MarketSessionLocal
    from src.market_data.session import engine as market_engine

    initialize_database(engine)
    initialize_market_database(market_engine)
    with SessionLocal() as account_db, MarketSessionLocal() as market_db:
        portfolio = account_db.get(Portfolio, args.portfolio_id)
        if portfolio is None:
            parser.error("portfolio was not found")
        report = monitor_portfolio(
            account_db,
            market_db,
            portfolio.user_id,
            portfolio.id,
            args.session_date,
        )
        print(
            f"{report.status} portfolio={report.portfolio_id} "
            f"session={report.session_date} "
            f"reasons={','.join(report.reason_codes)}"
        )


if __name__ == "__main__":
    main()
