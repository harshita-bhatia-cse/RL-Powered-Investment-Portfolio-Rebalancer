"""SQLAlchemy models for users and their India-listed portfolios."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


def _new_id() -> str:
    return str(uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    portfolios: Mapped[list[Portfolio]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Portfolio(Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_portfolio_owner_name"),
        CheckConstraint("base_currency = 'INR'", name="ck_portfolio_currency_inr"),
        CheckConstraint("cash_balance >= 0", name="ck_portfolio_cash_nonnegative"),
        Index("ix_portfolios_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    base_currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="INR", server_default="INR"
    )
    cash_balance: Mapped[Decimal] = mapped_column(
        Numeric(20, 2), nullable=False, default=Decimal("0.00"), server_default="0.00"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    owner: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped[list[Holding]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    ledger_baseline: Mapped[PortfolioLedgerBaseline | None] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Holding(Base):
    __tablename__ = "holdings"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "exchange", "symbol", name="uq_holding_listing"
        ),
        CheckConstraint("exchange IN ('NSE', 'BSE')", name="ck_holding_exchange"),
        CheckConstraint("quantity > 0", name="ck_holding_quantity_positive"),
        CheckConstraint(
            "average_purchase_price > 0", name="ck_holding_average_price_positive"
        ),
        Index("ix_holdings_portfolio", "portfolio_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    portfolio_id: Mapped[str] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    exchange: Mapped[str] = mapped_column(String(3), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    average_purchase_price: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id",
            "idempotency_key",
            name="uq_transactions_portfolio_idempotency_key",
        ),
        CheckConstraint(
            "transaction_type IN ('BUY', 'SELL', 'CASH_DEPOSIT', 'CASH_WITHDRAWAL')",
            name="ck_transaction_type",
        ),
        CheckConstraint(
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
            ")",
            name="ck_transaction_fields_match_type",
        ),
        CheckConstraint("fees >= 0", name="ck_transaction_fees_nonnegative"),
        CheckConstraint(
            "transaction_type IN ('BUY', 'SELL') OR fees = 0",
            name="ck_transaction_cash_fees_zero",
        ),
        Index("ix_transactions_portfolio_date", "portfolio_id", "transaction_date"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    portfolio_id: Mapped[str] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(3))
    symbol: Mapped[str | None] = mapped_column(String(32))
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 6))
    cash_amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 2))
    fees: Mapped[Decimal] = mapped_column(
        Numeric(20, 2), nullable=False, default=Decimal("0.00"), server_default="0.00"
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")


class PortfolioLedgerBaseline(Base):
    __tablename__ = "portfolio_ledger_baselines"

    portfolio_id: Mapped[str] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), primary_key=True
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)

    portfolio: Mapped[Portfolio] = relationship(back_populates="ledger_baseline")
    holdings: Mapped[list[PortfolioLedgerBaselineHolding]] = relationship(
        back_populates="baseline",
        cascade="all, delete-orphan",
        order_by=(
            "PortfolioLedgerBaselineHolding.exchange, "
            "PortfolioLedgerBaselineHolding.symbol"
        ),
    )


class PortfolioLedgerBaselineHolding(Base):
    __tablename__ = "portfolio_ledger_baseline_holdings"

    portfolio_id: Mapped[str] = mapped_column(
        ForeignKey("portfolio_ledger_baselines.portfolio_id", ondelete="CASCADE"),
        primary_key=True,
    )
    exchange: Mapped[str] = mapped_column(String(3), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    average_purchase_price: Mapped[Decimal] = mapped_column(
        Numeric(20, 6), nullable=False
    )

    baseline: Mapped[PortfolioLedgerBaseline] = relationship(
        back_populates="holdings"
    )


class PortfolioMonitoringReport(Base):
    __tablename__ = "portfolio_monitoring_reports"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id",
            "session_date",
            name="uq_monitoring_report_portfolio_session",
        ),
        CheckConstraint(
            "status IN ('NO_ACTION_REQUIRED', 'ACTION_REQUIRED', "
            "'DATA_UNAVAILABLE', 'EVALUATION_FAILED')",
            name="ck_monitoring_report_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    portfolio_id: Mapped[str] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    valuation_complete: Mapped[bool] = mapped_column(nullable=False)
    portfolio_nav: Mapped[Decimal | None] = mapped_column(Numeric(20, 2))
    evaluated_holdings_count: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluated_holdings: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    data_quality_warnings: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    ledger_reconciliation: Mapped[dict | None] = mapped_column(JSON)
    reason_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_metadata: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    error_details: Mapped[dict | None] = mapped_column(JSON)
    exposure_risk_assessment: Mapped[dict | None] = mapped_column(JSON)
    issuer_concentration_assessment: Mapped[dict | None] = mapped_column(JSON)
    sector_concentration_assessment: Mapped[dict | None] = mapped_column(JSON)
