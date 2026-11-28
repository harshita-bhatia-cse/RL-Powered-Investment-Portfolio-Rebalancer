"""Persistent EOD prices for NSE/BSE listings."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    BigInteger,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.market_data.base import MarketDataBase


class EODPrice(MarketDataBase):
    __tablename__ = "eod_prices"
    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "symbol",
            "market_date",
            name="uq_eod_price_listing_date",
        ),
        CheckConstraint("exchange IN ('NSE', 'BSE')", name="ck_eod_price_exchange"),
        CheckConstraint("currency = 'INR'", name="ck_eod_price_currency_inr"),
        CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0",
            name="ck_eod_price_positive_ohlc",
        ),
        CheckConstraint(
            "high >= low AND high >= open AND high >= close "
            "AND low <= open AND low <= close",
            name="ck_eod_price_ohlc_relationship",
        ),
        CheckConstraint("volume IS NULL OR volume >= 0", name="ck_eod_price_volume"),
        CheckConstraint(
            "length(trim(provider)) > 0 AND length(trim(source)) > 0",
            name="ck_eod_price_provider_source",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(3), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="INR", server_default="INR"
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    source: Mapped[str] = mapped_column(String(200), nullable=False)
    valuation_permitted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
