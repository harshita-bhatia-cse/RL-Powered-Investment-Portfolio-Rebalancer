"""Declarative base isolated from account and research databases."""

from sqlalchemy.orm import DeclarativeBase


class MarketDataBase(DeclarativeBase):
    """Base for the dedicated market-data database."""
