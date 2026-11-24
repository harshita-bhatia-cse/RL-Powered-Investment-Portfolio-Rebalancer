"""
Data Layer Module
================

Handles all data acquisition, processing, and storage operations:
- WRDS data fetching (S&P 500 components, fundamentals, prices)
- Data preprocessing and cleaning
- Database operations and caching
"""

from .india_fundamentals import (
    IndiaFundamentalCSVProvider,
    ingest_india_fundamentals,
    normalize_india_fundamentals,
)
from .india_news import (
    IndiaNewsCSVProvider,
    ingest_india_news,
    normalize_india_news,
)
from .india_stock_universe import (
    INDIA_STOCK_UNIVERSE,
    SUPPORTED_SECTORS,
    IndiaStock,
    build_india_universe_report,
    get_india_stock_universe,
    india_stock_universe_symbols,
    universe_to_sector_metadata,
    validate_india_stock_universe,
)

__all__ = [
    "IndiaFundamentalCSVProvider",
    "ingest_india_fundamentals",
    "normalize_india_fundamentals",
    "IndiaNewsCSVProvider",
    "ingest_india_news",
    "normalize_india_news",
    "INDIA_STOCK_UNIVERSE",
    "SUPPORTED_SECTORS",
    "IndiaStock",
    "build_india_universe_report",
    "get_india_stock_universe",
    "india_stock_universe_symbols",
    "universe_to_sector_metadata",
    "validate_india_stock_universe",
]
