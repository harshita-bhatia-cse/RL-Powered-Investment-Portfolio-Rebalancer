"""Historical portfolio analytics and risk management services."""

from .portfolio_metrics import calculate_portfolio_metrics, prices_to_returns
from .risk_metrics import calculate_risk_metrics
from .asset_analysis import analyze_assets
from .correlation_service import analyze_correlations
from .allocation_service import validate_weights, allocation_summary
from .benchmark_service import compare_benchmarks
from .india_market_analysis import analyze_india_market, calculate_indicators, build_stock_summary
from .india_sector_intelligence import (
    analyze_india_sector_intelligence,
    calculate_sector_intelligence,
    load_sector_metadata,
)
from .india_unified_intelligence import (
    analyze_unified_india_intelligence,
    build_unified_india_intelligence,
)

__all__ = [
    "calculate_portfolio_metrics",
    "prices_to_returns",
    "calculate_risk_metrics",
    "analyze_assets",
    "analyze_correlations",
    "validate_weights",
    "allocation_summary",
    "compare_benchmarks",
    "analyze_india_market",
    "calculate_indicators",
    "build_stock_summary",
    "analyze_india_sector_intelligence",
    "calculate_sector_intelligence",
    "load_sector_metadata",
    "analyze_unified_india_intelligence",
    "build_unified_india_intelligence",
]
