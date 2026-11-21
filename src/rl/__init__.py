"""Custom reinforcement-learning components for portfolio research."""

from .synthetic_data import (
    GENERATED_SERIES,
    REGIME_NAMES,
    RegimeConfig,
    SyntheticDataset,
    SyntheticMarketConfig,
    SyntheticScenarioSplits,
    generate_synthetic_market,
    generate_synthetic_splits,
)
from .phase4_evaluation import (
    calculate_portfolio_metrics,
    evaluate_phase4,
    evaluate_ppo_synthetic,
    save_phase4_csv,
    save_phase4_results,
)
from .phase4_training import (
    HistoricalDataAuthorization,
    HistoricalDataAuthorizationError,
    TrainingOrchestrator,
)

__all__ = [
    "GENERATED_SERIES",
    "REGIME_NAMES",
    "RegimeConfig",
    "SyntheticDataset",
    "SyntheticMarketConfig",
    "SyntheticScenarioSplits",
    "generate_synthetic_market",
    "generate_synthetic_splits",
    "calculate_portfolio_metrics",
    "evaluate_phase4",
    "evaluate_ppo_synthetic",
    "save_phase4_csv",
    "save_phase4_results",
    "HistoricalDataAuthorization",
    "HistoricalDataAuthorizationError",
    "TrainingOrchestrator",
]
