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

__all__ = [
    "GENERATED_SERIES",
    "REGIME_NAMES",
    "RegimeConfig",
    "SyntheticDataset",
    "SyntheticMarketConfig",
    "SyntheticScenarioSplits",
    "generate_synthetic_market",
    "generate_synthetic_splits",
]
