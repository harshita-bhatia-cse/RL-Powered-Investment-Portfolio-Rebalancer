"""Strategies supported by the CSV-backed historical portfolio environment."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.rl.environment.portfolio_env import ASSET_UNIVERSE


class HistoricalStrategy(Protocol):
    """Produce a target allocation or request a no-trade holding transition."""

    name: str

    def action(
        self, observation: np.ndarray, *, initial_step: bool
    ) -> np.ndarray | None:
        """Return asset-plus-cash target weights, or ``None`` to hold."""


class EqualWeightStrategy:
    """Target equal weights across the six ETFs and the cash sleeve each step."""

    name = "equal_weight"

    def action(self, observation: np.ndarray, *, initial_step: bool) -> np.ndarray:
        del observation, initial_step
        weight = 1.0 / (len(ASSET_UNIVERSE) + 1)
        return np.full(len(ASSET_UNIVERSE) + 1, weight, dtype=np.float32)


class BuyAndHoldStrategy:
    """Establish equal weights once, then retain holdings without rebalancing."""

    name = "buy_and_hold"

    def action(
        self, observation: np.ndarray, *, initial_step: bool
    ) -> np.ndarray | None:
        del observation
        if not initial_step:
            return None
        weight = 1.0 / (len(ASSET_UNIVERSE) + 1)
        return np.full(len(ASSET_UNIVERSE) + 1, weight, dtype=np.float32)


SUPPORTED_STRATEGIES = (EqualWeightStrategy.name, BuyAndHoldStrategy.name)


def create_strategy(name: str) -> HistoricalStrategy:
    """Create one supported strategy by its canonical API name."""
    if name == EqualWeightStrategy.name:
        return EqualWeightStrategy()
    if name == BuyAndHoldStrategy.name:
        return BuyAndHoldStrategy()
    raise ValueError(
        f"Unsupported strategy {name!r}; choose from {', '.join(SUPPORTED_STRATEGIES)}"
    )
