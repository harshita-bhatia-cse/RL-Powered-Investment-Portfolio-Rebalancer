"""Portfolio allocation validation and concentration analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate non-negative portfolio weights that sum to exactly 100% within tolerance."""
    if not weights:
        raise ValueError("At least one portfolio weight is required")
    normalized = {str(asset): float(weight) for asset, weight in weights.items()}
    if any(not np.isfinite(weight) or weight < 0 for weight in normalized.values()):
        raise ValueError("Portfolio weights must be finite and non-negative")
    if not np.isclose(sum(normalized.values()), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must sum to 1.0")
    return normalized


def allocation_summary(
    target_weights: dict[str, float],
    current_weights: dict[str, float] | None = None,
    sectors: dict[str, str] | None = None,
) -> dict:
    """Return target/current allocation, deviations, concentration, and sector exposure."""
    target = validate_weights(target_weights)
    current = validate_weights(current_weights) if current_weights else {}
    assets = sorted(set(target) | set(current))
    allocation = pd.DataFrame({
        "target_weight": [target.get(asset, 0.0) for asset in assets],
        "current_weight": [current.get(asset, 0.0) for asset in assets],
    }, index=assets)
    allocation["weight_deviation"] = allocation["current_weight"] - allocation["target_weight"]
    sector_exposure = {}
    if sectors:
        for asset, weight in allocation["current_weight"].items():
            sector = sectors.get(asset)
            if sector:
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + float(weight)
    return {
        "allocation": allocation,
        "largest_target_weight": float(allocation["target_weight"].max()),
        "herfindahl_index": float((allocation["target_weight"] ** 2).sum()),
        "sector_exposure": sector_exposure,
    }
