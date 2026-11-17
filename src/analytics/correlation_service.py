"""Correlation and diversification analysis."""

from __future__ import annotations

import pandas as pd


def analyze_correlations(returns: pd.DataFrame, high_correlation_threshold: float = 0.8) -> dict:
    """Calculate the return correlation matrix and highly correlated pairs."""
    if returns.shape[1] < 2:
        raise ValueError("At least two assets are required for correlation analysis")
    matrix = returns.corr()
    pairs = []
    for left_index, left in enumerate(matrix.columns):
        for right in matrix.columns[left_index + 1:]:
            value = float(matrix.loc[left, right])
            if abs(value) >= high_correlation_threshold:
                pairs.append({"asset_a": left, "asset_b": right, "correlation": value})
    return {
        "matrix": matrix,
        "highly_correlated_pairs": pairs,
        "average_pairwise_correlation": float(
            matrix.where(~pd.DataFrame(
                [[i == j for j in matrix.columns] for i in matrix.columns],
                index=matrix.index, columns=matrix.columns,
            )).stack().mean()
        ),
    }
