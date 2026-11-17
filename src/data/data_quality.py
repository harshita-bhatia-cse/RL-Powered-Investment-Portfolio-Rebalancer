"""Data-driven quality checks for stored historical price records."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def calculate_price_quality(price_data: pd.DataFrame, as_of: datetime | None = None) -> dict:
    """Calculate quality scores from actual OHLCV rows.

    Scores are percentages and are returned only when at least one row exists.
    """
    if price_data.empty:
        raise ValueError("No historical price data is available")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in price_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required price columns: {', '.join(missing_columns)}")

    frame = price_data.copy()
    total_cells = len(frame) * len(REQUIRED_COLUMNS)
    completeness = 100.0 * frame[REQUIRED_COLUMNS].notna().sum().sum() / total_cells
    numeric = frame[["open", "high", "low", "close", "adj_close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_prices = (
        numeric[["open", "high", "low", "close", "adj_close"]].gt(0).all(axis=1)
        & numeric["volume"].ge(0)
        & (numeric["high"] >= numeric[["open", "low", "close"]].max(axis=1))
        & (numeric["low"] <= numeric[["open", "high", "close"]].min(axis=1))
    )
    accuracy = 100.0 * float(valid_prices.mean())
    duplicate_rows = frame.duplicated(subset=["ticker", "date"]).sum()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid_dates = dates.notna()
    consistency = 100.0 * (1.0 - float(duplicate_rows) / len(frame)) * float(valid_dates.mean())
    reference = as_of or datetime.now(timezone.utc)
    latest = dates.max()
    age_days = max((pd.Timestamp(reference).tz_localize(None) - latest).days, 0)
    timeliness = max(0.0, 100.0 * (1.0 - age_days / 30.0))
    return {
        "completeness": round(float(completeness), 2),
        "accuracy": round(float(accuracy), 2),
        "consistency": round(float(consistency), 2),
        "timeliness": round(float(timeliness), 2),
        "record_count": int(len(frame)),
        "latest_date": latest.strftime("%Y-%m-%d") if pd.notna(latest) else None,
    }
