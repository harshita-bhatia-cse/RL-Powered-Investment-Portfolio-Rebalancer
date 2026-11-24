"""Offline technical and performance analysis for validated Indian market data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "india_daily" / "validated" / "indian_etf_validated.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "india_daily" / "analysis"

REQUIRED_COLUMNS = (
    "Date",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Adjusted Close",
    "Volume",
)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder-style RSI, retaining NaN until observations are sufficient."""
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate stock-wise indicators without filling or inventing prices."""
    required = set(REQUIRED_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing required India market columns: " + ", ".join(missing))

    working = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Symbol"] = working["Symbol"].astype("string").str.strip().str.upper()
    if working["Date"].isna().any():
        raise ValueError("India analysis input contains invalid dates")
    if working["Symbol"].isna().any() or working["Symbol"].eq("").any():
        raise ValueError("India analysis input contains missing symbols")

    for column in ("Open", "High", "Low", "Close", "Adjusted Close"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
        if not np.isfinite(working[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"India analysis input contains non-finite values in {column}")
        if (working[column] <= 0).any():
            raise ValueError(f"India analysis input contains non-positive values in {column}")
    working["Volume"] = pd.to_numeric(working["Volume"], errors="coerce")
    if not np.isfinite(working["Volume"].to_numpy(dtype=np.float64)).all():
        raise ValueError("India analysis input contains non-finite volume")
    if (working["Volume"] < 0).any():
        raise ValueError("India analysis input contains negative volume")

    if working.duplicated(subset=["Symbol", "Date"], keep=False).any():
        raise ValueError("India analysis input contains duplicate Symbol/Date records")

    inconsistent = (
        (working["High"] < working["Open"])
        | (working["High"] < working["Low"])
        | (working["High"] < working["Close"])
        | (working["Low"] > working["Open"])
        | (working["Low"] > working["High"])
        | (working["Low"] > working["Close"])
    )
    if inconsistent.any():
        raise ValueError("India analysis input contains inconsistent OHLC values")

    working = working.sort_values(["Symbol", "Date"], kind="stable").copy()
    pieces: list[pd.DataFrame] = []

    for symbol, symbol_frame in working.groupby("Symbol", sort=True):
        result = symbol_frame.sort_values("Date").copy()
        price = result["Adjusted Close"]

        result["SMA_20"] = price.rolling(window=20, min_periods=20).mean()
        result["SMA_50"] = price.rolling(window=50, min_periods=50).mean()
        result["EMA_20"] = price.ewm(span=20, adjust=False, min_periods=20).mean()
        result["EMA_50"] = price.ewm(span=50, adjust=False, min_periods=50).mean()
        result["Daily_Return"] = price.pct_change(fill_method=None)
        result["Rolling_Volatility_20"] = (
            result["Daily_Return"].rolling(window=20, min_periods=20).std(ddof=1)
            * np.sqrt(252.0)
        )
        result["Momentum_20"] = price.pct_change(periods=20, fill_method=None)

        ema_12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = price.ewm(span=26, adjust=False, min_periods=26).mean()
        result["MACD"] = ema_12 - ema_26
        result["MACD_Signal"] = result["MACD"].ewm(
            span=9,
            adjust=False,
            min_periods=9,
        ).mean()
        result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]
        result["RSI_14"] = rsi(price, period=14)
        pieces.append(result)

    if not pieces:
        raise ValueError("No Indian stock observations are available")
    return pd.concat(pieces, ignore_index=True)


def classify_trend(row: pd.Series) -> str:
    """Produce a descriptive trend label without making an investment recommendation."""
    price = row["Adjusted Close"]
    sma20 = row["SMA_20"]
    sma50 = row["SMA_50"]
    macd = row["MACD"]
    signal = row["MACD_Signal"]
    rsi14 = row["RSI_14"]

    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi14):
        return "insufficient history"
    if price > sma20 and sma20 > sma50 and macd > signal and rsi14 < 70:
        return "strong upward"
    if price > sma20 and macd > signal:
        return "upward"
    if price < sma20 and sma20 < sma50 and macd < signal and rsi14 > 30:
        return "strong downward"
    if price < sma20 and macd < signal:
        return "downward"
    return "sideways"


def build_stock_summary(indicators: pd.DataFrame) -> pd.DataFrame:
    """Build stock-wise performance and trend records."""
    rows: list[dict[str, object]] = []

    for symbol, frame in indicators.groupby("Symbol", sort=True):
        frame = frame.sort_values("Date").copy()
        valid = frame["Daily_Return"].dropna()
        latest = frame.iloc[-1]

        if len(valid) > 0:
            wealth = (1.0 + valid).cumprod()
            period_return = float(wealth.iloc[-1] - 1.0)
            calendar_days = max(
                (frame["Date"].iloc[-1] - frame["Date"].iloc[0]).days,
                1,
            )
            years = max(calendar_days / 365.25, 1.0 / 365.25)
            annualized_return = float((1.0 + period_return) ** (1.0 / years) - 1.0)
            rolling_drawdown = wealth / wealth.cummax() - 1.0
            maximum_drawdown = float(rolling_drawdown.min())
        else:
            period_return = np.nan
            annualized_return = np.nan
            maximum_drawdown = np.nan

        volatility = (
            float(valid.std(ddof=1) * np.sqrt(252.0))
            if len(valid) > 1
            else np.nan
        )

        rows.append({
            "Symbol": str(symbol),
            "Observation_Count": int(len(frame)),
            "First_Date": frame["Date"].iloc[0].strftime("%Y-%m-%d"),
            "Last_Date": frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "Latest_Close": float(latest["Close"]),
            "Latest_Adjusted_Close": float(latest["Adjusted Close"]),
            "Period_Return": period_return,
            "Annualized_Return": annualized_return,
            "Annualized_Volatility": volatility,
            "Maximum_Drawdown": maximum_drawdown,
            "SMA_20": latest["SMA_20"],
            "SMA_50": latest["SMA_50"],
            "EMA_20": latest["EMA_20"],
            "EMA_50": latest["EMA_50"],
            "RSI_14": latest["RSI_14"],
            "MACD": latest["MACD"],
            "MACD_Signal": latest["MACD_Signal"],
            "MACD_Histogram": latest["MACD_Histogram"],
            "Momentum_20": latest["Momentum_20"],
            "Rolling_Volatility_20": latest["Rolling_Volatility_20"],
            "Trend": classify_trend(latest),
        })

    return pd.DataFrame(rows).sort_values("Symbol").reset_index(drop=True)


def analyze_india_market(
    input_csv: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Read validated India data and save modular indicator and summary outputs."""
    input_path = Path(input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"India validated market data not found: {input_path}")

    frame = pd.read_csv(input_path)
    indicators = calculate_indicators(frame)
    summary = build_stock_summary(indicators)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indicator_path = output_path / "india_market_indicators.csv"
    summary_path = output_path / "india_stock_summary.csv"
    indicators.to_csv(indicator_path, index=False, date_format="%Y-%m-%d")
    summary.to_csv(summary_path, index=False)

    manifest = {
        "input": str(input_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "timezone": "Asia/Kolkata",
        "symbols": summary["Symbol"].tolist(),
        "observation_counts": {
            str(row["Symbol"]): int(row["Observation_Count"])
            for row in summary.to_dict("records")
        },
        "indicator_windows": {
            "sma": [20, 50],
            "ema": [20, 50],
            "rsi": 14,
            "macd": [12, 26, 9],
            "rolling_volatility": 20,
            "momentum": 20,
        },
        "indicators_output": str(indicator_path),
        "summary_output": str(summary_path),
        "missing_value_handling": (
            "No prices or observations are forward-filled. Indicators use "
            "their required history and remain NaN until sufficient."
        ),
        "analysis_only": True,
        "trades_executed": False,
        "ppo_training_executed": False,
    }
    manifest_path = output_path / "india_market_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "symbols": manifest["symbols"],
        "indicator_rows": int(len(indicators)),
        "stock_count": int(len(summary)),
        "indicator_output": str(indicator_path),
        "summary_output": str(summary_path),
        "manifest_output": str(manifest_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze validated Indian market OHLCV data"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(analyze_india_market(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
