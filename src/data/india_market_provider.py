"""Configurable daily market-data provider for Indian ETFs."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from src.data.indian_etf_csv import (
    IndianETFIngestionConfig,
    ingest_indian_etf_csv,
)


INDIA_DATA_DIR = Path("data") / "india_daily"
INDIA_PROVIDER_NAME = "Yahoo Finance via yfinance"
INDIA_ADJUSTMENT_METHOD = (
    "Yahoo/yfinance Adjusted Close; corporate-action completeness unverified"
)


class IndiaProviderError(RuntimeError):
    """Base error for Indian market-data provider operations."""


class IndiaProviderResponseError(IndiaProviderError):
    """The provider returned no usable or malformed completed data."""


class IndiaProviderRateLimitError(IndiaProviderError):
    """The provider rate-limited or throttled a request."""


@dataclass(frozen=True)
class IndiaMarketSymbol:
    symbol: str
    exchange: str = "NSE"

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        exchange = str(self.exchange).strip().upper()
        if not symbol:
            raise ValueError("India ETF symbol is required")
        if exchange not in ("NSE", "BSE"):
            raise ValueError("exchange must be NSE or BSE")
        suffix = ".NS" if exchange == "NSE" else ".BO"
        if not symbol.endswith((".NS", ".BO")):
            symbol = f"{symbol}{suffix}"
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "exchange", exchange)


@dataclass(frozen=True)
class IndiaMarketRequest:
    symbols: tuple[IndiaMarketSymbol, ...]
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if isinstance(self.start_date, str):
            object.__setattr__(self, "start_date", date.fromisoformat(self.start_date))
        if isinstance(self.end_date, str):
            object.__setattr__(self, "end_date", date.fromisoformat(self.end_date))
        if not self.symbols:
            raise ValueError("at least one India market symbol is required")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")


@dataclass(frozen=True)
class IndiaMarketProviderResponse:
    provider: str
    source: str
    symbols: tuple[IndiaMarketSymbol, ...]
    start_date: date
    end_date: date
    retrieved_at: str
    frames: dict[str, pd.DataFrame]
    action_frames: dict[str, pd.DataFrame]


class IndiaDailyMarketProvider(Protocol):
    """Provider interface that returns normalized Phase 1 daily OHLCV rows."""

    def fetch(self, request: IndiaMarketRequest) -> IndiaMarketProviderResponse:
        ...


class YahooFinanceIndiaProvider:
    """Yahoo/yfinance adapter for completed NSE/BSE ETF daily sessions.

    Yahoo/yfinance support is suitable for a fetch path but is not, by itself,
    evidence of permission to use the data for model training.
    """

    def __init__(
        self,
        *,
        ticker_factory=None,
        retry_count: int = 2,
        retry_wait_seconds: float = 0.5,
    ) -> None:
        if ticker_factory is None:
            try:
                import yfinance
            except ImportError as error:
                raise IndiaProviderError(
                    "yfinance is required; it is a project-declared dependency"
                ) from error
            ticker_factory = yfinance.Ticker
        if isinstance(retry_count, bool) or retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if isinstance(retry_wait_seconds, bool) or retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must be non-negative")
        self._ticker_factory = ticker_factory
        self._retry_count = retry_count
        self._retry_wait_seconds = retry_wait_seconds

    def fetch(self, request: IndiaMarketRequest) -> IndiaMarketProviderResponse:
        frames: dict[str, pd.DataFrame] = {}
        action_frames: dict[str, pd.DataFrame] = {}
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        india_timezone = timezone(timedelta(hours=5, minutes=30))
        today_india = datetime.now(india_timezone).date()

        for market_symbol in request.symbols:
            last_error: Exception | None = None
            frame: pd.DataFrame | None = None
            for attempt in range(self._retry_count + 1):
                try:
                    ticker = self._ticker_factory(market_symbol.symbol)
                    frame = ticker.history(
                        start=request.start_date.isoformat(),
                        end=(request.end_date + timedelta(days=1)).isoformat(),
                        actions=True,
                        auto_adjust=False,
                        back_adjust=False,
                        repair=False,
                        keepna=False,
                    )
                    break
                except Exception as error:
                    last_error = error
                    text = str(error).lower()
                    if "rate" in text or "429" in text:
                        last_error = IndiaProviderRateLimitError(
                            f"provider rate limit for {market_symbol.symbol}: {error}"
                        )
                    if attempt < self._retry_count:
                        time.sleep(self._retry_wait_seconds)
            if frame is None:
                if isinstance(last_error, IndiaProviderError):
                    raise last_error
                raise IndiaProviderResponseError(
                    f"provider request failed for {market_symbol.symbol}: {last_error}"
                )
            normalized = self._normalize_frame(
                frame,
                market_symbol=market_symbol,
                end_date=request.end_date,
                today_india=today_india,
            )
            frames[market_symbol.symbol] = normalized
            action_frames[market_symbol.symbol] = self._action_frame(frame)

        return IndiaMarketProviderResponse(
            provider=INDIA_PROVIDER_NAME,
            source="Yahoo Chart History API through yfinance",
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            retrieved_at=retrieved_at,
            frames=frames,
            action_frames=action_frames,
        )

    def _normalize_frame(
        self,
        frame: pd.DataFrame,
        *,
        market_symbol: IndiaMarketSymbol,
        end_date: date,
        today_india: date,
    ) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise IndiaProviderResponseError(
                f"provider returned no completed history for {market_symbol.symbol}"
            )
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame.index = frame.index.tz_convert("Asia/Kolkata").tz_localize(None)
        frame.index = pd.to_datetime(frame.index, errors="coerce").normalize()
        if frame.index.isna().any():
            raise IndiaProviderResponseError(
                f"provider returned invalid dates for {market_symbol.symbol}"
            )
        if frame.index.duplicated(keep=False).any():
            raise IndiaProviderResponseError(
                f"provider returned duplicate dates for {market_symbol.symbol}"
            )

        required = ("Open", "High", "Low", "Close", "Volume")
        adjusted_name = "Adj Close"
        missing = [column for column in (*required, adjusted_name) if column not in frame.columns]
        if missing:
            raise IndiaProviderResponseError(
                f"provider omitted required columns for {market_symbol.symbol}: "
                + ", ".join(missing)
            )

        frame = frame.loc[
            frame.index <= pd.Timestamp(end_date)
        ]
        completed_mask = np.asarray([
            timestamp.date() != today_india
            for timestamp in frame.index
        ], dtype=bool)
        frame = frame.loc[completed_mask]
        if frame.empty:
            raise IndiaProviderResponseError(
                f"provider returned no completed sessions for {market_symbol.symbol}"
            )
        frame = frame.sort_index()

        normalized = pd.DataFrame({
            "Date": frame.index.strftime("%Y-%m-%d"),
            "Symbol": market_symbol.symbol,
            "Open": pd.to_numeric(frame["Open"], errors="coerce"),
            "High": pd.to_numeric(frame["High"], errors="coerce"),
            "Low": pd.to_numeric(frame["Low"], errors="coerce"),
            "Close": pd.to_numeric(frame["Close"], errors="coerce"),
            "Adjusted Close": pd.to_numeric(frame["Adj Close"], errors="coerce"),
            "Volume": pd.to_numeric(frame["Volume"], errors="coerce"),
        }).reset_index(drop=True)
        normalized = _validate_normalized_india_frame(normalized)
        return normalized

    def _action_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        available = [column for column in ("Dividends", "Stock Splits") if column in frame.columns]
        if not available:
            return pd.DataFrame()
        actions = frame.loc[:, available].copy()
        actions.index = pd.to_datetime(actions.index).normalize()
        actions = actions.fillna(0)
        actions = actions.loc[
            (actions["Dividends"].fillna(0) != 0)
            if "Dividends" in actions.columns
            else pd.Series(True, index=actions.index)
        ] if "Dividends" in actions.columns else actions
        return actions


def _validate_normalized_india_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise IndiaProviderResponseError("provider returned an empty normalized frame")
    required = (
        "Date",
        "Symbol",
        "Open",
        "High",
        "Low",
        "Close",
        "Adjusted Close",
        "Volume",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise IndiaProviderResponseError(
            "normalized frame missing columns: " + ", ".join(missing)
        )
    if frame.duplicated(subset=["Symbol", "Date"], keep=False).any():
        raise IndiaProviderResponseError("provider returned duplicate symbol-date rows")

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any():
        raise IndiaProviderResponseError("provider returned invalid dates")
    if not dates.is_monotonic_increasing:
        raise IndiaProviderResponseError("provider returned non-chronological dates")

    for column in ("Open", "High", "Low", "Close", "Adjusted Close"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise IndiaProviderResponseError(
                f"provider returned invalid positive-price values in {column}"
            )
    volume = frame["Volume"].to_numpy(dtype=np.float64)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise IndiaProviderResponseError(
            "provider returned negative or non-finite volume"
        )

    high = frame["High"].to_numpy(dtype=np.float64)
    low = frame["Low"].to_numpy(dtype=np.float64)
    open_values = frame["Open"].to_numpy(dtype=np.float64)
    close = frame["Close"].to_numpy(dtype=np.float64)
    if np.any(high < open_values) or np.any(high < low) or np.any(high < close):
        raise IndiaProviderResponseError("provider returned inconsistent OHLC: high")
    if np.any(low > open_values) or np.any(low > high) or np.any(low > close):
        raise IndiaProviderResponseError("provider returned inconsistent OHLC: low")
    return frame


def download_india_etf_data(
    *,
    symbols: Sequence[IndiaMarketSymbol | str],
    start_date: date | str,
    end_date: date | str,
    provider: IndiaDailyMarketProvider | None = None,
    output_dir: str | Path = INDIA_DATA_DIR,
    license_status: str | None = None,
    retrieval_timezone: str = "UTC",
) -> dict[str, object]:
    """Download, preserve and ingest India ETF CSV data without training readiness assumptions."""
    del retrieval_timezone
    normalized_symbols: tuple[IndiaMarketSymbol, ...] = tuple(
        symbol
        if isinstance(symbol, IndiaMarketSymbol)
        else IndiaMarketSymbol(symbol=str(symbol))
        for symbol in symbols
    )
    if not normalized_symbols:
        raise ValueError("at least one India symbol is required")
    start = (
        date.fromisoformat(start_date)
        if isinstance(start_date, str)
        else start_date
    )
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    request = IndiaMarketRequest(normalized_symbols, start, end)
    selected_provider = provider or YahooFinanceIndiaProvider()
    response = selected_provider.fetch(request)

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    validated_dir = output_dir / "validated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_paths: list[Path] = []
    combined_frames: list[pd.DataFrame] = []
    for market_symbol in normalized_symbols:
        frame = response.frames[market_symbol.symbol]
        combined_frames.append(frame)
        filename_symbol = market_symbol.symbol.replace(".", "_")
        raw_path = raw_dir / f"{filename_symbol}_daily.csv"
        frame.to_csv(raw_path, index=False, encoding="utf-8")
        raw_csv_paths.append(raw_path)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined.sort_values(["Symbol", "Date"], inplace=True)
    provider_csv = raw_dir / "india_etf_provider_combined.csv"
    combined.to_csv(provider_csv, index=False, encoding="utf-8")

    status = (
        license_status
        if license_status is not None
        else os.environ.get("INDIA_DATA_LICENSE_STATUS", "unverified")
    )
    status = str(status).strip().lower()
    if status not in ("approved", "unverified"):
        raise ValueError("license_status must be approved or unverified")

    provider_name = str(response.provider).strip()
    config = IndianETFIngestionConfig(
        symbols=tuple(symbol.symbol for symbol in normalized_symbols),
        provider=provider_name,
        retrieval_timestamp=response.retrieved_at,
        adjustment_methodology=INDIA_ADJUSTMENT_METHOD,
        license_permission_status=status,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        ingestion_source = Path(temp_dir) / provider_csv.name
        ingestion_source.write_bytes(provider_csv.read_bytes())
        report = ingest_indian_etf_csv(
            ingestion_source,
            config,
            output_dir=output_dir,
        )
    report["source"] = response.source
    report["exchange"] = {
        symbol.symbol: symbol.exchange for symbol in normalized_symbols
    }
    report["historical_date_range"] = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    report["provider"] = provider_name
    report["retrieval_timestamp"] = response.retrieved_at
    report["adjustment_methodology"] = INDIA_ADJUSTMENT_METHOD
    report["license_permission_status"] = status
    report["training_ready"] = bool(
        status == "approved"
        and report.get("training_ready") is True
    )
    report["raw_csv_paths"] = [str(path) for path in raw_csv_paths]
    report["provider_combined_csv"] = str(provider_csv)
    report["corporate_actions"] = {
        symbol.symbol: {
            "verified": False,
            "coverage": "not_verified",
            "source": "provider_action_fields_only",
        }
        for symbol in normalized_symbols
    }

    actions_path = output_dir / "raw" / "india_etf_actions.csv"
    action_frames = [
        frame.assign(Symbol=symbol)
        for symbol, frame in response.action_frames.items()
        if not frame.empty
    ]
    if action_frames:
        actions = pd.concat(action_frames)
        actions.reset_index(inplace=True)
        actions.to_csv(actions_path, index=False, encoding="utf-8")
        report["actions_csv"] = str(actions_path)
    else:
        report["actions_csv"] = None

    manifest_path = output_dir / "india_market_data_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
