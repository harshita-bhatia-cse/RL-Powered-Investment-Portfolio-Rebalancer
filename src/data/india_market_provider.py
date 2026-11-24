"""Configurable daily market-data provider for Indian ETFs."""

from __future__ import annotations

import json
import os
import tempfile
import time
from urllib.parse import urlencode
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import requests
from dotenv import load_dotenv

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
    interval: str = "1d"
    freshness: str = "historical"
    raw_payloads: dict[str, dict[str, Any]] | None = None


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


class AlphaVantageIndiaProvider:
    """Authenticated Alpha Vantage adapter for NSE/BSE daily and intraday data.

    Alpha Vantage is an API-polling provider, not a streaming provider. Intraday
    responses are therefore reported as delayed_or_unverified and must never be
    presented as real-time. Subscription limits and endpoint availability are
    provider-plan dependent.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://www.alphavantage.co/query",
        retry_count: int = 2,
        retry_wait_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        selected_key = (
            api_key
            if api_key is not None
            else os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        )
        selected_key = str(selected_key).strip()
        if not selected_key:
            raise IndiaProviderError(
                "ALPHA_VANTAGE_API_KEY is required for the alpha-vantage provider"
            )
        if isinstance(retry_count, bool) or retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if isinstance(retry_wait_seconds, bool) or retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must be non-negative")
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.api_key = selected_key
        self.base_url = str(base_url).strip()
        self.retry_count = retry_count
        self.retry_wait_seconds = retry_wait_seconds
        self.timeout_seconds = timeout_seconds

    def fetch(self, request: IndiaMarketRequest) -> IndiaMarketProviderResponse:
        raise NotImplementedError(
            "Use fetch_interval(request, interval=...) for Alpha Vantage"
        )

    def fetch_interval(
        self,
        request: IndiaMarketRequest,
        *,
        interval: str = "1d",
    ) -> IndiaMarketProviderResponse:
        interval = str(interval).strip().lower()
        if interval not in ("1d", "1min", "5min", "15min", "30min", "60min"):
            raise ValueError(
                "interval must be 1d, 1min, 5min, 15min, 30min or 60min"
            )

        frames: dict[str, pd.DataFrame] = {}
        action_frames: dict[str, pd.DataFrame] = {}
        raw_payloads: dict[str, dict[str, Any]] = {}
        retrieved_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")

        for market_symbol in request.symbols:
            provider_symbol = self._provider_symbol(market_symbol)
            if interval == "1d":
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": provider_symbol,
                    "outputsize": "full",
                }
                series_name = "Time Series (Daily)"
                freshness = "historical"
            else:
                params = {
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": provider_symbol,
                    "interval": interval,
                    "outputsize": "full",
                }
                series_name = f"Time Series ({interval})"
                freshness = "delayed_or_unverified"

            payload = self._request_payload(params)
            raw_payloads[market_symbol.symbol] = payload
            series = payload.get(series_name)
            if not isinstance(series, dict):
                message = self._provider_error(payload)
                raise IndiaProviderResponseError(
                    f"Alpha Vantage returned no usable data for "
                    f"{market_symbol.symbol}: {message}"
                )

            frame = self._normalize_series(
                series,
                market_symbol=market_symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                intraday=interval != "1d",
            )
            frames[market_symbol.symbol] = frame
            action_frames[market_symbol.symbol] = pd.DataFrame()

        return IndiaMarketProviderResponse(
            provider="Alpha Vantage",
            source="Alpha Vantage authenticated API",
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            retrieved_at=retrieved_at,
            frames=frames,
            action_frames=action_frames,
            interval=interval,
            freshness=freshness,
            raw_payloads=raw_payloads,
        )

    @staticmethod
    def _provider_symbol(market_symbol: IndiaMarketSymbol) -> str:
        symbol = market_symbol.symbol.upper()
        if symbol.endswith(".NS"):
            return symbol[:-3] + ".NSE"
        if symbol.endswith(".BO"):
            return symbol[:-3] + ".BSE"
        if market_symbol.exchange == "BSE":
            return symbol + ".BSE"
        return symbol + ".NSE"

    def _request_payload(self, params: dict[str, str]) -> dict[str, Any]:
        for attempt in range(self.retry_count + 1):
            try:
                response = requests.get(
                    self.base_url,
                    params={**params, "apikey": self.api_key},
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as error:
                if attempt < self.retry_count:
                    time.sleep(self.retry_wait_seconds)
                    continue
                raise IndiaProviderResponseError(
                    f"Alpha Vantage connection failed ({type(error).__name__})"
                ) from error

            if response.status_code == 429:
                if attempt < self.retry_count:
                    time.sleep(self.retry_wait_seconds)
                    continue
                raise IndiaProviderRateLimitError(
                    "Alpha Vantage returned HTTP 429 rate limit"
                )
            if response.status_code in (401, 403):
                raise IndiaProviderResponseError(
                    "Alpha Vantage authentication failed"
                )
            if response.status_code >= 500:
                if attempt < self.retry_count:
                    time.sleep(self.retry_wait_seconds)
                    continue
                raise IndiaProviderResponseError(
                    f"Alpha Vantage server error: HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as error:
                raise IndiaProviderResponseError(
                    "Alpha Vantage returned a non-JSON response"
                ) from error
            if not isinstance(payload, dict):
                raise IndiaProviderResponseError(
                    "Alpha Vantage returned an unexpected JSON response"
                )

            lowered = json.dumps(payload).lower()
            if "invalid api call" in lowered or "invalid apikey" in lowered:
                raise IndiaProviderResponseError(
                    "Alpha Vantage authentication or symbol request failed"
                )
            if "rate limit" in lowered or "frequency" in lowered:
                raise IndiaProviderRateLimitError(
                    "Alpha Vantage rate limit or call-frequency restriction"
                )
            return payload
        raise IndiaProviderResponseError("Alpha Vantage request exhausted retries")

    @staticmethod
    def _provider_error(payload: dict[str, Any]) -> str:
        for key in ("Error Message", "Note", "Information"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:300]
        return "no recognized series or provider error"

    def _normalize_series(
        self,
        series: dict[str, Any],
        *,
        market_symbol: IndiaMarketSymbol,
        start_date: date,
        end_date: date,
        intraday: bool,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for timestamp_text, values in series.items():
            if not isinstance(values, dict):
                continue
            parsed = pd.to_datetime(timestamp_text, errors="coerce")
            if pd.isna(parsed):
                continue
            if parsed.tzinfo is not None:
                parsed = parsed.tz_convert("Asia/Kolkata").tz_localize(None)
            else:
                parsed = parsed.tz_localize("Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT").tz_localize(None)
            if parsed.date() < start_date or parsed.date() > end_date:
                continue
            records.append({
                "_Timestamp": parsed,
                "Open": values.get("1. open"),
                "High": values.get("2. high"),
                "Low": values.get("3. low"),
                "Close": values.get("4. close"),
                "Adjusted Close": values.get("5. adjusted close", values.get("4. close")),
                "Volume": values.get("6. volume"),
            })

        if not records:
            raise IndiaProviderResponseError(
                f"Alpha Vantage returned no records in range for {market_symbol.symbol}"
            )

        frame = pd.DataFrame(records).sort_values("_Timestamp")
        normalized = pd.DataFrame({
            "Date": (
                frame["_Timestamp"].dt.strftime("%Y-%m-%d")
                if not intraday
                else frame["_Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            ),
            "Symbol": market_symbol.symbol,
            "Open": pd.to_numeric(frame["Open"], errors="coerce"),
            "High": pd.to_numeric(frame["High"], errors="coerce"),
            "Low": pd.to_numeric(frame["Low"], errors="coerce"),
            "Close": pd.to_numeric(frame["Close"], errors="coerce"),
            "Adjusted Close": pd.to_numeric(frame["Adjusted Close"], errors="coerce"),
            "Volume": pd.to_numeric(frame["Volume"], errors="coerce"),
        }).reset_index(drop=True)
        normalized = _validate_normalized_india_frame(normalized)
        return normalized


def create_india_market_provider(
    *,
    provider_name: str | None = None,
) -> IndiaDailyMarketProvider:
    """Create an India provider from the configured provider name."""
    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=repository_root / ".env", override=False)
    # provider_name is an explicit override; otherwise read only the documented
    # INDIA_DATA_PROVIDER environment variable.
    name = str(
        provider_name
        if provider_name is not None
        else os.environ.get("INDIA_DATA_PROVIDER", "yahoo-finance-yfinance")
    ).strip().lower()
    if name in ("yahoo-finance-yfinance", "yahoo", "yahoo-finance", "yfinance"):
        return YahooFinanceIndiaProvider()
    if name in ("alpha-vantage", "alphavantage", "alpha_vantage"):
        return AlphaVantageIndiaProvider(
            api_key=os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
            base_url=os.environ.get(
                "INDIA_MARKET_DATA_BASE_URL",
                "https://www.alphavantage.co/query",
            )
        )
    raise ValueError(
        "unsupported India market data provider; use yahoo-finance-yfinance "
        "or alpha-vantage"
    )


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
    interval: str = "1d",
    backfill: bool = False,
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
    if not isinstance(backfill, bool):
        raise ValueError("backfill must be a boolean")
    start = (
        date.fromisoformat(start_date)
        if isinstance(start_date, str)
        else start_date
    )
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    request = IndiaMarketRequest(normalized_symbols, start, end)
    selected_provider = provider or create_india_market_provider()
    if isinstance(selected_provider, AlphaVantageIndiaProvider):
        response = selected_provider.fetch_interval(request, interval=interval)
    else:
        if interval != "1d":
            raise ValueError(
                "The yfinance adapter currently supports daily data only"
            )
        response = selected_provider.fetch(request)

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    validated_dir = output_dir / "validated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_paths: list[Path] = []
    combined_frames: list[pd.DataFrame] = []
    raw_payload_dir = raw_dir / "provider_json"
    raw_payload_dir.mkdir(parents=True, exist_ok=True)
    for market_symbol in normalized_symbols:
        frame = response.frames.get(market_symbol.symbol)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["Timezone"] = "Asia/Kolkata"
        frame["Provider"] = response.provider
        frame["Source"] = response.source
        frame["Retrieved At"] = response.retrieved_at
        frame["Interval"] = response.interval
        frame["Data Freshness"] = response.freshness
        if "Source Timestamp" not in frame.columns:
            frame["Source Timestamp"] = frame["Date"]
        combined_frames.append(frame)
        filename_symbol = market_symbol.symbol.replace(".", "_")
        suffix = "daily" if interval == "1d" else interval
        raw_path = raw_dir / f"{filename_symbol}_{suffix}.csv"
        if raw_path.exists():
            existing = pd.read_csv(raw_path)
            merged = pd.concat([existing, frame], ignore_index=True)
            merged.drop_duplicates(subset=["Symbol", "Date"], keep="last", inplace=True)
            merged.sort_values(["Symbol", "Date"], inplace=True)
            merged.to_csv(raw_path, index=False, encoding="utf-8")
        else:
            frame.to_csv(raw_path, index=False, encoding="utf-8")
        raw_csv_paths.append(raw_path)
        payload = (
            response.raw_payloads or {}
        ).get(market_symbol.symbol)
        if payload is not None:
            payload_path = raw_payload_dir / f"{filename_symbol}_{suffix}.json"
            payload_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined.sort_values(["Symbol", "Date"], inplace=True)
    else:
        combined = pd.DataFrame(columns=[
            "Date",
            "Symbol",
            "Open",
            "High",
            "Low",
            "Close",
            "Adjusted Close",
            "Volume",
        ])
    provider_csv = raw_dir / (
        "india_etf_provider_combined.csv"
        if interval == "1d"
        else f"india_etf_provider_{interval}.csv"
    )
    if provider_csv.exists():
        existing = pd.read_csv(provider_csv)
        combined = pd.concat([existing, combined], ignore_index=True)
        combined.drop_duplicates(subset=["Symbol", "Date"], keep="last", inplace=True)
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
    if interval == "1d":
        config = IndianETFIngestionConfig(
            symbols=tuple(symbol.symbol for symbol in normalized_symbols),
            provider=provider_name,
            retrieval_timestamp=response.retrieved_at,
            adjustment_methodology=(
                "Alpha Vantage adjusted close; raw OHLC and corporate-action "
                "completeness unverified"
                if isinstance(selected_provider, AlphaVantageIndiaProvider)
                else INDIA_ADJUSTMENT_METHOD
            ),
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
    else:
        live_dir = output_dir / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        live_csv = live_dir / f"india_etf_{interval}.csv"
        combined.to_csv(live_csv, index=False, encoding="utf-8")
        report = {
            "provider": provider_name,
            "source": response.source,
            "symbols": [symbol.symbol for symbol in normalized_symbols],
            "validation_issues": [],
            "training_ready": False,
            "normalized_csv": str(live_csv),
            "row_count": int(len(combined)),
        }
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
    report["interval"] = response.interval
    report["timezone"] = "Asia/Kolkata"
    report["data_freshness"] = response.freshness
    report["adjustment_methodology"] = (
        "Alpha Vantage adjusted close; raw OHLC and corporate-action "
        "completeness unverified"
        if isinstance(selected_provider, AlphaVantageIndiaProvider)
        else INDIA_ADJUSTMENT_METHOD
    )
    report["license_permission_status"] = status
    report["training_ready"] = bool(
        status == "approved"
        and report.get("training_ready") is True
    )
    report["raw_csv_paths"] = [str(path) for path in raw_csv_paths]
    report["provider_combined_csv"] = str(provider_csv)
    per_symbol_status: dict[str, dict[str, object]] = {}
    for market_symbol in normalized_symbols:
        frame = response.frames.get(market_symbol.symbol)
        if frame is None or frame.empty:
            per_symbol_status[market_symbol.symbol] = {
                "status": "failed",
                "requested_start_date": start.isoformat(),
                "requested_end_date": end.isoformat(),
                "actual_start_date": None,
                "actual_end_date": None,
                "row_count": 0,
                "failure": "provider returned no data",
            }
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        per_symbol_status[market_symbol.symbol] = {
            "status": (
                "complete" if len(dates) > 1 else "one_row_or_incomplete"
            ),
            "requested_start_date": start.isoformat(),
            "requested_end_date": end.isoformat(),
            "actual_start_date": (
                dates.min().strftime("%Y-%m-%d")
                if not dates.empty
                else None
            ),
            "actual_end_date": (
                dates.max().strftime("%Y-%m-%d")
                if not dates.empty
                else None
            ),
            "row_count": int(len(frame)),
            "failure": None,
        }
    report["feed_mode"] = "backfill" if backfill else "incremental"
    report["backfill"] = bool(backfill)
    report["per_symbol_status"] = per_symbol_status
    report["failed_symbols"] = sorted(
        symbol
        for symbol, status in per_symbol_status.items()
        if status["status"] == "failed"
    )
    report["one_row_symbols"] = sorted(
        symbol
        for symbol, status in per_symbol_status.items()
        if status["status"] == "one_row_or_incomplete"
    )
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
