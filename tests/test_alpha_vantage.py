"""Isolated, one-request Alpha Vantage SPY endpoint verification.

Run directly with ``python tests/test_alpha_vantage.py``. The API key and
credential-bearing request URL are never printed or included in results.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPOSITORY_ROOT / ".env"
API_URL = "https://www.alphavantage.co/query"
TIME_SERIES_KEY = "Time Series (Daily)"
PRICE_FIELDS = (
    "1. open",
    "2. high",
    "3. low",
    "4. close",
    "5. adjusted close",
    "6. volume",
)


def _safe_message(value: Any, api_key: str) -> str:
    """Return a short API message with any accidental credential redacted."""
    message = " ".join(str(value).split())
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message[:500]


def fetch_spy_summary() -> dict[str, Any]:
    """Make one compact-output request and return only non-sensitive metadata."""
    if not ENV_FILE.is_file():
        return {
            "test_date": date.today().isoformat(),
            "api_access": "NOT TESTED",
            "detail": ".env file is missing.",
        }

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        return {
            "test_date": date.today().isoformat(),
            "api_access": "NOT TESTED",
            "detail": "ALPHA_VANTAGE_API_KEY is missing or empty in .env.",
        }

    try:
        response = requests.get(
            API_URL,
            params={
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": "SPY",
                "outputsize": "compact",
                "apikey": api_key,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        return {
            "test_date": date.today().isoformat(),
            "http_status": None,
            "api_access": "REQUEST FAILED",
            "detail": f"Network/client error ({type(exc).__name__}); message omitted.",
        }

    status: dict[str, Any] = {
        "test_date": date.today().isoformat(),
        "http_status": response.status_code,
        "api_access": "UNKNOWN",
    }

    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        status["api_access"] = "NON-JSON RESPONSE"
        status["detail"] = "Response body was not valid JSON; body omitted."
        return status

    if not isinstance(payload, dict):
        status["api_access"] = "UNEXPECTED RESPONSE"
        status["detail"] = "JSON response was not an object; body omitted."
        return status

    message = next(
        (
            payload[key]
            for key in ("Error Message", "Note", "Information")
            if isinstance(payload.get(key), str)
        ),
        None,
    )
    if message is not None:
        normalized = _safe_message(message, api_key)
        lowered = normalized.lower()
        if any(term in lowered for term in ("premium", "subscription", "premium endpoint")):
            status["api_access"] = "PREMIUM RESTRICTION"
        elif any(term in lowered for term in ("rate limit", "call frequency", "requests per day")):
            status["api_access"] = "RATE LIMITED"
        else:
            status["api_access"] = "API ERROR"
        status["detail"] = normalized
        return status

    series = payload.get(TIME_SERIES_KEY)
    if not isinstance(series, dict):
        status["api_access"] = "NO DAILY SERIES"
        status["detail"] = "Response had no daily series or recognized API message."
        status["returned_ticker"] = payload.get("Meta Data", {}).get(
            "2. Symbol", "not reported"
        )
        return status

    dates = sorted(key for key in series if isinstance(key, str))
    sample = series[dates[0]] if dates else {}
    available_fields = (
        [field for field in PRICE_FIELDS if field in sample]
        if isinstance(sample, dict)
        else []
    )
    status.update(
        {
            "api_access": "ACCESSIBLE",
            "returned_ticker": payload.get("Meta Data", {}).get(
                "2. Symbol", "not reported"
            ),
            "record_count": len(dates),
            "latest_date": dates[-1] if dates else None,
            "earliest_date": dates[0] if dates else None,
            "available_fields": available_fields,
            "adjusted_ohlc_available": all(
                field in available_fields
                for field in ("1. open", "2. high", "3. low", "5. adjusted close")
            ),
            "detail": (
                "TIME_SERIES_DAILY_ADJUSTED returned records. "
                "The endpoint's available fields are listed; adjusted close alone "
                "does not establish adjusted open/high/low."
            ),
        }
    )
    return status


def test_alpha_vantage_spy_endpoint() -> None:
    """Exercise the endpoint once and print only a sanitized summary."""
    summary = fetch_spy_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    test_alpha_vantage_spy_endpoint()
