import unittest
import tempfile
from pathlib import Path
from datetime import date
from unittest.mock import Mock

import pandas as pd

from src.data.india_market_provider import (
    IndiaMarketProviderResponse,
    IndiaMarketRequest,
    IndiaMarketSymbol,
    IndiaProviderResponseError,
    YahooFinanceIndiaProvider,
    download_india_etf_data,
)


def mock_history_frame(start="2024-01-02", volume=100000):
    dates = pd.bdate_range(start, periods=3)
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [101.0, 102.0, 103.0],
        "Adj Close": [101.0, 102.0, 103.0],
        "Volume": [volume, volume, volume],
        "Dividends": [0.0, 0.0, 0.0],
        "Stock Splits": [0.0, 0.0, 0.0],
    }, index=dates)


class YahooFinanceIndiaProviderTests(unittest.TestCase):
    def test_normalizes_nse_symbol_and_completed_history(self):
        ticker = Mock()
        ticker.history.return_value = mock_history_frame()
        provider = YahooFinanceIndiaProvider(ticker_factory=lambda _: ticker)
        request = IndiaMarketRequest(
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )

        response = provider.fetch(request)

        frame = response.frames["NIFTYBEES.NS"]
        self.assertEqual(frame["Symbol"].iloc[0], "NIFTYBEES.NS")
        self.assertEqual(
            list(frame.columns),
            ["Date", "Symbol", "Open", "High", "Low", "Close", "Adjusted Close", "Volume"],
        )
        self.assertTrue(frame["Date"].is_monotonic_increasing)

    def test_duplicate_dates_are_rejected(self):
        frame = mock_history_frame()
        frame = pd.concat([frame, frame.iloc[[0]]])
        ticker = Mock()
        ticker.history.return_value = frame
        provider = YahooFinanceIndiaProvider(ticker_factory=lambda _: ticker)
        request = IndiaMarketRequest(
            symbols=(IndiaMarketSymbol("BANKBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )
        with self.assertRaises(IndiaProviderResponseError):
            provider.fetch(request)

    def test_invalid_ohlc_is_rejected(self):
        frame = mock_history_frame()
        frame.loc[frame.index[0], "High"] = 90.0
        ticker = Mock()
        ticker.history.return_value = frame
        provider = YahooFinanceIndiaProvider(ticker_factory=lambda _: ticker)
        request = IndiaMarketRequest(
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )
        with self.assertRaises(IndiaProviderResponseError):
            provider.fetch(request)

    def test_negative_volume_is_rejected(self):
        frame = mock_history_frame(volume=-1)
        ticker = Mock()
        ticker.history.return_value = frame
        provider = YahooFinanceIndiaProvider(ticker_factory=lambda _: ticker)
        request = IndiaMarketRequest(
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )
        with self.assertRaises(IndiaProviderResponseError):
            provider.fetch(request)

    def test_empty_provider_response_is_rejected(self):
        ticker = Mock()
        ticker.history.return_value = pd.DataFrame()
        provider = YahooFinanceIndiaProvider(ticker_factory=lambda _: ticker)
        request = IndiaMarketRequest(
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )
        with self.assertRaises(IndiaProviderResponseError):
            provider.fetch(request)

    def test_missing_trading_session_is_not_filled(self):
        frame = mock_history_frame()
        frame = frame.iloc[[0, 2]]
        ticker = Mock()
        ticker.history.return_value = frame
        provider = YahooFinanceIndiaProvider(ticker_factory=lambda _: ticker)
        request = IndiaMarketRequest(
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        )
        response = provider.fetch(request)
        self.assertEqual(len(response.frames["NIFTYBEES.NS"]), 2)

    def test_download_preserves_raw_data_and_stays_not_training_ready_by_default(self):
        frame = mock_history_frame()
        normalized = pd.DataFrame({
            "Date": frame.index.strftime("%Y-%m-%d"),
            "Symbol": "NIFTYBEES.NS",
            "Open": frame["Open"],
            "High": frame["High"],
            "Low": frame["Low"],
            "Close": frame["Close"],
            "Adjusted Close": frame["Adj Close"],
            "Volume": frame["Volume"],
        }).reset_index(drop=True)
        response = IndiaMarketProviderResponse(
            provider="Yahoo Finance via yfinance",
            source="mock provider",
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            retrieved_at="2024-01-05T00:00:00Z",
            frames={"NIFTYBEES.NS": normalized},
            action_frames={"NIFTYBEES.NS": pd.DataFrame()},
        )

        class MockProvider:
            def fetch(self, request):
                return response

        with tempfile.TemporaryDirectory() as temp_dir:
            report = download_india_etf_data(
                symbols=("NIFTYBEES",),
                start_date="2024-01-02",
                end_date="2024-01-05",
                provider=MockProvider(),
                output_dir=temp_dir,
            )
            output = Path(temp_dir)
            self.assertTrue((output / "raw" / "NIFTYBEES_NS_daily.csv").exists())
            self.assertTrue((output / "validated" / "indian_etf_validated.csv").exists())
            self.assertTrue((output / "india_market_data_manifest.json").exists())
            self.assertFalse(report["training_ready"])

    def test_backfill_manifest_records_per_symbol_request_and_result(self):
        frame = mock_history_frame()
        normalized = pd.DataFrame({
            "Date": frame.index.strftime("%Y-%m-%d"),
            "Symbol": "NIFTYBEES.NS",
            "Open": frame["Open"],
            "High": frame["High"],
            "Low": frame["Low"],
            "Close": frame["Close"],
            "Adjusted Close": frame["Adj Close"],
            "Volume": frame["Volume"],
        }).reset_index(drop=True)
        response = IndiaMarketProviderResponse(
            provider="Yahoo Finance via yfinance",
            source="mock provider",
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            retrieved_at="2024-01-05T00:00:00Z",
            frames={"NIFTYBEES.NS": normalized},
            action_frames={"NIFTYBEES.NS": pd.DataFrame()},
        )

        class MockProvider:
            def fetch(self, request):
                return response

        with tempfile.TemporaryDirectory() as temp_dir:
            report = download_india_etf_data(
                symbols=("NIFTYBEES",),
                start_date="2024-01-02",
                end_date="2024-01-05",
                provider=MockProvider(),
                output_dir=temp_dir,
                backfill=True,
            )
            status = report["per_symbol_status"]["NIFTYBEES.NS"]
            self.assertTrue(report["backfill"])
            self.assertEqual(report["feed_mode"], "backfill")
            self.assertEqual(status["requested_start_date"], "2024-01-02")
            self.assertEqual(status["requested_end_date"], "2024-01-05")
            self.assertEqual(status["row_count"], 3)
            self.assertEqual(status["actual_start_date"], "2024-01-02")
            self.assertEqual(status["actual_end_date"], "2024-01-04")
            self.assertNotIn("NIFTYBEES.NS", report["one_row_symbols"])

    def test_failed_provider_symbol_is_recorded_not_hidden(self):
        response = IndiaMarketProviderResponse(
            provider="Yahoo Finance via yfinance",
            source="mock provider",
            symbols=(IndiaMarketSymbol("NIFTYBEES", "NSE"),),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
            retrieved_at="2024-01-05T00:00:00Z",
            frames={},
            action_frames={},
        )

        class MockProvider:
            def fetch(self, request):
                return response

        with tempfile.TemporaryDirectory() as temp_dir:
            report = download_india_etf_data(
                symbols=("NIFTYBEES",),
                start_date="2024-01-02",
                end_date="2024-01-05",
                provider=MockProvider(),
                output_dir=temp_dir,
                backfill=True,
            )
            status = report["per_symbol_status"]["NIFTYBEES.NS"]
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["row_count"], 0)
            self.assertIn("NIFTYBEES.NS", report["failed_symbols"])

    def test_merge_preserves_existing_history_and_deduplicates_dates(self):
        old = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Symbol": ["NIFTYBEES.NS", "NIFTYBEES.NS"],
            "Open": [99.0, 100.0],
            "High": [100.0, 101.0],
            "Low": [98.0, 99.0],
            "Close": [99.5, 100.5],
            "Adjusted Close": [99.5, 100.5],
            "Volume": [900, 1000],
        })
        new = pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-03"],
            "Symbol": ["NIFTYBEES.NS", "NIFTYBEES.NS"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adjusted Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        })

        merged = pd.concat([old, new], ignore_index=True)
        merged.drop_duplicates(subset=["Symbol", "Date"], keep="last", inplace=True)

        self.assertEqual(len(merged), 3)
        self.assertEqual(
            merged.loc[
                (merged["Symbol"] == "NIFTYBEES.NS")
                & (merged["Date"] == "2024-01-02"),
                "Close",
            ].iloc[0],
            101.0,
        )


if __name__ == "__main__":
    unittest.main()
