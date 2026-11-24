import os
from datetime import date
import unittest
from unittest.mock import patch

from src.data.india_market_feed import (
    _symbols_from_environment,
    update_indian_market_feed,
)
from src.data.india_market_provider import download_india_etf_data


class IndiaMarketFeedUniverseTests(unittest.TestCase):
    def test_default_two_stock_feed_is_preserved(self):
        with patch.dict(os.environ, {
            "INDIA_USE_STOCK_UNIVERSE": "false",
            "INDIA_DATA_SYMBOLS": "RELIANCE.NS,TCS.NS",
        }, clear=False):
            symbols = _symbols_from_environment()
        self.assertEqual(
            tuple(symbol.symbol for symbol in symbols),
            ("RELIANCE.NS", "TCS.NS"),
        )

    def test_stock_universe_automobile_feed_has_new_securities_only(self):
        with patch.dict(os.environ, {
            "INDIA_USE_STOCK_UNIVERSE": "true",
            "INDIA_STOCK_UNIVERSE_SECTOR": "Automobile",
        }, clear=False):
            symbols = _symbols_from_environment()
        values = tuple(symbol.symbol for symbol in symbols)
        self.assertIn("TMPV.NS", values)
        self.assertIn("TMCV.NS", values)
        self.assertNotIn("TATAMOTORS.NS", values)

    def test_backfill_requires_full_configured_range_for_every_symbol(self):
        with patch(
            "src.data.india_market_feed.download_india_etf_data",
            return_value={
                "updated": True,
                "per_symbol_status": {},
            },
        ) as downloader:
            update_indian_market_feed(
                symbols=("RELIANCE.NS", "TCS.NS"),
                start_date="2024-01-01",
                end_date="2024-12-31",
                backfill=True,
                output_dir="/tmp/india-feed-test",
            )

        kwargs = downloader.call_args.kwargs
        self.assertEqual(kwargs["start_date"], date(2024, 1, 1))
        self.assertEqual(kwargs["end_date"], date(2024, 12, 31))
        self.assertTrue(kwargs["backfill"])
        self.assertEqual(len(kwargs["symbols"]), 2)

    def test_incremental_mode_remains_default(self):
        with patch(
            "src.data.india_market_feed.download_india_etf_data",
            return_value={
                "updated": True,
                "per_symbol_status": {},
            },
        ) as downloader:
            update_indian_market_feed(
                symbols=("RELIANCE.NS",),
                start_date="2026-11-23",
                end_date="2026-11-24",
                output_dir="/tmp/india-feed-test",
            )

        kwargs = downloader.call_args.kwargs
        self.assertFalse(kwargs["backfill"])

    def test_backfill_requires_configured_start_when_not_passed(self):
        with patch.dict(os.environ, {
            "INDIA_DATA_START_DATE": "2024-01-01",
            "INDIA_DATA_END_DATE": "2024-12-31",
            "INDIA_USE_STOCK_UNIVERSE": "false",
            "INDIA_DATA_SYMBOLS": "RELIANCE.NS",
        }, clear=False), patch(
            "src.data.india_market_feed.download_india_etf_data",
            return_value={"updated": True},
        ) as downloader:
            update_indian_market_feed(
                backfill=True,
                output_dir="/tmp/india-feed-test",
            )

        kwargs = downloader.call_args.kwargs
        self.assertEqual(kwargs["start_date"], date(2024, 1, 1))
        self.assertEqual(kwargs["end_date"], date(2024, 12, 31))


if __name__ == "__main__":
    unittest.main()
