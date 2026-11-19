
import unittest
from unittest.mock import Mock

import pandas as pd

from src.data.data_fetcher import FMPFetcher


class OfflinePriceFetcherTests(unittest.TestCase):

    def setUp(self):
        # Bypass constructor to avoid configuration, database and API setup.
        self.fetcher = FMPFetcher.__new__(FMPFetcher)
        self.fetcher.offline_mode = True
        self.fetcher.data_store = Mock()

        self.tickers = pd.DataFrame({"tickers": ["GLD"]})
        self.start_date = "2017-01-01"
        self.end_date = "2025-12-31"

        self.existing = pd.DataFrame({
            "tic": ["GLD"],
            "datadate": ["2024-01-02"],
            "adj_close": [180.0],
        })

    def test_fetches_completely_missing_ticker(self):
        self.fetcher.data_store.get_price_data.return_value = pd.DataFrame()

        self.fetcher.data_store.get_missing_price_dates_bulk.return_value = {
            "GLD": [(self.start_date, self.end_date)]
        }

        fetched = pd.DataFrame({"tic": ["GLD"]})
        self.fetcher._fetch_yahoo_price_data = Mock(return_value=fetched)

        self.fetcher.get_price_data(
            self.tickers, self.start_date, self.end_date
        )

        self.fetcher._fetch_yahoo_price_data.assert_called_once_with(
            ["GLD"], self.start_date, self.end_date
        )

        self.fetcher.data_store.save_price_data.assert_called_once_with(
            fetched
        )

    def test_fetches_partially_covered_ticker(self):
        self.fetcher.data_store.get_price_data.return_value = self.existing

        self.fetcher.data_store.get_missing_price_dates_bulk.return_value = {
            "GLD": [(self.start_date, "2023-12-31")]
        }

        fetched = pd.DataFrame({"tic": ["GLD"]})
        self.fetcher._fetch_yahoo_price_data = Mock(return_value=fetched)

        self.fetcher.get_price_data(
            self.tickers, self.start_date, self.end_date
        )

        self.fetcher.data_store.get_missing_price_dates_bulk.assert_called_once()

        self.fetcher._fetch_yahoo_price_data.assert_called_once_with(
            ["GLD"], self.start_date, self.end_date
        )

    def test_does_not_fetch_when_ticker_is_fully_covered(self):
        self.fetcher.data_store.get_price_data.return_value = self.existing

        self.fetcher.data_store.get_missing_price_dates_bulk.return_value = {
            "GLD": []
        }

        self.fetcher._fetch_yahoo_price_data = Mock(
            return_value=pd.DataFrame()
        )

        self.fetcher.get_price_data(
            self.tickers, self.start_date, self.end_date
        )

        self.fetcher.data_store.get_missing_price_dates_bulk.assert_called_once()

        self.fetcher._fetch_yahoo_price_data.assert_not_called()

        self.fetcher.data_store.save_price_data.assert_not_called()

    def test_returns_existing_data_when_fetch_returns_empty(self):
        self.fetcher.data_store.get_price_data.return_value = self.existing

        self.fetcher._fetch_yahoo_price_data = Mock(
            return_value=pd.DataFrame()
        )

        result = self.fetcher.get_price_data(
            self.tickers, self.start_date, self.end_date
        )

        pd.testing.assert_frame_equal(result, self.existing)


if __name__ == "__main__":
    unittest.main()
