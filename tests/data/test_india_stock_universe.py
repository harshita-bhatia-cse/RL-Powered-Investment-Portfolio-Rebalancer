import unittest

from src.data.india_stock_universe import (
    INDIA_STOCK_UNIVERSE,
    india_stock_universe_symbols,
    validate_india_stock_universe,
)


class IndiaStockUniverseTests(unittest.TestCase):
    def test_tata_motors_securities_replaced_with_two_distinct_entries(self):
        universe = validate_india_stock_universe()
        symbols = tuple(entry.nse_symbol for entry in universe)

        self.assertNotIn("TATAMOTORS.NS", symbols)
        self.assertIn("TMPV.NS", symbols)
        self.assertIn("TMCV.NS", symbols)

        tmpv = next(entry for entry in universe if entry.nse_symbol == "TMPV.NS")
        tmcv = next(entry for entry in universe if entry.nse_symbol == "TMCV.NS")

        self.assertEqual(tmpv.company_name, "Tata Motors Passenger Vehicles Limited")
        self.assertEqual(tmcv.company_name, "Tata Motors Limited (Commercial Vehicles)")
        self.assertEqual(tmpv.sector, "Automobile")
        self.assertEqual(tmcv.sector, "Automobile")
        self.assertEqual(tmpv.instrument_type, "stock")
        self.assertEqual(tmcv.instrument_type, "stock")
        self.assertIsNotNone(tmpv.instrument_metadata)
        self.assertIsNotNone(tmcv.instrument_metadata)
        self.assertNotEqual(tmpv.instrument_metadata, tmcv.instrument_metadata)

    def test_new_securities_do_not_assume_identical_history(self):
        universe = validate_india_stock_universe()
        tmpv = next(entry for entry in universe if entry.nse_symbol == "TMPV.NS")
        tmcv = next(entry for entry in universe if entry.nse_symbol == "TMCV.NS")

        for entry in (tmpv, tmcv):
            self.assertIsNone(entry.listing_date)
            self.assertIn("listing-date-not-verified", entry.listing_status)
            self.assertIn("separate", entry.historical_scope)

    def test_automobile_sector_symbols_contain_both_and_not_obsolete_symbol(self):
        symbols = india_stock_universe_symbols("Automobile")
        self.assertIn("TMPV.NS", symbols)
        self.assertIn("TMCV.NS", symbols)
        self.assertNotIn("TATAMOTORS.NS", symbols)

    def test_all_universe_entries_still_validate(self):
        entries = validate_india_stock_universe()
        self.assertEqual(len(entries), len(set(entry.nse_symbol for entry in entries)))
        self.assertGreater(len(entries), 20)


if __name__ == "__main__":
    unittest.main()
