import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from src.services.portfolio_exposure_risk import (
    EXPOSURE_RISK_THRESHOLDS,
    assess_portfolio_exposure,
)


_SESSION_DATE = date(2026, 2, 2)


def _assess(values, cash="0.00", *, exchanges=None, symbols=None):
    exchanges = exchanges or ["NSE"] * len(values)
    symbols = symbols or [f"STOCK{index}" for index in range(len(values))]
    holdings = [
        {
            "exchange": exchange,
            "symbol": symbol,
            "quantity": "1",
            "market_value": str(value),
            "valuation_status": "VALUED",
            "market_date": _SESSION_DATE.isoformat(),
        }
        for exchange, symbol, value in zip(exchanges, symbols, values)
    ]
    nav = sum((Decimal(value) for value in values), Decimal(cash))
    return assess_portfolio_exposure(
        session_date=_SESSION_DATE,
        valuation_complete=True,
        portfolio_nav=nav,
        cash_balance=Decimal(cash),
        holdings=holdings,
        evaluated_at=datetime(2026, 2, 2, tzinfo=timezone.utc),
    )


class PortfolioExposureRiskTests(unittest.TestCase):
    def test_single_stock_below_at_and_above_threshold(self):
        threshold_pct = (
            EXPOSURE_RISK_THRESHOLDS["single_stock_exposure"] * Decimal("100")
        )
        below_value = threshold_pct - Decimal("1")
        above_value = threshold_pct + Decimal("0.01")
        below = _assess([str(below_value)], str(Decimal("100") - below_value))
        exact = _assess([str(threshold_pct)], str(Decimal("100") - threshold_pct))
        above = _assess([str(above_value)], str(Decimal("100") - above_value))
        self.assertNotIn(
            "SINGLE_STOCK_EXPOSURE_HIGH", [alert["code"] for alert in below["alerts"]]
        )
        self.assertNotIn(
            "SINGLE_STOCK_EXPOSURE_HIGH", [alert["code"] for alert in exact["alerts"]]
        )
        self.assertIn(
            "SINGLE_STOCK_EXPOSURE_HIGH", [alert["code"] for alert in above["alerts"]]
        )
        self.assertEqual(
            Decimal(exact["listing_exposures"][0]["portfolio_weight_pct"]),
            threshold_pct,
        )

    def test_top_five_below_at_and_above_threshold(self):
        threshold_pct = (
            EXPOSURE_RISK_THRESHOLDS["top_five_concentration"] * Decimal("100")
        )
        below_each = (threshold_pct - Decimal("5")) / Decimal("5")
        exact_each = threshold_pct / Decimal("5")
        below = _assess(
            [str(below_each)] * 5 + ["1"],
            str(Decimal("100") - below_each * 5 - Decimal("1")),
        )
        exact = _assess(
            [str(exact_each)] * 5 + ["1"],
            str(Decimal("100") - exact_each * 5 - Decimal("1")),
        )
        above = _assess(
            [str(exact_each + Decimal("0.01"))]
            + [str(exact_each)] * 4
            + ["1"],
            str(Decimal("100") - exact_each * 5 - Decimal("0.01") - Decimal("1")),
        )

        def has_alert(result):
            return "TOP_FIVE_CONCENTRATION_HIGH" in [
                alert["code"] for alert in result["alerts"]
            ]

        self.assertFalse(has_alert(below))
        self.assertFalse(has_alert(exact))
        self.assertTrue(has_alert(above))
        self.assertEqual(Decimal(exact["top_five_weight_pct"]), threshold_pct)

    def test_cash_allocation_below_at_and_above_threshold(self):
        threshold_cash = (
            EXPOSURE_RISK_THRESHOLDS["minimum_cash_allocation"] * Decimal("100")
        )
        below_cash = threshold_cash - Decimal("0.01")
        above_cash = threshold_cash + Decimal("0.01")
        below = _assess([str(Decimal("100") - below_cash)], str(below_cash))
        exact = _assess(
            [str(Decimal("100") - threshold_cash)], str(threshold_cash)
        )
        above = _assess([str(Decimal("100") - above_cash)], str(above_cash))

        def has_alert(result):
            return "CASH_ALLOCATION_LOW" in [
                alert["code"] for alert in result["alerts"]
            ]

        self.assertTrue(has_alert(below))
        self.assertFalse(has_alert(exact))
        self.assertFalse(has_alert(above))

    def test_hhi_below_at_and_above_threshold(self):
        below = _assess(["10"] * 10)
        exact = _assess(["20", "20", "20", "20", "10", "10"])
        above = _assess(["20.01", "19.99", "20", "20", "10", "10"])

        def has_alert(result):
            return "EQUITY_HHI_HIGH" in [
                alert["code"] for alert in result["alerts"]
            ]

        self.assertFalse(has_alert(below))
        self.assertFalse(has_alert(exact))
        self.assertTrue(has_alert(above))
        self.assertEqual(Decimal(below["equity_hhi"]), Decimal("0.1"))
        self.assertEqual(
            Decimal(exact["equity_hhi"]),
            EXPOSURE_RISK_THRESHOLDS["equity_hhi"],
        )

    def test_one_five_and_ten_equally_weighted_holdings(self):
        one = _assess(["100"])
        five = _assess(["20"] * 5)
        ten = _assess(["10"] * 10)
        self.assertEqual(Decimal(one["equity_hhi"]), Decimal("1"))
        self.assertEqual(Decimal(five["equity_hhi"]), Decimal("0.2"))
        self.assertEqual(Decimal(ten["equity_hhi"]), Decimal("0.1"))

    def test_cash_only_portfolio_has_no_equity_concentration_alerts(self):
        result = _assess([], "250.00")
        self.assertEqual(result["assessment_status"], "EVALUATED")
        self.assertEqual(result["portfolio_nav"], "250.00")
        self.assertEqual(result["cash_weight_pct"], "100")
        self.assertEqual(result["equity_hhi"], None)
        self.assertEqual(result["listing_exposures"], [])
        self.assertEqual(result["alerts"], [])

    def test_zero_paise_market_value_holding_is_preserved_without_alerts(self):
        result = _assess(["0.00"], "100.00")
        self.assertEqual(result["assessment_status"], "EVALUATED")
        self.assertEqual(len(result["listing_exposures"]), 1)
        self.assertEqual(result["listing_exposures"][0]["market_value"], "0.00")
        self.assertEqual(result["listing_exposures"][0]["portfolio_weight_pct"], "0")
        self.assertIsNone(result["equity_hhi"])
        self.assertEqual(result["alerts"], [])

    def test_zero_nav_is_not_evaluated_without_partial_metrics(self):
        result = _assess([], "0.00")
        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(result["reason_codes"], ["portfolio_nav_not_positive"])
        self.assertIsNone(result["portfolio_nav"])
        self.assertIsNone(result["cash_weight_pct"])
        self.assertEqual(result["alerts"], [])

    def test_fewer_than_five_holdings_include_all_available_listings(self):
        result = _assess(["30", "20"], "50")
        self.assertEqual(Decimal(result["top_five_weight_pct"]), Decimal("50"))
        self.assertEqual(len(result["listing_exposures"]), 2)

    def test_market_value_ties_use_exchange_and_symbol_for_deterministic_order(self):
        result = _assess(
            ["10", "10", "10"],
            "70",
            exchanges=["NSE", "BSE", "NSE"],
            symbols=["ZED", "BETA", "ALFA"],
        )
        self.assertEqual(
            [row["listing"] for row in result["listing_exposures"]],
            ["BSE:BETA", "NSE:ALFA", "NSE:ZED"],
        )

    def test_decimal_values_are_kept_exact_to_paise(self):
        result = _assess(["25.01"], "74.99")
        exposure = result["listing_exposures"][0]
        self.assertEqual(exposure["market_value"], "25.01")
        self.assertEqual(Decimal(exposure["portfolio_weight_pct"]), Decimal("25.01"))
        self.assertIn(
            "SINGLE_STOCK_EXPOSURE_HIGH", [alert["code"] for alert in result["alerts"]]
        )

    def test_negative_nonfinite_and_float_values_are_rejected(self):
        for value in ("-1", "NaN", "Infinity"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _assess([value], "0")
        with self.assertRaises(ValueError):
            _assess([True], "0")
        with self.assertRaises(ValueError):
            _assess(["1"], "-1")
        with self.assertRaisesRegex(ValueError, "exact decimal"):
            assess_portfolio_exposure(
                session_date=_SESSION_DATE,
                valuation_complete=True,
                portfolio_nav=Decimal("1.25"),
                cash_balance=Decimal("0"),
                holdings=[
                    {
                        "exchange": "NSE",
                        "symbol": "TCS",
                        "quantity": Decimal("1"),
                        "market_value": 1.25,
                        "valuation_status": "VALUED",
                        "market_date": _SESSION_DATE,
                    }
                ],
            )
        with self.assertRaisesRegex(ValueError, "exact decimal"):
            assess_portfolio_exposure(
                session_date=_SESSION_DATE,
                valuation_complete=True,
                portfolio_nav=Decimal("1.00"),
                cash_balance=Decimal("0.00"),
                holdings=[
                    {
                        "exchange": "NSE",
                        "symbol": "TCS",
                        "quantity": Decimal("1"),
                        "market_value": True,
                        "valuation_status": "VALUED",
                        "market_date": _SESSION_DATE,
                    }
                ],
            )

    def test_negative_quantity_and_duplicate_listing_are_rejected(self):
        holding = {
            "exchange": "NSE",
            "symbol": "TCS",
            "quantity": Decimal("-1"),
            "market_value": Decimal("10.00"),
            "valuation_status": "VALUED",
            "market_date": _SESSION_DATE,
        }
        with self.assertRaisesRegex(ValueError, "holding quantity must be positive"):
            assess_portfolio_exposure(
                session_date=_SESSION_DATE,
                valuation_complete=True,
                portfolio_nav=Decimal("10.00"),
                cash_balance=Decimal("0.00"),
                holdings=[holding],
            )

        valid_holding = dict(holding, quantity=Decimal("1"))
        with self.assertRaisesRegex(ValueError, "duplicate holding listing"):
            assess_portfolio_exposure(
                session_date=_SESSION_DATE,
                valuation_complete=True,
                portfolio_nav=Decimal("20.00"),
                cash_balance=Decimal("0.00"),
                holdings=[valid_holding, valid_holding],
            )

    def test_subpaise_money_is_rejected_instead_of_rounded(self):
        with self.assertRaisesRegex(ValueError, "normalized to paise"):
            assess_portfolio_exposure(
                session_date=_SESSION_DATE,
                valuation_complete=True,
                portfolio_nav=Decimal("100.00"),
                cash_balance=Decimal("90.004"),
                holdings=[
                    {
                        "exchange": "NSE",
                        "symbol": "TCS",
                        "quantity": Decimal("1"),
                        "market_value": Decimal("9.996"),
                        "valuation_status": "VALUED",
                        "market_date": _SESSION_DATE,
                    }
                ],
            )

    def test_near_threshold_alert_uses_exact_minor_unit_comparison(self):
        nav = Decimal("10000000000000000000000000.00")
        market_value = Decimal("2500000000000000000000000.01")
        result = assess_portfolio_exposure(
            session_date=_SESSION_DATE,
            valuation_complete=True,
            portfolio_nav=nav,
            cash_balance=nav - market_value,
            holdings=[
                {
                    "exchange": "NSE",
                    "symbol": "TCS",
                    "quantity": Decimal("1"),
                    "market_value": market_value,
                    "valuation_status": "VALUED",
                    "market_date": _SESSION_DATE,
                }
            ],
        )
        exposure_alert = next(
            alert
            for alert in result["alerts"]
            if alert["code"] == "SINGLE_STOCK_EXPOSURE_HIGH"
        )
        self.assertGreater(Decimal(exposure_alert["observed_value"]), Decimal("25"))

    def test_nse_and_bse_same_symbol_remain_distinct(self):
        result = _assess(
            ["50", "50"],
            "0",
            exchanges=["NSE", "BSE"],
            symbols=["TCS", "TCS"],
        )
        self.assertEqual(
            {item["listing"] for item in result["listing_exposures"]},
            {"NSE:TCS", "BSE:TCS"},
        )
        self.assertEqual(len(result["listing_exposures"]), 2)

    def test_incomplete_valuation_suppresses_all_derived_metrics_and_alerts(self):
        result = assess_portfolio_exposure(
            session_date=_SESSION_DATE,
            valuation_complete=False,
            portfolio_nav=None,
            cash_balance=Decimal("100"),
            holdings=[],
            unavailable_listings=[
                {
                    "exchange": "NSE",
                    "symbol": "TCS",
                    "reason": "stored price is stale",
                }
            ],
            data_quality_warnings=[{"code": "stale_price"}],
        )
        self.assertEqual(result["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(
            result["reason_codes"], ["valuation_incomplete", "stale_price"]
        )
        self.assertIsNone(result["portfolio_nav"])
        self.assertIsNone(result["cash_weight_pct"])
        self.assertEqual(result["listing_exposures"], [])
        self.assertEqual(result["alerts"], [])
        self.assertEqual(
            result["unavailable_listings"][0]["listing"], "NSE:TCS"
        )

    def test_invalid_or_future_dated_complete_snapshot_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "on or before"):
            assess_portfolio_exposure(
                session_date=_SESSION_DATE,
                valuation_complete=True,
                portfolio_nav=Decimal("10"),
                cash_balance=Decimal("0"),
                holdings=[
                    {
                        "exchange": "NSE",
                        "symbol": "TCS",
                        "quantity": "1",
                        "market_value": "10",
                        "valuation_status": "VALUED",
                        "market_date": date(2026, 2, 3),
                    }
                ],
            )
        mismatch = assess_portfolio_exposure(
            session_date=_SESSION_DATE,
            valuation_complete=True,
            portfolio_nav=Decimal("11"),
            cash_balance=Decimal("0"),
            holdings=[
                {
                    "exchange": "NSE",
                    "symbol": "TCS",
                    "quantity": "1",
                    "market_value": "10",
                    "valuation_status": "VALUED",
                    "market_date": _SESSION_DATE,
                }
            ],
        )
        self.assertEqual(mismatch["assessment_status"], "NOT_EVALUATED")
        self.assertEqual(mismatch["reason_codes"], ["valuation_nav_mismatch"])


if __name__ == "__main__":
    unittest.main()
