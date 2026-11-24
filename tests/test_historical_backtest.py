import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest.historical_metrics import calculate_historical_metrics
from src.backtest.historical_service import (
    HISTORICAL_VERIFIED_CORPORATE_ACTIONS_ENV,
    HistoricalBacktestService,
    HistoricalDataUnavailableError,
    UnsupportedStrategyError,
    CorporateActionSourceValidationError,
    validate_issuer_verified_corporate_action_source,
)
from src.data.corporate_action_provider import (
    EVENT_COLUMNS,
    CorporateActionValidationError,
    load_normalized_events,
)
from src.rl.environment.portfolio_env import (
    MARKET_SYMBOLS,
    PortfolioTradingEnv,
    fit_training_scaler,
)
from src.strategies.allocation import CorporateActionEvent, CorporateActionType

FIXTURE_DATES = pd.bdate_range("2017-01-03", "2026-11-11")


def build_market_fixture():
    """Create deterministic data that satisfies the fixed environment contract."""
    data = {}
    positions = np.arange(len(FIXTURE_DATES), dtype=np.float64)
    for symbol_index, symbol in enumerate(MARKET_SYMBOLS):
        baseline = 50.0 + symbol_index * 20.0
        close = (
            baseline
            + positions * (0.008 + symbol_index * 0.0005)
            + np.sin(positions / (13.0 + symbol_index)) * (0.4 + symbol_index * 0.02)
        )
        open_prices = close * (1.0 + 0.001 * np.sin(positions / 5.0))
        frame = pd.DataFrame(
            {
                "open": open_prices,
                "high": np.maximum(open_prices, close) * 1.002,
                "low": np.minimum(open_prices, close) * 0.998,
                "close": close,
                "adj_close": close * (1.0 + 0.0001 * np.cos(positions / 17.0)),
                "volume": np.full(len(positions), 1_000_000.0),
            },
            index=FIXTURE_DATES,
        )
        frame.index.name = "date"
        data[symbol] = frame
    return data


class HistoricalMetricsTests(unittest.TestCase):
    def test_metrics_use_nav_and_actual_cost_inputs(self):
        nav = pd.Series(
            [100.0, 110.0, 99.0, 108.9],
            index=pd.date_range("2024-01-02", periods=4, freq="B"),
        )

        metrics, daily_returns = calculate_historical_metrics(
            nav,
            turnover=0.75,
            transaction_fees=1.5,
            slippage_cost=0.5,
        )

        self.assertAlmostEqual(metrics["initial_portfolio_value"], 100.0)
        self.assertAlmostEqual(metrics["final_portfolio_value"], 108.9)
        self.assertAlmostEqual(metrics["total_return"], 0.089)
        self.assertAlmostEqual(metrics["maximum_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["transaction_fees"], 1.5)
        self.assertAlmostEqual(metrics["slippage_cost"], 0.5)
        self.assertAlmostEqual(metrics["total_transaction_cost"], 2.0)
        self.assertEqual(len(daily_returns), 3)
        self.assertTrue(np.isfinite(daily_returns.to_numpy()).all())

    def test_zero_volatility_and_insufficient_observations_are_safe(self):
        constant_nav = pd.Series(
            [100.0, 100.0, 100.0],
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        metrics, _ = calculate_historical_metrics(
            constant_nav,
            turnover=0.0,
            transaction_fees=0.0,
            slippage_cost=0.0,
        )

        self.assertEqual(metrics["annualized_volatility"], 0.0)
        self.assertIsNone(metrics["sharpe_ratio"])
        self.assertIsNone(metrics["sortino_ratio"])
        self.assertIsNone(metrics["calmar_ratio"])

        one_mark = constant_nav.iloc[:1]
        insufficient_metrics, returns = calculate_historical_metrics(
            one_mark,
            turnover=0.0,
            transaction_fees=0.0,
            slippage_cost=0.0,
        )
        self.assertEqual(len(returns), 0)
        self.assertIsNone(insufficient_metrics["annualized_volatility"])
        self.assertIsNone(insufficient_metrics["sharpe_ratio"])


class HistoricalBacktestServiceTests(unittest.TestCase):
    def setUp(self):
        self.market = build_market_fixture()
        self.service = HistoricalBacktestService(data_dir="unused-in-tests")

    def test_valid_backtest_returns_finite_nav_and_corporate_action_warning(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ):
            result = self.service.run(
                strategy_name="equal_weight",
                partition="validation",
                initial_capital=25_000.0,
            )

        self.assertEqual(result["strategy"], "equal_weight")
        self.assertEqual(result["partition"], "validation")
        self.assertEqual(result["start_date"], "2022-01-03")
        self.assertEqual(result["end_date"], "2022-12-30")
        self.assertGreater(result["number_of_trading_sessions"], 2)
        self.assertEqual(
            len(result["nav_history"]), result["number_of_trading_sessions"]
        )
        self.assertEqual(
            len(result["daily_returns"]), result["number_of_trading_sessions"] - 1
        )
        self.assertTrue(
            np.isfinite([point["nav"] for point in result["nav_history"]]).all()
        )
        self.assertTrue(
            np.isfinite([point["value"] for point in result["daily_returns"]]).all()
        )
        self.assertFalse(result["corporate_actions_verified"])
        self.assertTrue(result["warnings"])
        self.assertIn("defaults transaction fees", " ".join(result["warnings"]))
        self.assertEqual(result["metrics"]["initial_portfolio_value"], 25_000.0)

    def test_invalid_strategy_and_partition_fail_before_loading_data(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            side_effect=AssertionError("data should not be loaded"),
        ):
            with self.assertRaises(UnsupportedStrategyError):
                self.service.run(strategy_name="unknown", partition="test")
            with self.assertRaisesRegex(ValueError, "partition"):
                self.service.run(strategy_name="equal_weight", partition="invalid")

    def test_missing_market_data_is_reported_as_unavailable(self):
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value={},
        ):
            with self.assertRaises(HistoricalDataUnavailableError):
                self.service.run(strategy_name="equal_weight", partition="test")

    def test_scaler_is_training_fitted_and_shared_for_comparison(self):
        changed_market = {symbol: frame.copy() for symbol, frame in self.market.items()}
        changed_market["SPY"].loc[
            changed_market["SPY"].index >= pd.Timestamp("2023-01-03"), "adj_close"
        ] *= 10.0
        baseline_scaler = fit_training_scaler(self.market)
        changed_scaler = fit_training_scaler(changed_market)
        self.assertEqual(changed_scaler.fitted_start, "2018-01-03")
        self.assertEqual(changed_scaler.fitted_end, "2021-12-31")
        np.testing.assert_array_equal(baseline_scaler.means, changed_scaler.means)
        np.testing.assert_array_equal(baseline_scaler.scales, changed_scaler.scales)

        captured_scalers = []
        original_env = PortfolioTradingEnv

        def capture_env(*args, **kwargs):
            captured_scalers.append(kwargs["scaler"])
            return original_env(*args, **kwargs)

        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ), patch(
            "src.backtest.historical_service.PortfolioTradingEnv",
            side_effect=capture_env,
        ):
            result = self.service.compare(
                strategy_names=["equal_weight", "buy_and_hold"],
                partition="test",
                initial_capital=10_000.0,
            )

        self.assertEqual(len(captured_scalers), 2)
        self.assertIs(captured_scalers[0], captured_scalers[1])
        self.assertEqual(
            [item["strategy"] for item in result["results"]],
            ["equal_weight", "buy_and_hold"],
        )
        self.assertEqual(
            {item["initial_nav"] for item in result["results"]},
            {10_000.0},
        )
        self.assertTrue(
            all(
                item["corporate_actions_verified"] is False
                for item in result["results"]
            )
        )
        self.assertEqual(
            result["results"][0]["number_of_trading_sessions"],
            result["results"][1]["number_of_trading_sessions"],
        )

    def test_invalid_strategy_parameters_are_rejected(self):
        with self.assertRaises(UnsupportedStrategyError):
            self.service.run(
                strategy_name="equal_weight",
                partition="test",
                parameters={"lookback": 20},
            )


class HistoricalCorporateActionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.market = build_market_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def write_source(
        self,
        *,
        filename="verified_normalized_events.csv",
        provider=(
            "State Street Global Advisors; Yahoo candidates used only for comparison"
        ),
        source_workbook_url=(
            "https://www.ssga.com/us/en/intermediary/library-content/products/"
            "fund-data/etfs/us/spdr-etf-historical-distributions.xlsx"
        ),
        declared_filename=None,
        declared_sha256=None,
    ):
        artifact = self.root / filename
        artifact.write_bytes(b"fixture artifact bytes\n")
        artifact_name = declared_filename or filename
        checksum = declared_sha256 or hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest = {
            "provider": provider,
            "source_workbook_url": source_workbook_url,
            "retrieved_at": "2026-11-24T00:00:00+00:00",
            "requested_symbols": ["SPY", "QQQ", "IEF", "GLD", "XLU", "XLV"],
            "date_range": {"start": "2018-01-03", "end": "2026-11-11"},
            "issuer_workbook": {
                "preserved_as": "evidence/ssga_spdr_historical_distributions.xlsx",
                "sha256": "0" * 64,
                "worksheet": "dividend",
            },
            "counts": {"issuer_verified_normalized_events": 0},
            "artifacts": {
                artifact_name: {"sha256": checksum, "bytes": artifact.stat().st_size}
            },
            "limitations": ["Issuer event coverage is incomplete for QQQ and IEF."],
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return artifact, manifest

    def write_event_source(self, rows):
        artifact, manifest = self.write_source()
        with artifact.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(EVENT_COLUMNS)
            writer.writerows(rows)
        artifact_info = manifest["artifacts"]["verified_normalized_events.csv"]
        artifact_info["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact_info["bytes"] = artifact.stat().st_size
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return artifact

    def test_default_configuration_is_disabled_without_environment_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            service = HistoricalBacktestService(data_dir="custom-market-data")

        self.assertEqual(service.data_dir, Path("custom-market-data"))
        self.assertIsNone(service.corporate_action_events_path)

    def test_explicit_constructor_path_takes_precedence_over_environment(self):
        with patch.dict(
            os.environ,
            {HISTORICAL_VERIFIED_CORPORATE_ACTIONS_ENV: str(self.root / "env.csv")},
        ):
            service = HistoricalBacktestService(
                corporate_action_events_path=self.root / "explicit.csv"
            )

        self.assertEqual(
            service.corporate_action_events_path, self.root / "explicit.csv"
        )

    def test_environment_variable_configures_source_when_constructor_omits_it(self):
        configured_path = self.root / "verified_normalized_events.csv"
        with patch.dict(
            os.environ,
            {HISTORICAL_VERIFIED_CORPORATE_ACTIONS_ENV: str(configured_path)},
        ):
            service = HistoricalBacktestService()

        self.assertEqual(service.corporate_action_events_path, configured_path)

    def test_valid_issuer_workflow_manifest_and_artifact_are_accepted(self):
        artifact, expected_manifest = self.write_source()

        actual_manifest = validate_issuer_verified_corporate_action_source(artifact)

        self.assertEqual(actual_manifest, expected_manifest)

    def test_configured_source_is_validated_before_market_data_loading(self):
        artifact, _ = self.write_source(filename="yahoo_action_candidates.csv")
        service = HistoricalBacktestService(corporate_action_events_path=artifact)

        with patch(
            "src.backtest.historical_service.load_market_data"
        ) as load_market_data, patch(
            "src.backtest.historical_service.load_normalized_events"
        ) as load_events:
            with self.assertRaisesRegex(
                CorporateActionSourceValidationError,
                "Expected the issuer workflow artifact",
            ):
                service._load_context()

        load_market_data.assert_not_called()
        load_events.assert_not_called()

    def test_unconfigured_source_does_not_load_events(self):
        with patch.dict(os.environ, {HISTORICAL_VERIFIED_CORPORATE_ACTIONS_ENV: ""}):
            service = HistoricalBacktestService()
        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ), patch(
            "src.backtest.historical_service.load_normalized_events"
        ) as load_events:
            _, _, events = service._load_context()

        self.assertEqual(events, ())
        load_events.assert_not_called()

    def test_configured_source_loads_events_with_manifest_bounds_and_full_calendar(
        self,
    ):
        artifact = self.write_event_source(
            [
                (
                    "SPY",
                    "cash_dividend",
                    "training-event",
                    "USD",
                    "2020-06-01",
                    "2020-06-15",
                    "2020-06-30",
                    "",
                    "0.5",
                    "",
                ),
                (
                    "QQQ",
                    "etf_distribution",
                    "test-event",
                    "USD",
                    "2024-06-03",
                    "2024-06-17",
                    "2024-06-28",
                    "",
                    "0.7",
                    "",
                ),
            ]
        )
        service = HistoricalBacktestService(corporate_action_events_path=artifact)

        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ), patch(
            "src.backtest.historical_service.load_normalized_events",
            wraps=load_normalized_events,
        ) as load_events:
            _, _, events = service._load_context()

        self.assertEqual(
            events,
            (
                CorporateActionEvent(
                    symbol="SPY",
                    event_type=CorporateActionType.CASH_DIVIDEND,
                    source_id="training-event",
                    announcement_date="2020-06-01",
                    ex_date="2020-06-15",
                    payment_date="2020-06-30",
                    distribution_per_share=0.5,
                ),
                CorporateActionEvent(
                    symbol="QQQ",
                    event_type=CorporateActionType.ETF_DISTRIBUTION,
                    source_id="test-event",
                    announcement_date="2024-06-03",
                    ex_date="2024-06-17",
                    payment_date="2024-06-28",
                    distribution_per_share=0.7,
                ),
            ),
        )
        load_events.assert_called_once()
        loader_kwargs = load_events.call_args.kwargs
        self.assertEqual(loader_kwargs["start_date"], "2018-01-03")
        self.assertEqual(loader_kwargs["end_date"], "2026-11-11")
        self.assertTrue(
            loader_kwargs["market_sessions"].equals(self.market["SPY"].index)
        )
        self.assertEqual(
            loader_kwargs["market_sessions"][0], self.market["SPY"].index[0]
        )
        self.assertEqual(
            loader_kwargs["market_sessions"][-1], self.market["SPY"].index[-1]
        )

    def test_invalid_event_data_raises_loader_validation_error(self):
        artifact = self.write_event_source(
            [
                (
                    "SPY",
                    "cash_dividend",
                    "weekend-event",
                    "USD",
                    "2021-06-01",
                    "2021-06-19",
                    "2021-06-30",
                    "",
                    "0.5",
                    "",
                )
            ]
        )
        service = HistoricalBacktestService(corporate_action_events_path=artifact)

        with patch(
            "src.backtest.historical_service.load_market_data",
            return_value=self.market,
        ):
            with self.assertRaisesRegex(
                CorporateActionValidationError, "does not align"
            ):
                service._load_context()

    def test_missing_manifest_is_rejected(self):
        artifact, _ = self.write_source()
        (self.root / "manifest.json").unlink()

        with self.assertRaisesRegex(
            CorporateActionSourceValidationError, "manifest is missing"
        ):
            validate_issuer_verified_corporate_action_source(artifact)

    def test_malformed_manifest_is_rejected(self):
        artifact, _ = self.write_source()
        (self.root / "manifest.json").write_text("{not-json", encoding="utf-8")

        with self.assertRaisesRegex(CorporateActionSourceValidationError, "malformed"):
            validate_issuer_verified_corporate_action_source(artifact)

    def test_missing_or_malformed_manifest_date_ranges_are_rejected(self):
        invalid_ranges = (
            (None, "date_range object"),
            ({}, "date_range.start"),
            ({"start": "2018-01-03"}, "date_range.end"),
            (
                {"start": "2018-1-3", "end": "2026-11-11"},
                "date_range.start",
            ),
            (
                {"start": "2018-02-30", "end": "2026-11-11"},
                "valid ISO-format date",
            ),
            (
                {"start": "2026-11-12", "end": "2026-11-11"},
                "must not be after",
            ),
        )
        for date_range, expected_error in invalid_ranges:
            with self.subTest(date_range=date_range):
                artifact, manifest = self.write_source()
                if date_range is None:
                    manifest.pop("date_range")
                else:
                    manifest["date_range"] = date_range
                (self.root / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

                with self.assertRaisesRegex(
                    CorporateActionSourceValidationError,
                    expected_error,
                ):
                    validate_issuer_verified_corporate_action_source(artifact)

    def test_incorrect_artifact_byte_size_is_rejected(self):
        artifact, manifest = self.write_source()
        manifest["artifacts"]["verified_normalized_events.csv"]["bytes"] += 1
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(
            CorporateActionSourceValidationError, "Byte-size mismatch"
        ):
            validate_issuer_verified_corporate_action_source(artifact)

    def test_unsupported_provider_or_workbook_identity_is_rejected(self):
        for manifest_change, expected_error in (
            ({"provider": "Yahoo Finance"}, "Unsupported issuer verification"),
            (
                {"source_workbook_url": "https://example.invalid/events.xlsx"},
                "unsupported source_workbook_url",
            ),
        ):
            with self.subTest(manifest_change=manifest_change):
                artifact, manifest = self.write_source()
                manifest.update(manifest_change)
                (self.root / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

                with self.assertRaisesRegex(
                    CorporateActionSourceValidationError,
                    expected_error,
                ):
                    validate_issuer_verified_corporate_action_source(artifact)

    def test_wrong_artifact_filename_or_undeclared_artifact_is_rejected(self):
        for filename in (
            "yahoo_action_candidates.csv",
            "issuer_verification_template.csv",
            "arbitrary_events.csv",
        ):
            with self.subTest(filename=filename):
                artifact, _ = self.write_source(filename=filename)
                with self.assertRaisesRegex(
                    CorporateActionSourceValidationError,
                    "Expected the issuer workflow artifact",
                ):
                    validate_issuer_verified_corporate_action_source(artifact)

        artifact, _ = self.write_source(declared_filename="yahoo_action_candidates.csv")
        with self.assertRaisesRegex(
            CorporateActionSourceValidationError, "does not declare"
        ):
            validate_issuer_verified_corporate_action_source(artifact)

    def test_sha256_checksum_mismatch_is_rejected(self):
        artifact, _ = self.write_source(declared_sha256="0" * 64)

        with self.assertRaisesRegex(
            CorporateActionSourceValidationError, "SHA-256 checksum mismatch"
        ):
            validate_issuer_verified_corporate_action_source(artifact)


if __name__ == "__main__":
    unittest.main()
