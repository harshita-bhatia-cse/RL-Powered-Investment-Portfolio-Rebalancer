import unittest

from src.strategies.allocation import PlanStatus, RebalanceConfig, RebalanceRequest, rebalance
from src.strategies.allocation.constraints import validate_and_adjust_weights


class TargetConstraintTests(unittest.TestCase):
    def test_valid_weights_and_residual_cash_are_preserved(self):
        weights, cash, events = validate_and_adjust_weights(
            {"A": 0.4, "B": 0.3}, None, RebalanceConfig()
        )
        self.assertEqual(weights, {"A": 0.4, "B": 0.3})
        self.assertAlmostEqual(cash, 0.3)
        self.assertEqual(events, [])

    def test_small_weight_sum_drift_is_normalized_and_logged(self):
        weights, cash, events = validate_and_adjust_weights(
            {"A": 0.500000004, "B": 0.5},
            None,
            RebalanceConfig(weight_sum_tolerance=1e-7),
        )
        self.assertAlmostEqual(sum(weights.values()) + cash, 1.0)
        self.assertEqual(events[0].code, "weight_normalization")

    def test_materially_invalid_weight_sum_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            validate_and_adjust_weights({"A": 0.6}, 0.3, RebalanceConfig())

    def test_negative_or_non_finite_weights_are_rejected(self):
        for weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    validate_and_adjust_weights({"A": weight}, None, RebalanceConfig())

    def test_missing_cash_is_only_inferred_for_valid_asset_sum(self):
        with self.assertRaisesRegex(ValueError, "exceed 100%"):
            validate_and_adjust_weights({"A": 1.01}, None, RebalanceConfig())

    def test_asset_cap_redistributes_excess_proportionally(self):
        weights, cash, events = validate_and_adjust_weights(
            {"A": 0.8, "B": 0.2},
            None,
            RebalanceConfig(per_asset_max_weights={"A": 0.5}),
        )
        self.assertAlmostEqual(weights["A"], 0.5)
        self.assertAlmostEqual(weights["B"], 0.5)
        self.assertAlmostEqual(cash, 0.0)
        self.assertEqual([event.symbol for event in events], ["A"])

    def test_insufficient_aggregate_caps_leave_excess_as_cash(self):
        weights, cash, _ = validate_and_adjust_weights(
            {"A": 0.8, "B": 0.2},
            None,
            RebalanceConfig(per_asset_max_weights={"A": 0.3, "B": 0.4}),
        )
        self.assertAlmostEqual(weights["A"], 0.3)
        self.assertAlmostEqual(weights["B"], 0.4)
        self.assertAlmostEqual(cash, 0.3)

    def test_cash_reserve_reduces_asset_targets_proportionally(self):
        weights, cash, events = validate_and_adjust_weights(
            {"A": 0.6, "B": 0.4},
            None,
            RebalanceConfig(cash_reserve_pct=0.2),
        )
        self.assertAlmostEqual(weights["A"], 0.48)
        self.assertAlmostEqual(weights["B"], 0.32)
        self.assertAlmostEqual(cash, 0.2)
        self.assertEqual(events[0].code, "cash_reserve")

    def test_cash_only_target_is_supported(self):
        weights, cash, _ = validate_and_adjust_weights({}, 1.0, RebalanceConfig())
        self.assertEqual(weights, {})
        self.assertEqual(cash, 1.0)

    def test_empty_zero_sum_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive total"):
            validate_and_adjust_weights({}, 0.0, RebalanceConfig())

    def test_config_rejects_invalid_limits(self):
        for config in (
            RebalanceConfig(max_weight_per_asset=1.1),
            RebalanceConfig(cash_reserve_pct=-0.1),
            RebalanceConfig(transaction_cost_rate=1.0),
            RebalanceConfig(slippage_bps=10_000),
            RebalanceConfig(max_turnover=-0.1),
            RebalanceConfig(fractional_share_precision=-1),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_and_adjust_weights({"A": 1.0}, None, config)

    def test_repeated_constraint_projection_is_deterministic(self):
        args = (
            {"A": 0.8, "B": 0.2},
            None,
            RebalanceConfig(per_asset_max_weights={"A": 0.5}),
        )
        self.assertEqual(
            validate_and_adjust_weights(*args),
            validate_and_adjust_weights(*args),
        )


class AllocationNumericalExamples(unittest.TestCase):
    def test_cash_reserve_and_asset_cap_example(self):
        result = rebalance(
            RebalanceRequest(
                current_holdings={},
                available_cash=1000,
                prices={"A": 100, "B": 100},
                target_weights={"A": 0.8, "B": 0.2},
                config=RebalanceConfig(
                    per_asset_max_weights={"A": 0.5},
                    cash_reserve_pct=0.2,
                ),
            )
        )
        self.assertEqual(result.status, PlanStatus.PARTIALLY_CONSTRAINED)
        self.assertAlmostEqual(result.adjusted_target_weights["A"], 0.5)
        self.assertAlmostEqual(result.adjusted_target_weights["B"], 0.3)
        self.assertAlmostEqual(result.cash_target_weight, 0.2)
        self.assertAlmostEqual(result.final_cash, 200)
        self.assertAlmostEqual(result.final_holdings["A"], 5)
        self.assertAlmostEqual(result.final_holdings["B"], 3)
        self.assertTrue(any(event.code == "asset_cap" for event in result.audit_events))


if __name__ == "__main__":
    unittest.main()
