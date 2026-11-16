import unittest

from src.strategies.allocation import (
    PlanStatus,
    RebalanceConfig,
    RebalanceRequest,
    rebalance,
)


def request(
    *,
    holdings=None,
    cash=0.0,
    prices=None,
    weights=None,
    cash_weight=None,
    config=None,
):
    return RebalanceRequest(
        current_holdings=holdings or {},
        available_cash=cash,
        prices=prices or {},
        target_weights=weights or {},
        target_cash_weight=cash_weight,
        config=config or RebalanceConfig(),
    )


class RebalanceAccountingTests(unittest.TestCase):
    def test_buy_asset_numerical_example(self):
        result = rebalance(
            request(cash=1000, prices={"A": 100}, weights={"A": 0.5}, cash_weight=0.5)
        )
        self.assertEqual(result.status, PlanStatus.COMPLETED)
        self.assertEqual([(trade.side, trade.quantity) for trade in result.trades], [("buy", 5)])
        self.assertAlmostEqual(result.final_cash, 500)
        self.assertAlmostEqual(result.final_nav, 1000)
        self.assertAlmostEqual(result.actual_weights["A"], 0.5)
        self.assertAlmostEqual(result.actual_cash_weight, 0.5)

    def test_sell_asset_numerical_example(self):
        result = rebalance(
            request(
                holdings={"A": 10},
                cash=0,
                prices={"A": 100},
                weights={"A": 0.5},
                cash_weight=0.5,
            )
        )
        self.assertEqual([(trade.side, trade.quantity) for trade in result.trades], [("sell", 5)])
        self.assertAlmostEqual(result.final_holdings["A"], 5)
        self.assertAlmostEqual(result.final_cash, 500)
        self.assertAlmostEqual(result.final_nav, 1000)

    def test_multi_asset_rebalance_numerical_example(self):
        result = rebalance(
            request(
                holdings={"A": 4, "B": 4},
                cash=200,
                prices={"A": 100, "B": 100},
                weights={"A": 0.3, "B": 0.4},
                cash_weight=0.3,
            )
        )
        self.assertEqual([(trade.side, trade.symbol, trade.quantity) for trade in result.trades],
                         [("sell", "A", 1)])
        self.assertAlmostEqual(result.final_holdings["A"], 3)
        self.assertAlmostEqual(result.final_holdings["B"], 4)
        self.assertAlmostEqual(result.final_cash, 300)
        self.assertAlmostEqual(result.final_nav, 1000)
        self.assertAlmostEqual(sum(result.actual_weights.values()) + result.actual_cash_weight, 1.0)

    def test_insufficient_cash_reduces_buys_and_reports_partial_status(self):
        result = rebalance(
            request(
                cash=500,
                prices={"A": 100},
                weights={"A": 1.0},
                config=RebalanceConfig(transaction_cost_rate=0.01),
            )
        )
        self.assertEqual(result.status, PlanStatus.PARTIALLY_CONSTRAINED)
        self.assertEqual(result.trades[0].quantity, 4)
        self.assertAlmostEqual(result.final_cash, 96)
        self.assertAlmostEqual(result.total_fees, 4)
        self.assertTrue(any(event.code == "cash_affordability" for event in result.audit_events))

    def test_cash_reserve_is_enforced_even_when_target_omits_cash(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 100},
                weights={"A": 1.0},
                config=RebalanceConfig(cash_reserve_pct=0.2),
            )
        )
        self.assertEqual(result.status, PlanStatus.PARTIALLY_CONSTRAINED)
        self.assertAlmostEqual(result.adjusted_target_weights["A"], 0.8)
        self.assertAlmostEqual(result.cash_target_weight, 0.2)
        self.assertAlmostEqual(result.final_cash, 200)
        self.assertGreaterEqual(result.final_cash, result.initial_nav * 0.2)

    def test_costs_cannot_consume_the_hard_cash_reserve(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 100},
                weights={"A": 0.8},
                cash_weight=0.2,
                config=RebalanceConfig(
                    cash_reserve_pct=0.2,
                    transaction_cost_rate=0.01,
                ),
            )
        )
        self.assertGreaterEqual(result.final_cash, 200)
        self.assertEqual(result.trades[0].quantity, 7)
        self.assertAlmostEqual(result.total_fees, 7)
        self.assertTrue(any(event.code == "cash_affordability" for event in result.audit_events))

    def test_asset_cap_and_redistribution_example(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 100, "B": 100},
                weights={"A": 0.8, "B": 0.2},
                config=RebalanceConfig(per_asset_max_weights={"A": 0.5}),
            )
        )
        self.assertAlmostEqual(result.adjusted_target_weights["A"], 0.5)
        self.assertAlmostEqual(result.adjusted_target_weights["B"], 0.5)
        self.assertEqual(result.final_holdings, {"A": 5, "B": 5})

    def test_turnover_limit_is_respected_after_rounding(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 100},
                weights={"A": 1.0},
                config=RebalanceConfig(max_turnover=0.25),
            )
        )
        self.assertEqual(result.status, PlanStatus.PARTIALLY_CONSTRAINED)
        self.assertEqual(result.trades[0].quantity, 2)
        self.assertAlmostEqual(result.gross_turnover, 200)
        self.assertLessEqual(result.turnover_ratio, 0.25)
        self.assertTrue(any(event.code == "turnover_limit" for event in result.audit_events))

    def test_whole_share_rounding(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 300},
                weights={"A": 0.5},
                cash_weight=0.5,
                config=RebalanceConfig(allow_fractional_shares=False),
            )
        )
        self.assertEqual(result.trades[0].quantity, 1)
        self.assertEqual(result.final_holdings["A"], 1)
        self.assertAlmostEqual(result.final_cash, 700)
        self.assertEqual(result.status, PlanStatus.PARTIALLY_CONSTRAINED)

    def test_fractional_share_rounding(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 300},
                weights={"A": 0.5},
                cash_weight=0.5,
                config=RebalanceConfig(
                    allow_fractional_shares=True, fractional_share_precision=3
                ),
            )
        )
        self.assertEqual(result.trades[0].quantity, 1.666)
        self.assertAlmostEqual(result.final_holdings["A"], 1.666)
        self.assertAlmostEqual(result.final_cash, 500.2)
        self.assertEqual(result.status, PlanStatus.PARTIALLY_CONSTRAINED)

    def test_complete_liquidation(self):
        result = rebalance(
            request(
                holdings={"A": 5},
                cash=500,
                prices={"A": 100},
                weights={},
                cash_weight=1.0,
            )
        )
        self.assertEqual(result.status, PlanStatus.COMPLETED)
        self.assertEqual(result.trades[0].side, "sell")
        self.assertEqual(result.trades[0].quantity, 5)
        self.assertAlmostEqual(result.final_holdings["A"], 0)
        self.assertAlmostEqual(result.final_cash, 1000)
        self.assertAlmostEqual(result.actual_cash_weight, 1)

    def test_fees_and_slippage_are_applied_once(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 100},
                weights={"A": 0.5},
                cash_weight=0.5,
                config=RebalanceConfig(
                    transaction_cost_rate=0.01,
                    slippage_bps=100,
                ),
            )
        )
        trade = result.trades[0]
        self.assertAlmostEqual(trade.execution_price, 101)
        self.assertAlmostEqual(trade.execution_notional, 505)
        self.assertAlmostEqual(trade.fee, 5.05)
        self.assertAlmostEqual(trade.slippage_cost, 5)
        self.assertAlmostEqual(result.total_transaction_cost, 10.05)
        self.assertAlmostEqual(result.final_cash, 489.95)
        self.assertAlmostEqual(result.final_nav, 989.95)

    def test_sell_trade_proceeds_reflect_costs_once(self):
        result = rebalance(
            request(
                holdings={"A": 5},
                cash=500,
                prices={"A": 100},
                weights={},
                cash_weight=1.0,
                config=RebalanceConfig(
                    transaction_cost_rate=0.01,
                    slippage_bps=100,
                ),
            )
        )
        self.assertAlmostEqual(result.final_cash, 990.05)
        self.assertAlmostEqual(result.final_nav, 990.05)
        self.assertAlmostEqual(result.total_transaction_cost, 9.95)

    def test_sells_execute_before_buys(self):
        result = rebalance(
            request(
                holdings={"A": 5},
                cash=0,
                prices={"A": 100, "B": 100},
                weights={"B": 1.0},
                cash_weight=0.0,
                config=RebalanceConfig(max_turnover=2.0),
            )
        )
        self.assertEqual([trade.side for trade in result.trades], ["sell", "buy"])
        self.assertEqual(result.final_holdings["A"], 0)
        self.assertEqual(result.final_holdings["B"], 5)
        self.assertAlmostEqual(result.final_cash, 0)

    def test_insufficient_turnover_to_meet_cash_reserve_rejects_without_trades(self):
        result = rebalance(
            request(
                holdings={"A": 10},
                cash=0,
                prices={"A": 100},
                weights={},
                cash_weight=1.0,
                config=RebalanceConfig(cash_reserve_pct=0.2, max_turnover=0),
            )
        )
        self.assertEqual(result.status, PlanStatus.REJECTED)
        self.assertEqual(result.trades, ())
        self.assertEqual(result.final_holdings, {"A": 10})
        self.assertEqual(result.final_cash, 0)
        self.assertTrue(result.rejection_reasons)

    def test_target_unchanged_generates_no_trade(self):
        result = rebalance(
            request(
                holdings={"A": 5},
                cash=500,
                prices={"A": 100},
                weights={"A": 0.5},
                cash_weight=0.5,
            )
        )
        self.assertEqual(result.status, PlanStatus.COMPLETED)
        self.assertEqual(result.trades, ())
        self.assertEqual(result.audit_events, ())

    def test_zero_turnover_limit_permits_no_trades(self):
        result = rebalance(
            request(
                cash=1000,
                prices={"A": 100},
                weights={"A": 1.0},
                config=RebalanceConfig(max_turnover=0),
            )
        )
        self.assertEqual(result.trades, ())
        self.assertEqual(result.turnover_ratio, 0)

    def test_cash_only_target_liquidates_all_holdings(self):
        result = rebalance(
            request(
                holdings={"A": 2, "B": 3},
                cash=100,
                prices={"A": 100, "B": 100},
                weights={},
                cash_weight=1,
            )
        )
        self.assertEqual(result.final_holdings, {"A": 0, "B": 0})
        self.assertAlmostEqual(result.final_cash, 600)

    def test_actual_weights_and_nav_satisfy_accounting_invariants(self):
        result = rebalance(
            request(
                holdings={"A": 4},
                cash=400,
                prices={"A": 100, "B": 50},
                weights={"A": 0.3, "B": 0.4},
                cash_weight=0.3,
                config=RebalanceConfig(transaction_cost_rate=0.002, slippage_bps=5),
            )
        )
        marked_assets = sum(
            quantity * {"A": 100, "B": 50}[symbol]
            for symbol, quantity in result.final_holdings.items()
        )
        self.assertAlmostEqual(result.final_nav, result.final_cash + marked_assets)
        self.assertAlmostEqual(
            sum(result.actual_weights.values()) + result.actual_cash_weight, 1.0
        )
        self.assertAlmostEqual(
            result.final_nav,
            result.initial_nav - result.total_fees - result.total_slippage_cost,
        )

    def test_invalid_prices_holdings_and_cash_are_rejected(self):
        invalid_requests = (
            request(cash=10, prices={"A": 0}, weights={"A": 1}),
            request(holdings={"A": -1}, cash=100, prices={"A": 10}, weights={"A": 1}),
            request(holdings={"A": 1}, cash=0, prices={}, weights={"A": 1}),
            request(cash=-1, prices={"A": 10}, weights={"A": 1}),
            request(cash=0, prices={"A": 10}, weights={"A": 1}),
        )
        for invalid in invalid_requests:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                rebalance(invalid)

    def test_invalid_weight_total_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            rebalance(request(cash=100, prices={"A": 100}, weights={"A": 0.5}, cash_weight=0.4))

    def test_deterministic_trade_and_audit_order(self):
        req = request(
            cash=1000,
            prices={"B": 100, "A": 100},
            weights={"B": 0.3, "A": 0.4},
            cash_weight=0.3,
            config=RebalanceConfig(max_weight_per_asset=0.5, max_turnover=0.5),
        )
        first = rebalance(req)
        second = rebalance(req)
        self.assertEqual(first, second)
        self.assertEqual([trade.symbol for trade in first.trades], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
