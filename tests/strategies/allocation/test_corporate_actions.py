import unittest

from src.strategies.allocation import (
    CorporateActionEvent,
    CorporateActionLedger,
    CorporateActionType,
    RebalanceConfig,
    RebalanceRequest,
    rebalance,
)


def split_event(source_id="split-1", *, ratio=2.0, symbol="A"):
    return CorporateActionEvent(
        symbol=symbol,
        event_type=CorporateActionType.SPLIT,
        source_id=source_id,
        effective_date="2024-01-03",
        split_ratio=ratio,
    )


def distribution_event(
    source_id="distribution-1",
    *,
    event_type=CorporateActionType.CASH_DIVIDEND,
    payment_date="2024-01-10",
    symbol="A",
    amount=2.0,
):
    return CorporateActionEvent(
        symbol=symbol,
        event_type=event_type,
        source_id=source_id,
        announcement_date="2023-12-01",
        ex_date="2024-01-03",
        payment_date=payment_date,
        distribution_per_share=amount,
    )


class CorporateActionLedgerTests(unittest.TestCase):
    def test_forward_split_preserves_value_and_fractional_shares(self):
        ledger = CorporateActionLedger({"A": 1.25}, 0.0)
        before = ledger.snapshot({"A": 80.0})

        ledger.process_date("2024-01-03", [split_event(ratio=2.0)])
        after = ledger.snapshot({"A": 40.0})

        self.assertEqual(ledger.holdings["A"], 2.5)
        self.assertEqual(after.total_nav, before.total_nav)

    def test_reverse_split_preserves_value_and_fractional_shares(self):
        ledger = CorporateActionLedger({"A": 1.5}, 0.0)
        before = ledger.snapshot({"A": 20.0})

        ledger.process_date(
            "2024-01-03", [split_event(ratio=0.25)]
        )
        after = ledger.snapshot({"A": 80.0})

        self.assertEqual(ledger.holdings["A"], 0.375)
        self.assertEqual(after.total_nav, before.total_nav)

    def test_distribution_entitlement_uses_pre_split_holdings(self):
        ledger = CorporateActionLedger({"A": 10.0}, 100.0)
        dividend = distribution_event(amount=1.5)
        split = split_event(ratio=2.0)

        ledger.process_date("2024-01-03", [split, dividend])

        self.assertEqual(ledger.holdings["A"], 20.0)
        self.assertEqual(ledger.dividend_receivables, {"A:distribution-1": 15.0})
        self.assertEqual(ledger.settled_cash, 100.0)

    def test_receivable_is_in_nav_but_not_settled_cash(self):
        ledger = CorporateActionLedger({"A": 10.0}, 100.0)
        ledger.process_date("2024-01-03", [distribution_event(amount=2.0)])

        snapshot = ledger.snapshot({"A": 50.0})

        self.assertEqual(snapshot.dividend_receivables["A:distribution-1"], 20.0)
        self.assertEqual(snapshot.settled_cash, 100.0)
        self.assertEqual(snapshot.total_nav, 620.0)

    def test_allocator_receives_only_ledger_settled_cash(self):
        event = distribution_event(amount=2.0)
        ledger = CorporateActionLedger({"A": 1.0}, 10.0)
        ledger.process_date("2024-01-03", [event])

        result = rebalance(
            RebalanceRequest(
                current_holdings=ledger.holdings,
                available_cash=ledger.settled_cash,
                prices={"A": 10.0, "B": 10.0},
                target_weights={"B": 1.0},
                target_cash_weight=0.0,
                config=RebalanceConfig(max_turnover=2.0),
            )
        )
        ledger.update_investable_balances(result.final_holdings, result.final_cash)

        snapshot = ledger.snapshot({"A": 10.0, "B": 10.0})
        self.assertEqual(result.final_holdings, {"A": 0.0, "B": 2.0})
        self.assertEqual(result.final_cash, 0.0)
        self.assertEqual(snapshot.dividend_receivables["A:distribution-1"], 2.0)
        self.assertEqual(snapshot.total_nav, 22.0)

    def test_payment_date_moves_receivable_to_cash_once(self):
        event = distribution_event(amount=2.0)
        ledger = CorporateActionLedger({"A": 10.0}, 100.0)
        ledger.process_date("2024-01-03", [event])
        before_payment = ledger.snapshot({"A": 50.0})

        ledger.process_date("2024-01-10", [event])
        after_payment = ledger.snapshot({"A": 50.0})
        ledger.process_date("2024-01-10", [event])
        repeated_payment = ledger.snapshot({"A": 50.0})

        self.assertEqual(ledger.settled_cash, 120.0)
        self.assertEqual(ledger.dividend_receivables, {})
        self.assertEqual(after_payment.total_nav, before_payment.total_nav)
        self.assertEqual(repeated_payment.total_nav, after_payment.total_nav)

    def test_same_day_payment_settles_new_ex_date_entitlement(self):
        event = distribution_event(payment_date="2024-01-03", amount=2.0)
        ledger = CorporateActionLedger({"A": 3.0}, 5.0)

        ledger.process_date("2024-01-03", [event])

        self.assertEqual(ledger.dividend_receivables, {})
        self.assertEqual(ledger.settled_cash, 11.0)

    def test_repeated_action_event_is_not_applied_twice(self):
        event = split_event(ratio=2.0)
        ledger = CorporateActionLedger({"A": 3.0}, 0.0)
        ledger.process_date("2024-01-03", [event])
        ledger.process_date("2024-01-03", [event, event])

        self.assertEqual(ledger.holdings["A"], 6.0)
        self.assertEqual(ledger.applied_event_ids, frozenset({"A:split-1"}))

    def test_invalid_balance_sync_does_not_partially_update_ledger(self):
        ledger = CorporateActionLedger({"A": 1.0}, 5.0)
        with self.assertRaisesRegex(ValueError, "settled_cash"):
            ledger.update_investable_balances({"B": 2.0}, -1.0)

        self.assertEqual(ledger.holdings, {"A": 1.0})
        self.assertEqual(ledger.settled_cash, 5.0)

    def test_conflicting_duplicate_event_identity_is_rejected(self):
        ledger = CorporateActionLedger({"A": 3.0}, 0.0)
        ledger.process_date("2024-01-03", [split_event(ratio=2.0)])
        with self.assertRaisesRegex(ValueError, "Conflicting event records"):
            ledger.process_date("2024-01-03", [split_event(ratio=3.0)])

    def test_optional_fractional_reinvestment_is_disabled_by_default(self):
        event = distribution_event(amount=2.0)
        ledger = CorporateActionLedger({"A": 1.0}, 0.0)
        ledger.process_date("2024-01-03", [event])
        ledger.process_date("2024-01-10", [event])
        self.assertEqual(ledger.holdings["A"], 1.0)
        self.assertEqual(ledger.settled_cash, 2.0)

    def test_fractional_reinvestment_uses_payment_date_price(self):
        event = distribution_event(amount=2.0)
        ledger = CorporateActionLedger(
            {"A": 1.0}, 0.0, reinvest_distributions=True
        )
        ledger.process_date("2024-01-03", [event])
        ledger.process_date(
            "2024-01-10", [event], reinvestment_prices={"A": 3.0}
        )

        self.assertAlmostEqual(ledger.holdings["A"], 1.0 + 2.0 / 3.0)
        self.assertAlmostEqual(ledger.settled_cash, 0.0)

    def test_whole_share_reinvestment_keeps_uninvested_remainder_in_cash(self):
        event = distribution_event(amount=2.0)
        ledger = CorporateActionLedger(
            {"A": 1.0},
            0.0,
            reinvest_distributions=True,
            allow_fractional_shares=False,
        )
        ledger.process_date("2024-01-03", [event])
        ledger.process_date(
            "2024-01-10", [event], reinvestment_prices={"A": 3.0}
        )

        self.assertEqual(ledger.holdings["A"], 1.0)
        self.assertEqual(ledger.settled_cash, 2.0)

    def test_etf_distribution_uses_same_receivable_accounting(self):
        event = distribution_event(
            event_type=CorporateActionType.ETF_DISTRIBUTION,
            amount=0.5,
        )
        ledger = CorporateActionLedger({"A": 4.0}, 0.0)
        ledger.process_date("2024-01-03", [event])

        self.assertEqual(ledger.dividend_receivables["A:distribution-1"], 2.0)

    def test_skipped_action_date_is_rejected(self):
        ledger = CorporateActionLedger({"A": 1.0}, 0.0)
        with self.assertRaisesRegex(ValueError, "Missed action date"):
            ledger.process_date("2024-01-04", [split_event()])

    def test_invalid_or_mismatched_event_currency_is_rejected(self):
        event = CorporateActionEvent(
            symbol="A",
            event_type=CorporateActionType.CASH_DIVIDEND,
            source_id="eur-dividend",
            currency="EUR",
            ex_date="2024-01-03",
            distribution_per_share=1.0,
        )
        ledger = CorporateActionLedger({"A": 1.0}, 0.0)
        with self.assertRaisesRegex(ValueError, "currency does not match"):
            ledger.process_date("2024-01-03", [event])


class SplitPriceAccountingTests(unittest.TestCase):
    def test_execution_and_mark_prices_are_used_for_separate_purposes(self):
        result = rebalance(
            RebalanceRequest(
                current_holdings={},
                available_cash=1000.0,
                prices={},
                target_weights={"A": 0.5},
                target_cash_weight=0.5,
                execution_prices={"A": 100.0},
                mark_prices={"A": 110.0},
                config=RebalanceConfig(
                    transaction_cost_rate=0.01,
                    slippage_bps=100,
                ),
            )
        )

        trade = result.trades[0]
        self.assertEqual(trade.quantity, 5.0)
        self.assertAlmostEqual(trade.execution_price, 101.0)
        self.assertAlmostEqual(trade.fee, 5.05)
        self.assertAlmostEqual(trade.slippage_cost, 5.0)
        self.assertAlmostEqual(result.final_cash, 489.95)
        self.assertAlmostEqual(result.final_nav, 1039.95)
        self.assertAlmostEqual(result.actual_weights["A"], 550.0 / 1039.95)

    def test_legacy_prices_api_remains_backward_compatible(self):
        result = rebalance(
            RebalanceRequest(
                current_holdings={},
                available_cash=1000.0,
                prices={"A": 100.0},
                target_weights={"A": 0.5},
                target_cash_weight=0.5,
            )
        )

        self.assertEqual(result.trades[0].quantity, 5.0)
        self.assertEqual(result.final_nav, 1000.0)

    def test_each_cost_is_applied_once_with_legacy_prices(self):
        result = rebalance(
            RebalanceRequest(
                current_holdings={},
                available_cash=1000.0,
                prices={"A": 100.0},
                target_weights={"A": 0.5},
                target_cash_weight=0.5,
                config=RebalanceConfig(
                    transaction_cost_rate=0.01,
                    slippage_bps=100,
                ),
            )
        )

        trade = result.trades[0]
        self.assertEqual(trade.quantity, 5.0)
        self.assertAlmostEqual(trade.fee, 5.05)
        self.assertAlmostEqual(trade.slippage_cost, 5.0)
        self.assertAlmostEqual(result.total_transaction_cost, 10.05)
        self.assertAlmostEqual(result.final_cash, 489.95)
        self.assertAlmostEqual(result.final_nav, 989.95)

    def test_missing_or_non_positive_explicit_prices_are_rejected(self):
        base = dict(
            current_holdings={},
            available_cash=1000.0,
            prices={},
            target_weights={"A": 1.0},
        )
        for kwargs, error in (
            ({"execution_prices": {"A": 100.0}}, "mark prices"),
            (
                {
                    "execution_prices": {"A": 0.0},
                    "mark_prices": {"A": 100.0},
                },
                "Execution price",
            ),
            (
                {
                    "execution_prices": {"A": 100.0},
                    "mark_prices": {"A": -1.0},
                },
                "Mark price",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, error):
                    rebalance(RebalanceRequest(**base, **kwargs))


if __name__ == "__main__":
    unittest.main()
