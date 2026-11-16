# Deterministic allocation and rebalance API

This package calculates a simulated long-only rebalance. It is independent of
the RL environment, the adaptive rotation strategy, and all brokerage code.
Calling it never submits orders or changes an account.

```python
from src.strategies.allocation import RebalanceConfig, RebalanceRequest, rebalance

plan = rebalance(
    RebalanceRequest(
        current_holdings={"AAPL": 2},
        available_cash=800,
        prices={"AAPL": 100, "MSFT": 100},
        target_weights={"AAPL": 0.5, "MSFT": 0.3},
        target_cash_weight=0.2,
        config=RebalanceConfig(
            max_weight_per_asset=0.6,
            cash_reserve_pct=0.1,
            transaction_cost_rate=0.001,
            slippage_bps=5,
            max_turnover=0.5,
        ),
    )
)

print(plan.status.value)
for trade in plan.trades:
    print(trade.side, trade.quantity, trade.symbol, trade.execution_price)
print(plan.final_holdings, plan.final_cash, plan.actual_weights)
```

When only the legacy `prices` mapping is supplied, it is used for both
execution references and marks. Simulated execution prices include adverse
slippage. Fees are charged once on execution notional. Sells are simulated
before buys. The cash reserve is a minimum amount equal to
`cash_reserve_pct * initial_nav`. Gross turnover is the sum of absolute
execution-reference trade notionals for buys and sells divided by initial NAV;
a full-portfolio switch can therefore have a turnover ratio near 2.0.

For next-session fills and separate portfolio marks, callers may pass
`execution_prices`, `mark_prices`, and `initial_mark_prices` on
`RebalanceRequest`. The original `prices` mapping remains supported and is
used for mappings that are omitted. Initial investable NAV uses
`initial_mark_prices` (defaulting to `mark_prices`); target share quantities,
reference notionals, and turnover use execution prices; final NAV and realized
weights use mark prices. Each effective mapping must cover every held and
targeted symbol.

Target weights are relative to the **investable sleeve** (asset holdings plus
settled cash). Dividend receivables are not passed as available cash and are
not included in allocator NAV. The whole-account corporate-action ledger owns
receivables and includes them in its separate NAV snapshot.

## Corporate-action accounting foundation

`CorporateActionLedger` processes normalized `CorporateActionEvent` records;
it does not fetch, infer, or fabricate events. It supports splits, reverse
splits, cash dividends, and ETF distributions. Distribution entitlement is
based on holdings before same-date split adjustments. Receivables count toward
the ledger snapshot NAV but remain unavailable as allocator cash until the
payment date. Each event is keyed by symbol and provider `source_id`; repeating
an identical event is idempotent and a conflicting duplicate is rejected.
Calls must be chronological and must process every known event date.
After a non-rejected rebalance, `update_investable_balances` synchronizes the
ledger from the allocator's final holdings and settled cash without touching
receivables or event application state.

Fractional shares created by splits are retained. Optional distribution
reinvestment is disabled by default; when enabled, it uses the supplied
payment-date reinvestment price, has no modeled fees, and any uninvested
remainder remains settled cash. The ledger currently supports one currency at
a time and does not model FX, withholding/taxes, or cash-in-lieu for split
fractions. Announcement dates are retained as event metadata; the calling
environment remains responsible for not exposing an event before it was
publicly known.

No verified historical corporate-action provider is integrated yet. Until a
provider and its event coverage are validated, this foundation does not make
the local price data corporate-action complete and must not be described as a
verified total-return backtest.

The target weights must sum to 1.0 including cash, within
`weight_sum_tolerance`. When `target_cash_weight` is omitted, cash is inferred
as the residual. Only floating-point drift within tolerance is normalized;
materially invalid sums raise `ValueError`. Held symbols absent from target
weights are targeted to zero.

`PlanStatus.COMPLETED` means no constraint adjustments were needed.
`PARTIALLY_CONSTRAINED` means constraints, costs, or share precision altered
the target result. `REJECTED` means the request was valid, but the simulated
plan could not maintain the hard cash reserve or another accounting
invariant; rejected plans contain no trades and retain the original portfolio.
Invalid input raises `ValueError`.
