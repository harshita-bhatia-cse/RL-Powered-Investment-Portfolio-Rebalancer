# Custom portfolio Gymnasium environment

`PortfolioTradingEnv` is a broker-independent research environment. It uses
the fixed, ordered asset universe `SPY, QQQ, IEF, GLD, XLU, XLV`, with `^GSPC`
as benchmark context and `^VIX` as a risk feature. The environment does not
change its action universe during an episode.

## Data contract and partitions

Pass a mapping from every required symbol to a DataFrame indexed by session
date (or with a `date` column). Each frame must contain numeric
`open, high, low, close, adj_close, volume` columns. Dates are normalized and
aligned to common `^GSPC` sessions. Missing, invalid, duplicate, or
inconsistent price rows are rejected; executable prices are never forward
filled.

The fixed date intervals are:

| Partition | Inclusive dates |
|---|---|
| Feature warm-up only | 2017-01-03–2018-01-02 |
| Training | 2018-01-03–2021-12-31 |
| Validation | 2022-01-03–2022-12-30 |
| Test | 2023-01-03–2026-11-11 |

The warm-up portion is used only to calculate trailing features. Environments
terminate on the last session in their partition, and validation/test require
a scaler fitted on training dates only.

## Observation vector

Observations are flat `float32` vectors for `MlpPolicy`, ordered as:

1. For each of the six assets in the fixed order: 1-, 5-, 21-, and 63-session
   adjusted-close returns; 21- and 63-session realized volatility; distance
   from the 21- and 63-session adjusted-close moving average.
2. `^GSPC` one-session adjusted-close return and 21-session realized
   volatility.
3. `^VIX` close divided by 100.
4. Six whole-account asset weights, settled-cash weight, dividend-receivable
   weight, portfolio drawdown from peak NAV, and NAV relative to initial
   episode NAV.

Market features are trailing-only. The scaler stores training-period means
and standard deviations and is reused unchanged for validation and test.

## Actions, allocation, and reward

The action is a Gymnasium `Box` with seven dimensions in `[-1, 1]`: one per
asset and one for cash. Finite out-of-range values are clipped and reported;
non-finite or incorrectly shaped actions are rejected. The clipped vector is
projected deterministically onto the non-negative unit simplex. Its first six
weights and final cash weight are passed to the existing deterministic
allocation engine. Asset caps, cash reserve, turnover, share precision, fees,
and slippage are configured with `RebalanceConfig`.

The reward is the net whole-account logarithmic NAV return:

```text
log(next_total_nav / current_total_nav)
```

The allocator applies execution costs once. No additional reward-level
transaction-cost penalty is applied.

Strategies that need to retain their current positions can call
`PortfolioTradingEnv.step_hold()`. It advances through the same next-session
corporate-action processing and NAV marking without running the allocator or
submitting trades; the environment still returns the normal NAV-based reward.

## Transition and corporate actions

At the close of session `t`, the environment forms the observation and receives
an action. It processes only supplied corporate-action events effective or
payable by the next session, simulates fills at that session's raw open, then
marks the account at that session's raw close. The allocator receives settled
cash only; the corporate-action ledger keeps dividend receivables outside
spendable cash and includes them in whole-account NAV.

Target weights are relative to the allocator's investable sleeve (assets plus
settled cash), not receivables. Whole-account realized weights can therefore
differ while a dividend is receivable. Initial holdings and settled cash are
assumed to be balances at the reset-date close. Events effective on or before
that date are not replayed, so pre-existing receivables must be reflected in
the initial accounting state; this environment currently has no parameter for
initial receivables.

The allocator uses its backward-compatible `prices` field as a legacy
fallback, or explicit `execution_prices`, `initial_mark_prices`, and
`mark_prices` mappings for the environment's open/close transition. Rejected
allocation plans are reported in `info` and do not update holdings/cash.

No verified historical corporate-action provider is integrated. By default,
environment metadata and each `info` record state that corporate-action
coverage and total-return accuracy are unverified. Only caller-supplied
normalized events are processed; no events are inferred or fabricated. Do not
describe results as corporate-action-complete or total-return accurate unless
the event source and coverage have been independently verified.

## Example

```python
from src.rl.environment import (
    PortfolioTradingEnv,
    fit_training_scaler,
    load_market_data,
)

market = load_market_data("data/fmp_daily")
training_scaler = fit_training_scaler(market)
env = PortfolioTradingEnv(
    market,
    partition="train",
    scaler=training_scaler,
)

observation, info = env.reset(seed=42)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

The project already declares Gymnasium and Stable-Baselines3 under the optional
`ml` dependency group in `setup.py`. Install that project extra in an isolated
environment before running the environment or a future PPO experiment. PPO is
not trained by this module.
