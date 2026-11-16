# Phase 5.12 - Portfolio Environment Implementation

**Status:** Implemented and tested with deterministic in-memory synthetic
inputs. This environment is for software development and controlled tests only;
it does not establish real-market behavior or investment performance.

## Files created or modified

- Created `src/rl/environment/synthetic_portfolio_env.py`.
- Modified `src/rl/environment/__init__.py` to export
  `SyntheticPortfolioEnv`.
- Created `tests/rl/test_synthetic_portfolio_env.py`.
- Created this implementation report.

No existing market-data environment, production data processor, dashboard,
historical CSV, `.env`, PPO model, backtest, or trade-execution code was
changed. No market data was loaded or used.

## Environment design

`SyntheticPortfolioEnv` accepts in-memory daily simple-return arrays for six
assets in the fixed order `SPY, QQQ, IEF, GLD, XLU, XLV`, one synthetic
benchmark return series, and one positive synthetic VIX-level series. Inputs
must align in length, have finite values, and be explicitly accompanied by
metadata with both `source_type="synthetic"` and `synthetic=true`.

The constructor copies and marks input arrays read-only. It does not load files
or generate data. At least 64 sessions are required: 63 sessions warm up the
observation features, followed by at least one decision/return session.
The action projection reuses the existing fixed asset-order and simplex
projection helpers from `portfolio_env.py`; importing that module does not
invoke its file loader.

The standard Gymnasium API is implemented:

- `reset(seed=..., options=...)` returns `(observation, info)`, resets NAV to
  the initial capital and starts with all cash. Non-empty reset options are
  rejected.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`.
  Natural exhaustion, including a final partial period, sets
  `terminated=true`; this environment does not impose an external time limit,
  so `truncated=false`.
- Calling `step()` after termination raises `RuntimeError`.
- `render()` supports `human` and `ansi` modes; `close()` is a no-op because
  the environment owns no external resources.

`reset(seed=...)` follows Gymnasium's seeded API. The environment itself uses no
randomness, so a fixed scenario and action sequence are deterministic
regardless of seed.

## Observation and action definitions

The observation is a finite `float32` vector of **43** values, in this order:

1. For each of the six ETFs: latest one-session return, compounded 5- and
   21-session returns, and annualized sample volatility over 21 and 63
   sessions (30 features).
2. Synthetic benchmark latest return and annualized 21-session volatility
   (2 features).
3. Latest synthetic VIX level divided by `vix_level_scale` (default 100)
   (1 feature).
4. Current six ETF portfolio weights and cash weight (7 features).
5. NAV divided by initial capital, drawdown as `NAV / peak_NAV - 1`, and the
   remaining fraction of return sessions (3 features).

Every market observation uses data ending before the next return interval. The
observation `Box` is bounded by the finite `float32` range; non-finite generated
observations raise an explicit error.

The action space is `Box(-1, 1, shape=(7,), dtype=float32)`, where the first six
components correspond to ETFs and the final component is cash. Finite values
outside the declared bounds are clipped and noted in `info`; wrong-shape and
NaN/Inf actions raise `ValueError`. The clipped action is deterministically
projected onto a non-negative unit simplex. A bounded-simplex projection then
enforces asset caps and minimum cash. Finally, if gross turnover exceeds its
limit, the change from current weights is scaled linearly toward the constrained
target. Projected targets always satisfy asset caps and minimum cash. Market
returns can cause marked holdings to drift outside those bounds between
rebalance dates; when the turnover limit prevents full restoration at the next
rebalance, the environment permits only a normalized, long-only transition that
does not worsen the existing breach and reports `constraint_recovery=true`.
This preserves the turnover limit without mistaking recoverable market drift
for an invalid action. Requested, constraint-adjusted, and applied weight maps
are returned in `info`.

## Accounting conventions and formulas

- Initial NAV is **100,000** synthetic units (reported as `SIM` by future
  application clients), initially held entirely as cash.
- Cash earns **0%**. The account is long-only and unlevered. Cash is a seventh
  portfolio component; `^GSPC` and `^VIX` are observation-only.
- Defaults follow the Phase 5.11 engineering fixture: monthly decision period
  of 21 sessions, 35% maximum weight per ETF, 2% minimum cash, 100% maximum
  gross turnover, 10 bps fee, and 5 bps slippage. These values are arbitrary
  software-test assumptions, not empirically calibrated market parameters.
- An action at session index `t` consumes returns `[t, min(t + 21, n))`.
  Observations at `t` use only sessions before `t`; the interval's returns
  cannot influence the action for that interval.
- Per-asset growth factor over the interval is
  `G_i = product(1 + r_i,d)` over its daily simple returns. The cash growth
  factor is `1`.
- Gross turnover is
  `sum(abs(applied_weight_i - pre_trade_weight_i))`, including cash.
  One-way turnover is half gross turnover. Traded notional is
  `pre_trade_NAV * gross_turnover`.
- Transaction fee is `traded_notional * transaction_cost_rate`.
  Slippage cost is `traded_notional * slippage_bps / 10_000`. Costs are
  deducted once at rebalance:
  `post_cost_NAV = pre_trade_NAV - fees - slippage_cost`.
- Interval ending NAV is
  `post_cost_NAV * sum(applied_weight_i * G_i)`, including the cash term.
  End weights are the marked component values divided by ending NAV.
- Reward is exactly the net log NAV return:
  `log(ending_NAV / pre_trade_NAV)`.
  Reward is dimensionless (log return); NAV and cost amounts use synthetic
  account units. There are no additional reward penalties or bonuses.
- Invalid returns below -100%, non-finite inputs, non-positive VIX, or
  non-positive starting capital are rejected. If costs or returns produce
  non-positive/non-finite NAV, `FloatingPointError` is raised rather than
  substituting a success-shaped value.

## Automated tests and actual results

Executed from the repository root with its `.venv` interpreter:

```text
python -m unittest discover -s tests\rl -v
Ran 35 tests in 12.707s
OK
```

The run included 13 new synthetic-environment tests and 22 existing RL
environment tests. Coverage includes:

- Gymnasium environment checker and reset/observation contracts.
- Action projection, clipping, invalid/non-finite actions, and constraints.
- Hand-calculated NAV, transaction costs, turnover, and log reward.
- Final partial-period termination and rejection of post-terminal steps.
- Synthetic-only input metadata, invalid returns, and unaligned inputs.
- Same-seed reset reproducibility and rendering.

Pylance syntax checks reported no syntax errors for the new environment or test
file. The selected Pylance interpreter reported unresolved Gymnasium/NumPy
imports, while the repository `.venv` successfully imported them and executed
the test suite; no Pylance type-check pass is claimed. The `.venv` did not have
`pytest`, so the relevant suites were run using their `unittest` runner instead.

## Known limitations

- No synthetic data generator is implemented here. Callers must provide the
  synthetic arrays and source metadata; metadata is a declared contract, not
  independent proof of how arrays were produced.
- This is a return-based simulator, not an OHLC/price/share execution model.
  It does not model distributions, corporate actions, spread, market impact,
  borrow, taxes, fractional-share rounding, or cash yield.
- The final partial interval is consumed as-is. The environment does not
  randomize starting points, parallelize scenarios, or perform train/validation
  splits.
- `terminated` reflects data exhaustion only. `truncated` remains false because
  no separate time-limit wrapper is built into this class.
- The reward is intended for code-path validation on synthetic fixtures only;
  PPO integration, risk reporting, dashboard/API integration, model artifact
  handling, and any strategy-quality assessment are out of scope.

## Remaining blockers

1. Historical market data remains NO-GO pending independent resolution of
   provenance, licensing, price-adjustment methodology, and corporate-action
   coverage.
2. The synthetic data generator and scenario-split pipeline described in
   Phase 5.11 must be implemented and tested before generated scenarios can
   drive this environment beyond hand-built fixtures.
3. Align the selected editor/Pylance Python interpreter with the project's
   `.venv` before treating editor import diagnostics as representative of the
   tested environment.
4. PPO training and historical evaluation were not performed and require
   separate authorization/gates. No synthetic output may be described as real
   investment performance.
