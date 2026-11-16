# Phase 5.11 - RL Prototype Architecture and Synthetic Data Feasibility

**Status:** Architecture only. No production code, model, trade, or historical
backtest is created by this phase.

## Scope and safety boundary

The prototype described here is for software development and controlled
unit/integration tests only. It uses generated data and must not be used to
claim real-market performance, validate an investment strategy, or support
investment decisions. Historical PPO training and real-market backtesting
remain **NO-GO** until provenance, licensing, price-adjustment methodology,
and corporate-action coverage are independently resolved.

The fixed investable universe is `SPY, QQQ, IEF, GLD, XLU, XLV`. `^GSPC` and
`^VIX` are synthetic context series, not investable assets. The six assets,
benchmark, and volatility index must all carry synthetic source metadata when
used by the prototype.

No `data/fmp_daily/` CSVs, `.env` contents, market-data APIs, downloads,
historical backtests, PPO training, or trade execution are part of this design
or assessment.

## 1. Existing repository assessment

### Structure and relevant components

The repository is a Python project. Relevant areas are:

| Area | Existing contents and assessment |
|---|---|
| `src/rl/environment/` | An uncommitted Gymnasium `PortfolioTradingEnv`, environment README, and an uncommitted focused test suite already exist. The environment fixes the six-ETF universe plus `^GSPC`/`^VIX`, has a seven-element asset-plus-cash action, feature scaling, date partitions, and next-session OHLC execution. Its loader and fixed train/validation/test dates are designed around market CSVs, so do not use that loader or its historical partitions for this phase. |
| `src/strategies/allocation/` | An uncommitted deterministic allocation package provides target-weight validation, cash reserves, per-asset caps, turnover limits, simulated costs/slippage, and auditable results. Its constraints and tests are useful prior art. Keep synthetic execution isolated; do not connect it to brokerage execution. |
| `src/data/` | Existing fetch, storage, and processing modules are oriented to external or historical data. Do not route synthetic fixtures through a fetcher or the existing unresolved CSV inputs. |
| `src/backtest/` | `BacktestEngine` is a historical backtesting component and imports `bt` and a data-fetching helper. It is not the synthetic environment or its evaluation runner. |
| `src/web/` | `src/web/app.py` is a Streamlit dashboard with overview, data management, backtesting, live-trading, portfolio-analysis, and settings views. It imports data-store and trading components directly. `src/web/components.py` is the accompanying UI component module. |
| `src/trading/` | Contains brokerage/trade-execution integrations. The synthetic prototype must not import or call these modules. |
| `tests/` | Existing allocation and RL test directories provide patterns for focused tests. The RL fixture constructs controlled prices in memory rather than requiring an external feed. |
| `docs/` | Contains preceding phase audits; their unresolved provenance and corporate-action blockers govern this proposal. |

### Reusable code and boundaries

- Reuse the fixed symbol ordering and the idea of projecting continuous PPO
  actions onto a seven-way simplex (six assets plus cash) from
  `src/rl/environment/portfolio_env.py`. Keep the synthetic input path separate
  from that module's market CSV loader and date partitions.
- Reuse and/or call the allocation validation concepts in
  `src/strategies/allocation/constraints.py` for long-only weights, cash reserve,
  asset caps, and deterministic normalization. The current rebalance engine
  assumes price inputs and simulated fills; do not make real-data claims based
  on it.
- Reuse testing patterns in `tests/rl/` and `tests/strategies/allocation/`.
- Treat `src/data/data_processor.py`, `src/backtest/backtest_engine.py`,
  data-fetch/store utilities, and `src/trading/` as out of scope for synthetic
  source generation and execution.

### Dependencies and compatibility considerations

- The project declares Python `>=3.11`. NumPy and pandas are already in
  `requirements.txt`.
- Gymnasium and Stable-Baselines3 are declared under the `ml` extra in
  `setup.py` (`gymnasium>=0.29.0`, `stable-baselines3>=2.1.0`). They are not
  listed as general requirements. Install the project `ml` extra in an
  isolated development environment before implementing/running RL code.
- `torch` is listed in `requirements.txt` and also appears in optional setup
  extras; dependency resolution should be checked in the project's chosen
  environment before adding or upgrading packages. Do not install dependencies
  as part of this architecture-only phase.
- Keep Gymnasium's five-value `step()` return contract and seeded `reset()`
  contract consistent with the installed Gymnasium/SB3 versions. No REST
  framework is currently declared; the current dashboard is Python/Streamlit,
  not a separate API service.

## 2. Proposed architecture

Keep the prototype as a small, isolated Python service layer, independent of
market-data acquisition, historical evaluation, and live-trading modules:

```text
src/rl/
  synthetic_data.py              # seeded generator and provenance metadata
  environment/
    synthetic_portfolio_env.py   # Gymnasium env driven only by generated returns
  risk.py                        # pure risk/turnover calculations
  training.py                    # future SB3 construction/save/load (not run now)
  service.py                     # typed application-facing read/decision methods
src/web/
  app.py                         # future Streamlit synthetic dashboard page
tests/rl/
  test_synthetic_data.py
  test_synthetic_portfolio_env.py
  test_risk.py
  test_training_artifacts.py
```

These are proposed locations, not files created by this phase. The environment
receives generated arrays/DataFrames in memory; it must not discover or load
files from `data/`. The service exposes calculations and model state to the UI
without importing data-fetching, backtest, or trading modules. Keep a future
REST API optional; the contracts in Section 7 can initially be Python
dataclasses/functions passed directly to Streamlit.

### Portfolio and account conventions

- Order assets exactly as `SPY, QQQ, IEF, GLD, XLU, XLV`; use cash as the
  seventh allocation component. `^GSPC` and `^VIX` are context observations
  only and never receive portfolio weights.
- Start with **$100,000** synthetic NAV. NAV is a unitless simulation balance,
  not a funding recommendation or a real account.
- Long-only, unlevered portfolio: all asset weights and cash weight are
  non-negative and sum to 1. No shorting, borrowing, margin, derivatives, or
  fractional brokerage-order execution.
- Cash is explicit, earns **0%** in the baseline fixture, and is included in
  normalization, turnover, NAV, and observations. A future configurable cash
  yield must be clearly synthetic and must not be mistaken for a risk-free
  market rate.
- Starter test constraints: per-asset cap **35%**, minimum cash reserve
  **2%**, gross turnover cap **100% per rebalance**. These are engineering
  fixtures, not calibrated investment limits; keep them configurable and test
  their enforcement.
- Starter cost assumptions: transaction fee **10 bps** of traded notional and
  slippage **5 bps** of traded notional. These values are arbitrary synthetic
  test parameters, not estimates of actual ETF trading costs.

## 3. Portfolio environment design

Implement a separate `SyntheticPortfolioEnv(gymnasium.Env)` that consumes a
validated generated return dataset and generator metadata. It must not read
historical price files.

### Observation space

Use a flat finite `float32` vector suitable for SB3 `MlpPolicy`. A proposed
fixed feature order is:

1. For each of six ETFs: latest one-period return, trailing 5- and 21-session
   returns, and trailing 21- and 63-session realized volatility (30 values).
2. `^GSPC`: latest return and trailing 21-session volatility (2 values).
3. `^VIX`: synthetic level scaled by the generator's documented level scale
   (1 value).
4. Current six ETF weights and cash weight (7 values).
5. Current NAV divided by initial NAV, peak-to-current drawdown, and fraction
   of the synthetic episode remaining (3 values).

That yields **43 features**. Validate shape, finiteness, and matching
`observation_space` bounds. Use only trailing windows available at the decision
point. Define warm-up handling in generated fixtures (for example, generate a
warm-up prefix and exclude it from decision steps); never fill missing
features from future data. If scaling is used, fit it only on synthetic
training scenarios and reuse it unchanged for synthetic validation/test
scenarios.

### Action space and weight conversion

- `Box(low=-1.0, high=1.0, shape=(7,), dtype=float32)` gives PPO a continuous
  action compatible with the existing simplex-projection pattern.
- Reject wrong shape and non-finite values. Clip finite out-of-range direct
  inputs to action-space bounds, record that clipping in `info`, and project
  the resulting vector deterministically onto the non-negative unit simplex.
- Map the first six components to the fixed ETF order and the seventh to cash.
  Then apply cash-reserve, asset-cap, and turnover constraints; report requested
  and applied weights and any adjustment reasons in `info`.
- Never silently replace an invalid action with a favorable allocation. Raise a
  clear validation error for malformed/non-finite actions.

### Timing, returns, costs, and reward

- One environment decision step represents one **monthly rebalance period**
  (21 synthetic trading sessions by default). At each decision boundary, use
  only information through the just-completed session; apply the action and
  charge modeled costs once, then hold target weights over the following
  period. The last period may be shorter.
- Generate synthetic daily asset simple returns from the log-return process.
  Compound each asset's daily returns over the interval. Do not fabricate
  OHLC, volume, dividends, or corporate actions to resemble historical
  records.
- Gross turnover fraction is
  `sum(abs(target_weight - pre_trade_weight))`, including cash. One-way
  turnover fraction is half the gross fraction. Traded notional is gross
  turnover fraction times pre-trade NAV.
- Cost fraction is `turnover_fraction * (transaction_cost_rate +
  slippage_bps / 10_000)`. Deduct the cost exactly once at rebalance; allocate
  post-cost NAV across the target weights and then apply the interval's
  compounded returns. No second reward-level transaction-cost penalty is
  applied.
- Reward is the net log NAV change over the interval:
  `log(NAV_after_interval / NAV_before_rebalance)`. NAV must be positive and
  finite. Record gross return, costs, net return, turnover, and NAV in `info`
  for deterministic tests and charts.

### Constraints and episode end

After simplex projection, enforce the stated long-only, full-investment,
per-asset cap, cash-reserve, and gross-turnover limits deterministically.
Reject infeasible configuration at environment construction (for example,
minimum cash plus asset caps that cannot sum to the required total). Check
normalization after every projection/constraint step within a documented
floating-point tolerance.

End an episode when the generated scenario has no further full/partial return
period. Natural generated-data exhaustion sets `terminated=True`; an externally
imposed step limit, if added, sets `truncated=True`. A zero/non-finite NAV is an
explicit terminal failure with finite diagnostic output, not a silently
substituted portfolio value. `reset(seed=...)` must reset the RNG and all
portfolio state reproducibly.

## 4. PPO design

Do not instantiate, train, or evaluate PPO in this phase. When implementation
is approved, use Stable-Baselines3 PPO with the custom Gymnasium environment:

- **Policy:** `MlpPolicy`; two fully connected hidden layers of 128 units each
  with ReLU activations. Normalize observations through the training-scenario
  scaler only; do not normalize actions after the environment's simplex
  conversion.
- **Starter configuration:** `n_steps=1024`, `batch_size=256`, `gamma=0.99`,
  `gae_lambda=0.95`, `learning_rate=3e-4`, `clip_range=0.2`,
  `ent_coef=0.0`, `vf_coef=0.5`, `max_grad_norm=0.5`, and a recorded fixed
  seed. Confirm `batch_size` compatibility with chosen `n_steps` and number of
  vectorized environments before training.
- **Environment wrappers:** use SB3 `Monitor`; use `DummyVecEnv` for a simple
  local prototype. Parallel vectorization is unnecessary until its seed and
  reproducibility behavior are tested.
- **Evaluation:** use separately seeded, held-out **synthetic** scenarios and
  deterministic policy inference. Compare software outputs against a
  deterministic equal-weight and all-cash baseline to validate plumbing only.
  Do not describe any synthetic return, reward, Sharpe ratio, or comparison as
  investment performance.
- **Saving/loading:** store model artifacts outside tracked source (for example
  an ignored `models/synthetic/` directory). Save a sidecar manifest containing
  `synthetic=true`, generator schema/version, seed(s), asset order, environment
  and constraint configuration, scaler parameters, PPO config, Python and
  dependency versions, and model checksum. Loading must validate manifest
  compatibility and fail explicitly if model or metadata is missing, corrupt,
  or incompatible. Never load a synthetic model into live-trading components.

## 5. Synthetic data methodology

Provide a deterministic generator API such as
`generate_synthetic_scenarios(config, seed) -> SyntheticDataset`. The returned
object should contain dates or integer session indices, daily simple returns,
synthetic `^GSPC` returns, a synthetic positive `^VIX` level series, regime
labels, and metadata. Metadata must include `source_type="synthetic"`,
`synthetic=true`, generator/schema version, seed, configuration hash, and
generation timestamp. Use a fixed logical calendar (or integer session index);
dates are labels only and must not imply real observations.

### Return and regime generation

- Generate a joint daily **log-return** vector for the six ETF symbols and
  `^GSPC` from a multivariate normal distribution for the active regime. For
  regime `k`, configure annualized log-drift vector `mu_k`, annualized
  volatility vector `sigma_k`, and a 7-by-7 correlation matrix `C_k`. Convert
  to daily parameters with `mu_k / 252` and
  `diag(sigma_k / sqrt(252)) * C_k * diag(sigma_k / sqrt(252))`.
  Convert log returns to simple returns with `exp(log_return) - 1`.
- Validate matching dimensions, finite values, non-negative volatilities,
  symmetric correlation matrices, unit diagonal, entries in `[-1, 1]`, and
  positive-semidefinite covariance (within numerical tolerance). Reject invalid
  configurations; do not silently repair a materially invalid matrix.
- Support named, user-configurable synthetic regimes such as `calm`,
  `volatile`, and `stress`, with explicit per-regime parameters and a valid
  row-stochastic Markov transition matrix. Names describe generator states,
  not historical episodes. Store sampled regime labels with every output row.
- Generate a positive synthetic `^VIX` level independently using configurable
  regime-specific mean-reverting dynamics in log-level space, with a floor,
  ceiling, and documented parameters. It is an observation-only synthetic
  index, not actual volatility-index history.
- Construct index levels from a configurable positive starting value and
  cumulative products. Do not generate volume, OHLC, distributions, corporate
  actions, or any fields that could be mistaken for observed market data.

### Reproducibility and split discipline

Use only `numpy.random.Generator` created from the caller's seed. Ensure the
same config and seed produce identical returns, regimes, and index levels.
Keep training/validation/test synthetic scenarios separated by distinct seeds
and generator outputs; fit transformations on training scenarios only. Persist
the configuration and seed in test logs/artifacts. All UI labels and exported
fixtures must identify the data as synthetic.

## 6. Risk engine specifications

Place pure calculations in `src/rl/risk.py`; accept finite, aligned arrays or
Series and return explicit `None`/JSON `null` for undefined metrics. Do not
silently convert invalid or insufficient inputs into plausible-looking zeros.
Assume 252 synthetic trading sessions per year by default; make the annualizer
explicit in function arguments/metadata.

Let periodic simple returns be `r_t`, NAV be `V_t`, and target/pre-trade
weights include the six ETFs plus cash:

| Metric | Definition | Edge cases |
|---|---|---|
| Cumulative return | `prod(1 + r_t) - 1`, equivalently `V_T / V_0 - 1` | Require at least one finite return and positive starting NAV. Reject `r_t < -1`. |
| Annualized volatility | `sample_std(r_t, ddof=1) * sqrt(periods_per_year)` | Require at least two finite observations. Return null for fewer observations; a constant series returns 0. |
| Sharpe ratio | `(mean(r_t - rf_t) / sample_std(r_t - rf_t, ddof=1)) * sqrt(periods_per_year)` | Default synthetic `rf_t=0`. Require at least two observations; return null when excess-return volatility is zero or non-finite. Do not present this as a forward estimate. |
| Maximum drawdown | `max_t((running_peak(V)_t - V_t) / running_peak(V)_t)` for positive NAV | Require positive finite NAV series. Return a non-negative fraction; no drawdown yields 0. |
| Gross turnover | `sum_i(abs(w_target_i - w_pretrade_i))` per rebalance | Require aligned finite weights. Include cash; report both gross and one-way (`gross / 2`) turnover. |
| Transaction cost | `traded_notional * (fee_rate + slippage_bps / 10_000)`; `traded_notional = NAV_pre * gross_turnover` | Rates must be finite and non-negative; zero turnover gives zero cost. Report fee and slippage components separately and deduct once. |

Validate no nulls, infinities, misaligned dates, invalid weights, zero/negative
NAV, or returns below -100%. For no-return episodes or insufficient sample
length, preserve undefined values rather than presenting them as zero.

## 7. Dashboard and API design

### Existing UI and proposed integration

`src/web/app.py` is the existing Streamlit frontend and currently includes
historical backtest and live-trading views. There is no separate REST backend
or API framework in the inspected project dependencies. Add a dedicated
synthetic prototype page in a future implementation phase; do not change the
existing historical or live-trading screens as part of this architecture
phase. Initially, the page can call typed in-process service functions. If a
REST boundary is later needed, expose equivalent versioned JSON contracts.

The synthetic page must have a persistent high-visibility banner:

> **SYNTHETIC DATA — software testing only. Not historical market data and not
> investment performance.**

Display source type, generator version, seed, scenario, and timestamp near
every allocation chart/table, rebalance result, risk-metric panel, and export.
Never use green/red performance styling or benchmark language that could
suggest real returns without this label. Disable order/trading controls on this
page; no synthetic allocation can be sent to a broker.

### Proposed API contracts

All responses include `schema_version` and `data_source`. These are proposed
contracts and do not imply that routes currently exist.

`GET /api/v1/portfolio/allocations`

```json
{
  "schema_version": "1.0",
  "data_source": {
    "type": "synthetic",
    "synthetic": true,
    "scenario_id": "scenario-001",
    "seed": 42,
    "generator_version": "1"
  },
  "currency": "SIM",
  "nav": 100000.0,
  "weights": {
    "SPY": 0.2, "QQQ": 0.2, "IEF": 0.2,
    "GLD": 0.2, "XLU": 0.1, "XLV": 0.05, "CASH": 0.05
  },
  "as_of_step": 20
}
```

`POST /api/v1/rebalancing/decisions` accepts seven requested weights or a
seven-value action plus current portfolio state and scenario/step identifiers.
Validate the action server-side; never trust client-normalized weights.

```json
{
  "schema_version": "1.0",
  "data_source": {"type": "synthetic", "synthetic": true, "scenario_id": "scenario-001", "seed": 42},
  "decision_id": "decision-0001",
  "requested_weights": {"SPY": 0.2, "QQQ": 0.2, "IEF": 0.2, "GLD": 0.2, "XLU": 0.1, "XLV": 0.05, "CASH": 0.05},
  "applied_weights": {"SPY": 0.2, "QQQ": 0.2, "IEF": 0.2, "GLD": 0.2, "XLU": 0.1, "XLV": 0.05, "CASH": 0.05},
  "gross_turnover": 0.4,
  "one_way_turnover": 0.2,
  "transaction_cost": {"fee": 40.0, "slippage": 20.0, "total": 60.0, "currency": "SIM"},
  "constraint_adjustments": [],
  "execution_status": "simulated_only"
}
```

`GET /api/v1/portfolio/risk-metrics?scenario_id=...`

```json
{
  "schema_version": "1.0",
  "data_source": {"type": "synthetic", "synthetic": true, "scenario_id": "scenario-001", "seed": 42},
  "periods_per_year": 252,
  "cumulative_return": 0.012,
  "annualized_volatility": 0.08,
  "sharpe_ratio": null,
  "maximum_drawdown": 0.03,
  "gross_turnover": 0.4,
  "one_way_turnover": 0.2,
  "undefined_reasons": {"sharpe_ratio": "zero_excess_return_volatility"}
}
```

Numbers above illustrate payload shape only; they are not results. Undefined
metrics are JSON `null` with a reason.

`GET /api/v1/training/status`

```json
{
  "schema_version": "1.0",
  "status": "not_started",
  "training_allowed": false,
  "model_id": null,
  "data_source": {"type": "synthetic", "synthetic": true},
  "current_step": 0,
  "total_steps": null,
  "message": "Training is not started in Phase 5.11."
}
```

`GET /api/v1/data-source/status`

```json
{
  "schema_version": "1.0",
  "prototype_data_status": "synthetic_only",
  "active_source": "synthetic",
  "synthetic": true,
  "historical_approved": false,
  "historical_blockers": [
    "provenance", "licensing", "adjustment_methodology", "corporate_actions"
  ],
  "trade_execution_enabled": false
}
```

The UI must distinguish `synthetic_only`, `historical_unapproved`, and
`historical_approved`. Approval must come from an explicit future governance
decision with evidence; it must never be inferred from a CSV's presence,
filename, or source label. In this phase, only `synthetic_only` is permitted.

## 8. Testing strategy

Use deterministic in-memory synthetic fixtures; tests must not load market CSVs,
call data providers, start PPO training, or invoke the historical backtest or
trade-execution modules.

### Unit tests

- **Weight normalization:** simplex projection is deterministic, finite,
  non-negative, and sums to 1; malformed shape and NaN/Inf fail clearly.
- **Portfolio constraints:** enforce six-asset ordering, long-only weights,
  total-weight invariant, 35% cap, 2% minimum cash, and turnover cap; reject
  infeasible configs; assert every adjustment is reported.
- **Action validation:** test correct seven-dimensional actions, clipping
  reporting, wrong dimensions, NaN/Inf, and boundary values.
- **Reward calculations:** compare hand-computed compounded returns and
  log-NAV reward; ensure costs are deducted once and reward is net of costs.
- **Transaction costs:** verify zero-turnover cost is zero, fee/slippage
  components match formulas, and increased turnover increases costs under
  equal assumptions.
- **Risk metrics:** test known cumulative return, annualized volatility, Sharpe,
  drawdown, gross/one-way turnover, empty/one-point arrays, zero volatility,
  invalid returns/NAV, and null/reason behavior.
- **Synthetic reproducibility:** same seed/config yields identical asset and
  benchmark returns, VIX levels, regimes, and metadata-relevant config hash;
  different seeds yield different stochastic paths. Check correlation and
  volatility parameter effects statistically with tolerances, not exact
  values.
- **Model loading:** use a tiny locally created test artifact only if model
  code is later implemented; verify round trip, missing/corrupt artifact,
  metadata mismatch, and synthetic-only provenance rejection. No training
  test in this phase.

### Integration tests

- Run one short seeded synthetic episode end-to-end from generator through
  environment, allocation constraints, risk functions, and API/service payload
  serialization. Assert the exact `synthetic=true` status survives every
  layer.
- Repeat the same episode with the same seed and action sequence; compare
  observations, rewards, costs, weights, regime IDs, and terminal state.
- Verify separate synthetic seeds/scenarios do not share fitted transforms or
  leak validation/test rows into training-only scaling.
- Verify model metadata and all dashboard-facing result contracts reject
  absent or non-synthetic provenance in the prototype path.
- Add a static/import boundary test where practical: the synthetic generator
  and environment must not depend on data fetchers, historical backtest
  modules, `.env` settings, or trading execution modules.

## 9. Implementation sequence

1. **Lock contracts:** implement typed config, fixed symbol order, source
   metadata, generator schema, and JSON/service payload models.
2. **Build synthetic generator:** validate covariance/regime config, seed all
   randomness explicitly, generate the six assets plus synthetic benchmark and
   VIX series, and add deterministic tests.
3. **Implement environment:** add the 43-feature observation, seven-value
   action, constraints, monthly stepping, costs, net log reward, reproducible
   reset, and termination behavior; test against hand-computed fixtures.
4. **Implement pure risk functions:** add formulas and explicit undefined
   results, then test edge cases.
5. **Add PPO integration only after the above tests pass:** construct SB3 PPO
   and model artifact helpers; first test configuration and serialization with
   a minimal test artifact. Training on synthetic scenarios requires a
   separately approved follow-on task.
6. **Add service/API boundary:** expose typed allocation, decision, risk,
   training-status, and data-status responses. Keep all source metadata.
7. **Add isolated Streamlit page:** show synthetic warnings and source status
   throughout; do not link the page to live-trading actions.
8. **Run focused validation:** generator/environment/risk/API tests and
   Gymnasium environment checks; verify no path reaches historical CSVs,
   external providers, backtest engine, or brokerage modules.

This sequence does not authorize historical ingestion, PPO fitting, or
real-market evaluation. Each is a separate gated decision.

## 10. Remaining blockers

1. **Historical data provenance and licensing:** the preceding phases have not
   cleared existing market CSVs for research or redistribution. Historical
   loading, training, and performance claims remain blocked.
2. **Price and corporate-action methodology:** split/dividend adjustments,
   event completeness, and total-return calculation are unresolved. Do not
   describe any historical return as accurate or complete.
3. **Dependency environment:** Gymnasium/SB3 are optional extras rather than
   universal project dependencies; test their supported versions and install
   configuration in an isolated environment before code implementation.
4. **API hosting choice:** no separate REST backend is currently present.
   Decide later whether in-process Streamlit services are sufficient or a
   separately deployed API is justified.
5. **Constraint/cost parameters:** the starter values in this document exist
   only to make software tests concrete. Product owners must approve any
   different simulation assumptions; none should be represented as empirical.
6. **Model governance and artifact location:** define retention, review, and
   authorization policy before any future training workflow. Synthetic model
   files must remain isolated from live trading and approved-historical
   workflows.
7. **Future approval gate:** a separate review must verify provenance,
   licensing, adjustment methodology, corporate actions, benchmark alignment,
   and leakage controls before historical market data can be considered.

**Phase 5.11 exit condition:** architecture and test contracts are documented;
no production code was changed, and no market-data or investment-performance
claim is made.
