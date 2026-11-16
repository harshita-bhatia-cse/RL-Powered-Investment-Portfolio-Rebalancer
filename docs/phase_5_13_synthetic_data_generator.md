# Phase 5.13 - Reproducible Synthetic Market Data Generator

**Status:** Implemented for software development and controlled tests only.
Generated values are synthetic and are not estimates, observations, or claims
about real market behavior or investment performance.

## Files created or modified

- Created `src/rl/synthetic_data.py`.
- Modified `src/rl/__init__.py` to export the configuration, dataset, and
  generation APIs.
- Created `tests/rl/test_synthetic_data.py`.
- Created this report.

No existing portfolio-environment accounting code was modified. No market
files, `.env`, data providers, downloads, PPO training, historical backtests,
or trade execution were used.

## 1. Generation methodology and equations

`generate_synthetic_market(config, seed=..., scenario_id=...)` generates one
aligned scenario using only `numpy.random.default_rng(seed)`. It produces six
ETF simple-return columns, one synthetic benchmark return series, one positive
synthetic VIX level series, regime labels, and provenance/configuration
metadata. It does not create prices, OHLCV, dividends, or corporate actions.

For each session, a Markov regime state selects the applicable regime
parameters. For the six ETFs and benchmark, in fixed order
`SPY, QQQ, IEF, GLD, XLU, XLV, ^GSPC`, the generator samples a multivariate
normal **log-return** vector:

```text
z_t ~ N(0, I)
log_return_t = mu_k / 252 + D_k / sqrt(252) * C_root * z_t
simple_return_t = exp(log_return_t) - 1
```

`k` is the current regime; `mu_k` is its annualized log-return drift vector;
`D_k` is diagonal with annualized volatilities; `C_root` is an eigen-based
square root of the configured correlation matrix. Thus daily covariance is
`D_k C D_k / 252`. Correlation is configurable and shared across regimes;
regime-specific drifts and volatilities vary.

For finite numerical safety, sampled log returns are clipped to `[-20, 20]`
before conversion. The minimum produced simple return is therefore
`exp(-20) - 1`, strictly greater than -1, and all generated simple returns
remain finite. This is a numerical safeguard, not a financial model of
extreme-market behavior.

VIX is generated separately as a positive, regime-targeted mean-reverting
log-level process:

```text
v_t = clip(v_(t-1) + kappa * (log(target_k) - v_(t-1)) + epsilon_t,
           log(vix_min), log(vix_max))
epsilon_t ~ Normal(0, vix_annualized_log_volatility_k^2 / 252)
VIX_t = exp(v_t)
```

The per-step innovation is clipped by the configured `vix_log_level_clip`.
Initial level, bounds, mean-reversion speed, and regime targets/volatilities
are configurable. This series is a synthetic observation feature only.

## 2. Asset parameters and correlation assumptions

`SyntheticMarketConfig.regimes` holds one `RegimeConfig` per state. Each
`annualized_log_returns` and `annualized_volatilities` tuple has seven values:
the six ETFs followed by `^GSPC`. Values below are the default configuration;
all are **illustrative generator parameters**, not empirical estimates.

| Regime | Series order | Annualized log-return drift | Annualized volatility | Synthetic VIX target |
|---|---|---|---|---:|
| Bull | SPY, QQQ, IEF, GLD, XLU, XLV, ^GSPC | 0.12, 0.16, 0.025, 0.04, 0.07, 0.08, 0.10 | 0.18, 0.27, 0.07, 0.18, 0.16, 0.15, 0.17 | 16 |
| Bear | SPY, QQQ, IEF, GLD, XLU, XLV, ^GSPC | -0.18, -0.25, 0.04, -0.08, -0.10, -0.08, -0.20 | 0.35, 0.48, 0.12, 0.32, 0.28, 0.25, 0.32 | 38 |
| Sideways | SPY, QQQ, IEF, GLD, XLU, XLV, ^GSPC | 0.04, 0.05, 0.025, 0.02, 0.04, 0.05, 0.04 | 0.16, 0.23, 0.06, 0.15, 0.14, 0.13, 0.15 | 21 |

The default 7-by-7 correlation matrix has ones on its diagonal and **0.25 on
every off-diagonal entry**, including benchmark correlations. This equicorrelation
matrix is a transparent, positive-definite placeholder. It is not intended to
represent ETF/benchmark correlations.

Configuration validation rejects incorrect dimensions, non-finite values,
asymmetry, non-unit diagonal entries, correlations outside `[-1, 1]`, or a
minimum eigenvalue below `-1e-10`. Semidefinite matrices within that numerical
tolerance are accepted; negative eigenvalues within tolerance are floored to
zero when computing the square root.

## 3. Market regimes and transition mechanism

Regime order is fixed as `bull, bear, sideways`. The default initial
probabilities are `(0.50, 0.20, 0.30)`. The default row-stochastic transition
matrix, with rows and columns in that order, is:

```text
          next bull  next bear  next sideways
bull         0.94       0.02         0.04
bear         0.04       0.92         0.04
sideways     0.08       0.08         0.84
```

The first session's regime is sampled from the initial probabilities. Each
subsequent session's regime is sampled from the transition-matrix row
corresponding to the previous session. Transition and initial probabilities
are configurable, checked for values in `[0, 1]`, and must sum to one (rows
for the transition matrix).

The regime names and parameters describe states of this simulator only; they
are not labels for historical periods.

## 4. Random seed and metadata

- Pass an integer seed in `[0, 2**64 - 1]`; booleans and invalid seeds are
  rejected. No global NumPy random state is changed.
- The same configuration and seed reproduce asset returns, benchmark returns,
  VIX levels, regime labels, and configuration hash within the same supported
  NumPy/runtime implementation.
- A different seed produces another pseudo-random path.
- Returned arrays are marked read-only to reduce accidental mutation.
- Dataset metadata includes `source_type="synthetic"`, `synthetic=true`,
  generator version, scenario ID, seed, full generation configuration,
  configuration SHA-256, series/asset/regime ordering, and a UTC generation
  timestamp.

The UTC generation timestamp records when a generation call ran and therefore
differs between calls. It is provenance metadata, not part of the reproducible
stochastic path. Compare the arrays, labels, seed, and configuration hash when
checking deterministic generation.

## 5. Train, validation, and test scenario separation

`generate_synthetic_splits(config, seed=..., sessions=...)` derives one
deterministic seed per split using NumPy `SeedSequence` with stable split
indices for `train`, `validation`, and `test`. Each split is independently
generated, has its own scenario ID and seed, and receives the same
configuration (or the same explicit session-count override). Each invocation
with the same root seed and config reproduces the same three paths and split
identities.

Splits are **independent stochastic scenarios**, not chronological slices of
one larger path and not proxies for historical train/validation/test dates.
They have separate arrays and pseudo-random streams; no observations are
copied or shared between split objects. This helps software test split
handling but does not by itself establish statistical independence or prevent
model leakage. Any future scaler must be fit on training scenarios only and
then reused unchanged for validation/test.

## 6. Environment compatibility

`SyntheticDataset.environment_kwargs()` returns the exact keyword inputs
required by Phase 5.12's `SyntheticPortfolioEnv`:

- `asset_returns`: shape `(sessions, 6)` in fixed ETF order.
- `benchmark_returns`: shape `(sessions,)`, representing synthetic `^GSPC`.
- `vix_levels`: shape `(sessions,)`, all finite and strictly positive.
- `data_metadata`: explicitly carries `source_type="synthetic"` and
  `synthetic=true`.

The environment handles its own 63-session observation warm-up. Therefore, a
scenario supplied to it must contain more than 63 sessions (the generator
allows shorter paths for data-generation use, but the environment will reject
them). Integration tests instantiate the environment from a generated dataset
and check its observation space.

## 7. Automated test results

Executed from the repository root with the project's `.venv`:

```text
python -m unittest discover -s tests\rl -v
Ran 46 tests in 13.898s
OK
```

The run covers 11 new generator tests plus 35 environment tests (13 synthetic
environment tests and 22 existing market-environment tests). Generator
coverage includes output shape/order/alignment, finite values,
simple-return lower bound, positive VIX, correlation validation, invalid
regime/transition configuration, regime transition behavior, same-seed
reproducibility, different-seed variation, scenario split seed/path separation,
dataset metadata validation, and direct environment compatibility.

Pylance syntax checks reported no syntax errors for the generator and its test
file. No PPO training, historical backtest, performance evaluation, or trade
execution was run.

## 8. Limitations and remaining blockers

- This is a simple regime-switching multivariate-normal model with a separate
  mean-reverting VIX process. It omits fat tails, jumps, volatility clustering
  within a regime, dynamic correlations, asynchronous sessions, distributions,
  corporate actions, and instrument mechanics.
- Default parameters and transition probabilities are synthetic fixtures and
  have no claim to fit or reproduce real market data or market behavior.
- `^GSPC` is a generated series following the same multivariate process as the
  ETF return dimensions; no real benchmark alignment or index calculation is
  implied.
- The generation timestamp is intentionally wall-clock metadata and is not
  identical across repeat calls; all stochastic outputs and deterministic
  metadata fields are reproducible for a fixed seed/config/runtime.
- Seeded pseudo-random output can vary across NumPy versions or numerical
  libraries; reproducibility should be evaluated with the same recorded
  generator/runtime versions.
- The project-wide historical data blockers remain: provenance, licensing,
  price-adjustment methodology, corporate-action coverage, and associated
  leakage/benchmark validation. This generator does not resolve or bypass
  those gates.
- This phase does not train PPO or assess a strategy. Synthetic outputs are
  strictly for software development and controlled testing, not investment
  performance claims.
