# Phase 2: India-Specific Daily Preprocessing and Portfolio Environment

Phase 2 consumes the validated output from Phase 1. It does not download data,
train PPO, place trades, or use the synthetic or US-specific environment.

## Input contract

The preprocessing module requires:

- A Phase 1 validated CSV with `Date`, `Symbol`, `Open`, `High`, `Low`,
  `Close`, `Adjusted Close`, and `Volume`.
- A Phase 1 JSON manifest with `training_ready=true`.
- A configured, non-empty Indian ETF symbol universe.
- The manifest symbol set matching the configured universe.
- Explicit and non-ambiguous adjusted-price methodology.

The manifest must not be bypassed to mark unvalidated or unapproved data as
training-ready.

## Preprocessing

The pipeline:

1. validates schema and numeric values;
2. rejects missing, duplicate, invalid, non-finite and inconsistent OHLCV rows;
3. pivots by date and configured symbol;
4. intersects the configured symbols to common sessions;
5. removes incomplete sessions instead of forward-filling or inventing prices;
6. calculates simple adjusted-close returns;
7. creates 21-session trailing return and volatility features; and
8. applies chronological train, validation and test splits.

The split feature bounds are stored in `IndiaDailyDataset.splits`. Validation
and test data do not participate in scaler fitting.

`IndiaFeatureScaler` fits means and standard deviations only on the training
feature rows. Validation and test environments receive that unchanged scaler.

## Environment

`IndiaDailyPortfolioEnv` is a separate Gymnasium environment:

- Action: asset weights followed by cash, shape `(number_of_assets + 1,)`.
- Raw actions are finite, clipped to `[-1, 1]`, projected onto the simplex,
  and constrained by asset and cash limits.
- Initial state is all cash.
- At session `t`, the observation uses completed prices through `t`.
- The next session open is used for simulated execution.
- The next session close marks holdings and NAV.
- No market prices are forward-filled.
- Rewards are `log(next_nav / current_nav)`.
- Fees and slippage are configurable and are deducted during the transition.
- Termination occurs at the end of the selected partition.

Observations contain trailing standardized features, current asset/cash
weights, NAV ratio, drawdown and remaining episode fraction.

Step information contains NAV, daily return, requested/applied weights,
turnover, fees, slippage, total cost and termination status.

## Corporate-action limitations

No corporate-action ledger is assumed in this phase. `Adjusted Close` is used
only when the Phase 1 manifest has a known approved adjustment methodology.
This does not verify corporate-action completeness. Results must not be
described as total-return accurate without an independent corporate-action
provider and event verification.
