# Phase 5.14 - End-to-End Synthetic Pipeline Integration and Validation

**Scope:** Synthetic software integration only. No PPO training, historical
performance evaluation, or real-market conclusion.

## 1. Integration architecture

`src/rl/synthetic_pipeline.py` provides two small helpers:

- `build_synthetic_pipeline(config, seed)` calls the Phase 5.13 train,
  validation, and test scenario generator and constructs a fresh
  `SyntheticPortfolioEnv` for each dataset using `environment_kwargs()`.
- `run_deterministic_actions(dataset, actions)` creates a fresh environment,
  resets it, executes the supplied action sequence, and collects immutable
  observation snapshots, rewards, info (including NAV and weights), and
  terminated/truncated flags.

The pipeline intentionally has no PPO, model, scaler, market-data loader,
backtest, or broker dependency.

## 2. Files created or modified

- Created `src/rl/synthetic_pipeline.py`.
- Created `tests/rl/test_synthetic_pipeline.py`.
- Modified `src/rl/environment/synthetic_portfolio_env.py` to deep-copy nested
  synthetic metadata when accepting it and when returning `info`. This is a
  Gymnasium API object-isolation fix discovered during integration; accounting
  formulas were not changed.
- Created this report.

Earlier-phase uncommitted changes were preserved.

## 3. Test scenarios

The integration tests use only generated or explicitly hand-built in-memory
synthetic arrays:

1. Generate three distinct 110-session scenarios and create/reset/check all
   three environments.
2. Execute fixed all-cash and balanced action sequences and inspect each
   collected transition.
3. Perturb all inputs from session 63 onward and compare the initial
   observation against an unchanged dataset to check for future-return leakage.
4. Use a hand-calculated 65-session fixture with two 10% SPY returns to verify
   the full rebalance accounting transition and terminal weights.
5. Check scenario seed, metadata, array non-sharing/read-only behavior, and
   that environment instances own copies of caller-provided arrays.
6. Generate identical train/validation/test splits twice and replay identical
   actions to compare every generated path and environment output, while
   explicitly allowing generation timestamps to differ.
7. Mutate nested metadata returned by reset/step and verify it cannot mutate
   environment metadata or data in another response.

## 4. Accounting validation evidence

The hand fixture begins with **100,000** synthetic units entirely in cash and
uses a target of **50% SPY / 50% cash**, with costs of 10 bps fee and 5 bps
slippage. The two consumed daily SPY returns are both 10%.

```text
gross turnover = |0.50 - 0.00| + |0.50 - 1.00| = 1.00
one-way turnover = 1.00 / 2 = 0.50
traded notional = 100,000 * 1.00 = 100,000
fee = 100,000 * 0.001 = 100
slippage = 100,000 * 5 / 10,000 = 50
post-cost NAV = 100,000 - 100 - 50 = 99,850
SPY interval growth = (1 + 0.10)^2 = 1.21
portfolio interval growth = 0.50 * 1.21 + 0.50 * 1 = 1.105
ending NAV = 99,850 * 1.105 = 110,334.25
SPY ending weight = (99,850 * 0.50 * 1.21) / 110,334.25
cash ending weight = (99,850 * 0.50) / 110,334.25
reward = log(110,334.25 / 100,000)
```

The test asserts the exact fee, slippage, total cost, gross/one-way turnover,
post-cost NAV, ending NAV, ending SPY/cash weights, and log-NAV reward.
Matching the expected NAV verifies costs are deducted once, before applying
the compounded interval returns, with no second deduction in reward.

## 5. Scenario isolation evidence

- Train, validation, and test receive distinct derived seeds and scenario IDs.
- Their asset, benchmark, and VIX arrays do not share memory and are
  read-only in generated datasets.
- All three scenarios preserve the same configuration hash for an unchanged
  configuration.
- Each environment receives its corresponding dataset metadata and copies
  caller arrays, so later caller-array mutation does not alter environment
  inputs.
- No scaler or preprocessing step is present in the integration layer.
- Nested metadata returned from `reset()` and `step()` is isolated from
  environment state and from later responses.

## 6. Reproducibility results

Two generation runs with the same root seed and configuration produced
identical asset returns, benchmark returns, VIX values, regime labels, split
seeds, scenario IDs, and configuration hashes. Replaying the same actions
produced identical initial/step observations, rewards, NAV values, weights,
and termination flags for all three splits.

The UTC `generated_at_utc` metadata field differed between generation calls as
expected; it is provenance time, not stochastic output.

## 7. Gymnasium checker results

`gymnasium.utils.env_checker.check_env(..., skip_render_check=True)` passed for
each integrated train, validation, and test environment. It checks the
Gymnasium reset/step contract, action and observation spaces, return shapes,
and info object non-reuse. Natural dataset exhaustion is reported through
`terminated`; `truncated` remains false because the environment has no
external time limit.

The first integrated checker run found shared nested metadata in environment
`info` responses. That issue was fixed by deep-copying metadata. No accounting
formula changes were needed.

## 8. Actual test results

Executed from the repository root using the project `.venv`:

```text
python -m unittest discover -s tests\rl -v
Ran 54 tests
OK
```

This includes the Phase 5.12 environment tests, Phase 5.13 generator tests,
and the new end-to-end integration tests. Pylance syntax checks found no syntax
errors in the new integration module and tests, or the metadata isolation
change. No PPO or historical evaluation command was run.

## 9. Defects and limitations

### Defect found and fixed

Gymnasium's checker found `reset()` and `step()` returned info mappings sharing
nested metadata objects. Mutating a prior reset result could affect later
responses, contrary to Gymnasium's info object-isolation expectation. The
environment now deep-copies metadata both on construction and at each response
boundary. Its accounting formulas are unchanged.

### Limitations

- The pipeline executes caller-supplied deterministic test actions only; it
  does not contain PPO or training/evaluation logic.
- Scenario separation uses independent seeded synthetic paths, not
  chronological historical partitions.
- Environment observations retain the documented synthetic assumptions and
  63-session warm-up.
- Integration proves software contract consistency for the tested fixtures;
  it does not validate financial realism, model quality, or investment
  performance.
- The selected Pylance interpreter has previously reported missing NumPy and
  Gymnasium imports even though the repository `.venv` runs the tests. Align
  the editor interpreter before using those import diagnostics as project
  environment evidence.

## 10. GO/NO-GO decision for PPO integration

**GO for a separate, synthetic-only PPO integration implementation phase.**
The generator-to-environment pipeline, seed/split contracts, deterministic
accounting fixture, Gymnasium checker, and the requested RL test suite pass.
This decision authorizes only implementing and validating PPO plumbing against
synthetic scenarios under the existing safety labels. It does **not** authorize
PPO training, strategy-performance claims, historical data use, historical
backtesting, or trading. Those require separate approval and the unresolved
historical provenance, licensing, adjustment-methodology, and corporate-action
blockers remain in force.
