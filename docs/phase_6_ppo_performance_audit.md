# Phase 6 — PPO Performance Audit and Allocation Diagnostics

## Scope and evidence

The previously requested Phase 6 audit file was not present when this task
started. This document consolidates the relevant audit findings and records
Phase 6.1. No PPO training or historical-data access occurred. The existing
seed-42 model was run once in deterministic synthetic validation solely to
produce diagnostics; the existing `evaluation.json` baseline was byte-for-byte
preserved.

The saved evaluation uses 693 sessions after a 63-session warm-up, 33 rebalance
decisions, $100,000 initial capital, a 0.1% transaction fee, and 5 bps
slippage. Existing results remain unchanged:

| Strategy | Cumulative return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| Synthetic PPO | 15.14% | 0.491 | -14.11% |
| Equal-weight | 18.19% | 0.533 | -12.88% |
| Monthly rebalancing | 18.67% | 0.545 | -13.03% |
| Threshold rebalancing | 20.86% | 0.596 | -13.25% |
| SPY buy-and-hold | 138.03% | 1.425 | -19.45% |

These are results from one synthetic validation path, not evidence of
real-market performance or robust benchmark outperformance.

## Architecture and audit findings

- [`SyntheticPortfolioEnv`](../src/rl/environment/synthetic_portfolio_env.py)
  exposes a 43-value observation and a seven-value continuous action. It
  projects actions to long-only weights, applies a 35% per-asset cap, 2%
  minimum target cash, and a 1.0 gross-turnover cap. Cash earns 0%.
- The reward is net log-NAV growth. Trading fees and slippage are deducted
  from NAV once; there is no separate volatility or drawdown reward penalty.
- The saved PPO configuration is seed 42, CPU, `n_steps=1024`,
  `batch_size=256`, `gamma=0.99`, `gae_lambda=0.95`, learning rate `3e-4`,
  `clip_range=0.2`, and `ent_coef=0`. No reward or hyperparameter changes were
  made for Phase 6.1.
- Synthetic splits use independently derived train, validation, and test
  seeds. The training script uses the train environment and evaluates the
  validation environment; it does not use the test environment. However,
  `TrainingOrchestrator._guard_environment()` verifies synthetic provenance,
  not that the supplied environment is specifically the train split. This is
  a guard limitation, not evidence that the current run leaked validation
  data.
- Training repeatedly resets the same generated training path. This creates a
  credible overfitting risk. One seed and one validation path are insufficient
  to establish robustness.
- Benchmarks share the evaluation window, starting capital, and cost rate, but
  do not all share PPO’s asset caps, cash floor, turnover limit, or rebalance
  frequency. In particular, SPY buy-and-hold is fully concentrated in one
  asset. Treat it as context, not a risk-matched comparator.
- The saved terminal marked cash allocation was about 60.2%. That is not the
  2% target floor and is not the average allocation over the run. The original
  saved summary did not contain per-decision actions, so the cause of the
  terminal cash level could not be established. Candidate causes such as
  repeated-path overfitting or a conservative learned policy remain
  hypotheses.

## Phase 6.1 implementation

`evaluate_ppo()` now retains each environment transition's requested target
weights and actually applied weights in its existing `EvaluationStep`, while
preserving the existing raw `action`, marked ending `weights`, NAV, turnover,
cost, and daily-NAV fields.

`evaluate_ppo_synthetic()` adds an `allocation_diagnostics` object without
removing or renaming existing metrics. It contains:

- One `rebalance_history` entry per decision: decision number, raw action,
  requested target weights, applied portfolio weights, applied cash weight,
  interval-ending portfolio NAV, gross and one-way turnover, fees, slippage,
  and total transaction costs.
- A `summary` over applied rebalance weights: mean/minimum/maximum cash,
  average risky allocation, per-asset mean/minimum/maximum, average gross and
  one-way turnover, and totals for fees, slippage, and transaction costs.

Requested and applied allocations are checked for finite, long-only weights
that sum to one. Turnover and cost relationships are checked; total costs are
not added to fees and slippage a second time. `portfolio_nav` in each record is
the NAV after the subsequent return interval, while `applied_portfolio_weights`
are the weights used before that interval.

`save_phase4_diagnostics()` writes a separate `ppo_diagnostics.json`.
`evaluation.json` and its benchmark evidence are not overwritten by this
function. The existing training/validation script calls the new saver for
future runs, but it was not run in this task.

The generated diagnostic artifact for the existing validation model records
33 decisions, **38.84% average applied cash**, 17.61% minimum and 64.63%
maximum applied cash, 61.16% average risky allocation, 0.0879 average
one-way turnover, and $910.81 total costs ($607.20 fees plus $303.60
slippage). This average applied cash differs from terminal marked cash because
the measurements describe different things.

## Phase 6.4 observation-to-action diagnostics

`EvaluationStep.decision_observation` now preserves the exact 43-value
observation passed to deterministic `model.predict()` before the environment
step. The existing `EvaluationStep.observation` field retains its previous
post-step meaning. The optional observation-diagnostics callback on
`evaluate_ppo_synthetic()` and `evaluate_phase4()` exposes a separate trace
without changing their default metrics or API response.

Each decision record includes the raw observation vector and named features:

- Per-asset one-, five-, and 21-session returns, plus 21- and 63-session
  volatility.
- Benchmark one-session return and 21-session volatility.
- Actual VIX level, its scaled model input, and the scale used.
- Current asset weights and cash weight, portfolio-value ratio, drawdown, and
  fraction of the episode remaining.
- The raw PPO action, requested target weights, applied weights, reward, and
  interval-ending NAV.

The training/validation script saves this trace separately as
`.artifacts/ppo/ppo_observation_action_diagnostics.json`. The artifact was
generated for all 33 decisions using the existing saved synthetic model and
its validation scenario. This was deterministic inference only; PPO was not
retrained. The regenerated metrics, actions, and allocations matched the
existing evidence, and both `evaluation.json` and `ppo_diagnostics.json` were
verified unchanged.

## Tests and verification

- Focused PPO and evaluation tests: **19 passed**.
- Full offline pytest suite, excluding `tests/test_alpha_vantage.py`:
  **201 passed, 14 warnings, 35 subtests passed**.
- Python compilation and `git diff --check`: passed.
- The saved model reproduced the existing return, Sharpe, drawdown, and cost
  metrics, raw actions, and allocations; both previous evaluation artifacts
  were confirmed unchanged after writing the new observation/action trace.
- No model training, PPO configuration change, reward change, dashboard
  modification, broker call, or historical-data access was performed.

## Relevant implementation and test locations

- [`ppo_agent.py`](../src/rl/ppo_agent.py): `EvaluationStep`, `evaluate_ppo()`.
- [`phase4_evaluation.py`](../src/rl/phase4_evaluation.py):
  `evaluate_ppo_synthetic()`, `summarize_allocation_diagnostics()`,
  `build_observation_action_diagnostics()`, and the diagnostics savers.
- [`train_synthetic_ppo_dashboard.py`](../scripts/train_synthetic_ppo_dashboard.py):
  future explicit artifact saving.
- [`test_ppo_agent.py`](../tests/rl/test_ppo_agent.py): per-step allocation
  capture and validity.
- [`test_phase4_workflow.py`](../tests/rl/test_phase4_workflow.py): diagnostic
  summaries, observation/action alignment, JSON serialization, decision
  counts, invalid allocation rejection, cost accounting, and preservation of
  existing evaluation files.
- Existing synthetic evidence:
  [`evaluation.json`](../.artifacts/ppo/evaluation.json),
  [`experiment.json`](../.artifacts/ppo/experiment.json), and
  [`model.manifest.json`](../.artifacts/ppo/model.manifest.json).

## Remaining limitations

Diagnostics cover rebalance-point observations and allocations, not daily
marked allocation history or training learning curves. Evaluation remains one
seed and one validation path. A controlled multi-seed synthetic robustness
study is still needed before making policy-quality claims.

## Phase 7.0 — Configuration B randomized training horizons

The optional `RandomizedEpisodeHorizonEnv` wrapper is applied only to the
synthetic training split by `build_ppo_environments()`. It samples an inclusive
uniform number of complete rebalance decisions (default 11 through the maximum
available, 33 for the 756-session configuration) using an explicit horizon
seed. Each sampled endpoint includes the 63-session warm-up plus complete
21-session periods. The base environment remains fixed-length by default;
validation and test environments are unchanged.

Use the explicit training command
`python -m scripts.train_synthetic_ppo_dashboard --randomized-training-horizons`
to opt in. This mode defaults to the separate
`.artifacts/ppo_randomized_horizon/seed42` directory and refuses to write
inside `.artifacts/ppo` or over a non-empty experiment directory. No model was
trained for this implementation task.

## Phase 7.0 — Configuration C without time remaining

The default observation schema remains
`synthetic_portfolio_v1_with_time` (43 features). The opt-in
`synthetic_portfolio_v1_without_time` schema removes only
`time_remaining_fraction`, yielding 42 features with all preceding values in
the same order. New model manifests record the schema name, schema version,
and observation shape; schema-less legacy manifests are interpreted as the
existing 43-feature schema only.

Select the no-time schema with
`python -m scripts.train_synthetic_ppo_dashboard --observation-schema synthetic_portfolio_v1_without_time`.
This mode defaults to `.artifacts/ppo_without_time/seed42`, requires a new or
empty output directory, and refuses paths inside `.artifacts/ppo` or
`.artifacts/ppo_randomized_horizon`. Configuration B randomized horizons and
Configuration C are deliberately rejected together. No model was trained for
this implementation task.

## Phase 7.2 — Experiment protocol and safe run planning

The versioned protocol is
[`phase7_2_experiment_protocol.json`](../configs/phase7_2_experiment_protocol.json).
It fixes configurations A/B/C, training seeds 101/202/303/404/505, validation
roots 7201–7220, reserved test roots 9201–9220, common PPO and generator
settings, costs, initial capital, and the 693-session/33-decision evaluation
window. B uses complete horizons of 11–33 decisions and records one sampler
seed per training seed.

`src/rl/phase7_experiment.py` validates the protocol and seed-role separation,
plans a unique `.artifacts/phase7_2/config_<A|B|C>/seed_<n>` directory for
each run, and can reserve the full set of output directories only after
checking all paths. It refuses to use `.artifacts/ppo`. Run metadata is
validated against the protocol before a Phase 7.2 `TrainingOrchestrator` call
and is copied into the model manifest and experiment metadata. This work adds
preparation only; no model training, validation evaluation, or test evaluation
has been performed.
