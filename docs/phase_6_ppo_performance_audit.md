# Phase 6 — PPO Performance Audit and Allocation Diagnostics

## Scope and evidence

This document records the Phase 6.1 PPO performance audit and allocation
diagnostics, along with the later Phase 7.2 synthetic experiment protocol and
Phase 8A read-only research API. The Phase 6.1 audit did not train PPO or
access historical data. The existing seed-42 model was run once in deterministic
synthetic validation solely to produce diagnostics; the existing
`evaluation.json` baseline was byte-for-byte preserved.

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

These Phase 6.1 diagnostics cover rebalance-point observations and
allocations, not daily marked allocation history or training learning curves.
That Phase 6 evaluation used one seed and one validation path; its limits are
not changed by the later, separate Phase 7.2 multi-seed synthetic experiment.
Neither evaluation establishes real-market performance or policy quality
beyond its tested scenarios.

## Earlier synthetic PPO configuration option: randomized training horizons

The configuration notes in this section are not separate Phase 7 sub-phases.
Phase 7.2 is the only implemented Phase 7 sub-phase; Phase 7.1 is not part of
this project.

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
inside `.artifacts/ppo` or over a non-empty experiment directory. This is a
synthetic PPO configuration option and does not imply real-market evaluation.

## Earlier synthetic PPO configuration option: observation schema

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
Configuration C are deliberately rejected together. This describes a
synthetic PPO configuration option, not a separate Phase 7 sub-phase or
real-market evaluation.

## Phase 7.2 — Experiment protocol and safe run planning

The versioned protocol is
[`phase7_2_experiment_protocol.json`](../configs/phase7_2_experiment_protocol.json).
It fixes configurations A/B/C, training seeds 101/202/303/404/505, validation
roots 7201–7220, reserved test roots 9201–9220, common PPO and generator
settings, costs, initial capital, and the 693-session/33-decision evaluation
window. B uses complete horizons of 11–33 decisions and records one sampler
seed per training seed.

`src/rl/phase7_experiment.py` validates the protocol against a reviewed
canonical SHA-256, plans a unique
`.artifacts/phase7_2/config_<A|B|C>/seed_<n>` directory for each run, and
reserves the output tree transactionally. A failed reservation removes only
empty directories created by that attempt. It refuses to use `.artifacts/ppo`.
Training provenance is checked against the protocol's generator configuration
and the generator's recomputed canonical configuration hash before it is
copied into the model manifest and experiment metadata. Updating the approved
protocol requires an intentional checksum update in the validator. These
protocol settings and runner behavior describe the experiment design; they do
not, by themselves, establish that an experiment completed. The recorded
Phase 7.2 experiment outcome is that all 15 configuration/seed runs were
trained, 300 validation results were recorded, Configuration A was selected,
and 100 reserved-test evaluations were recorded. These outputs are synthetic
experiment results only, not NSE/BSE or real-market performance.

### Phase 7.2 staged runner

`src/rl/phase7_runner.py` and `scripts/run_phase7_experiment.py` provide
separate `plan`, `train`, `validate`, and `test` stages. The runner consumes the
checked-in protocol without changing its seeds, model settings, environment
assumptions, or evaluation horizons. It generates only the training scenario
during training; the twenty validation roots are generated only during
validation, and the twenty reserved-test roots are generated only after a
selection manifest has been frozen.

Validation evaluates all fifteen configuration/seed candidates on all twenty
validation roots. It computes a mean Sharpe ratio across roots per seed and then
averages all five seed-level values for each configuration. Selection chooses
the highest configuration mean, with a named tolerance of `1e-12`; ties are
resolved by the lowest mean maximum-drawdown magnitude, then lowest mean
turnover, then lexicographically smallest configuration ID. The stored
`maximum_drawdown` metric is non-positive, so the tie-break uses its absolute
magnitude to select the smaller drawdown rather than the numerically smaller,
worse negative value. All five seeds are retained for configuration comparison
and, for the selected configuration only, final evaluation.

The final test stage requires the validated selection and checks its manifest,
the completed validation result matrix, and all selected model checksums before
evaluating exactly five models against the twenty reserved roots. Those one
hundred held-out results are reports only; they are not fed back into selection.
Stage locks use one exclusive-create operation for ownership and record host,
stable machine identity, PID, process creation identity, timestamp, and
invocation token. Automatic stale-lock recovery is disabled: an existing,
malformed, remote, or otherwise unverifiable lock always fails closed and
requires manual inspection. On Windows, the creator retains a native file
handle opened without delete sharing for the full stage. Release verifies
ownership through that handle, confirms the path still identifies the same
file, then marks that handle's file for deletion by handle. The held share mode
prevents another process from renaming or replacing the lock path between
verification and deletion; release never unlinks the pathname. If safe
handle-based release is unavailable, the runner refuses lock acquisition
instead of falling back to a check-then-unlink sequence. The runner verifies
the volume filesystem with `GetVolumeInformationW` and allows only NTFS;
unknown, empty, query-failure, ReFS, and other filesystem results fail closed.
Consequently, race-safe stage-lock execution is currently supported only on
local fixed Windows NTFS volumes that honor these handle-sharing semantics.
Network, removable, unknown, OneDrive-synchronized, and unsupported
filesystems remain rejected.
Other synchronization products may not be detectable and must not be used for
concurrent experiment execution. This does not provide a distributed lock
guarantee for shared or synchronized folders.
Result JSON and its checksum sidecar are created without
overwriting; the stage manifest is atomically updated and protected by a
canonical SHA-256 integrity check. These hashes detect inconsistency when the
trusted original evidence remains protected; they do not prove authenticity.
Completed training/evaluation results can be reused only after their protocol,
model, metadata, parameter, result, and sidecar checks pass. Partial or failed
training is not automatically retried; incomplete validation and test stages
may resume from individually verified result files. A completed final-test
replay rechecks the exact 100 result references and their sidecars against the
frozen selection before returning the existing report; it never regenerates
completed results.

The runner has separate commands for each stage:

```powershell
python -m scripts.run_phase7_experiment plan
python -m scripts.run_phase7_experiment train
python -m scripts.run_phase7_experiment validate
python -m scripts.run_phase7_experiment test
```

`plan` reserves `.artifacts/phase7_2` and creates the run manifest; it refuses
an already-existing output root. The `train`, `validate`, and `test` commands
perform PPO training, validation, and final held-out evaluation, respectively.
The approved run is complete and its outputs are frozen. Do not rerun these
commands against the frozen artifact root. Synthetic metrics from this
experiment are not real-market investment performance.

### Explicit recovery of the failed Phase 7.2 A/101 run

The completed experiment used one explicitly authorized retry for
configuration A, training seed 101, as attempt `attempt_0002`. Attempt 1
remains recorded as failed; attempt 2 is the active run. The retry command is
not reproduced here as an instruction to rerun against frozen outputs.

The retry uses
`.artifacts/phase7_2/retries/attempt_0002/config_A/seed_101/`. It never
reuses or cleans the canonical `config_A/seed_101/` directory. Before
authorization, the runner checks the NTFS/OneDrive/network guard, holds the
normal stage lock, verifies the protocol and manifest, requires the original
run to be marked failed and empty, and refuses pre-existing retry paths or
recovery-event records. It creates the retry output directory exclusively
and writes the recovery event using create-new semantics. Retry and recovery
paths reject Windows reparse points, including directory junctions, at every
path component and verify that the resolved destination remains beneath the
approved experiment output root.

These checks are fail-closed for pre-existing redirection, but they do not
provide a handle-pinned defense against a hostile process that can replace
directories concurrently between validation and filesystem operations. Run
the retry only in a trusted workspace without concurrent external filesystem
mutation; the stage lock coordinates runner processes, not arbitrary local
processes.

The recovery event records the original manifest digest, the complete
historical failed A/101 entry, the fixed retry identity and path, its unique
reservation token, the unchanged protocol digest, timestamp, and operator
reason. The event and its digest are linked from a v2 manifest. Attempt 1
remains recorded as failed; attempt 2 progresses through `planned`,
`training`, then `trained` or `failed`. Attempt 2 becomes active only after
the normal model, experiment metadata, synthetic provenance, timestep, and
checksum checks pass. A failed retry is not retried automatically.

The v1 reader retains canonical-path behavior for existing planned,
interrupted, selected, and completed manifests. Only the explicit retry
operation converts the eligible A/101 interrupted manifest to
`phase72-run-manifest-v2`; the other fourteen canonical run identities and
their original reservations remain unchanged. V2 resolution accepts the
retry path only from its validated attempt record and linked recovery event.
Training continuation, validation, selection reconciliation, final-test
evaluation, and completed-test replay resolve artifacts through the active
attempt. The active attempt cannot be changed after authorization, and retry
is blocked after validation or later-stage artifacts exist.

If a process stops after writing an unlinked event or reserving only part of
the retry directory, that orphan state does not authorize training; a later
retry refuses to reuse it and requires manual inspection. If the manifest
records attempt 2 as `training`, the runner may reconcile it only when the
existing training result and all artifacts pass the normal verifier; it never
starts that attempt again. An in-progress attempt without a verifiable
completed result remains fail-closed. Do not delete or edit orphan evidence
or manifest files to force recovery.

The recovery procedure did not change the canonical experiment protocol,
seed roles, validation aggregation, selection rule, or reserved-test policy.
It does not represent synthetic results as NSE/BSE performance. SHA-256
digests detect inconsistency against retained evidence but are not signatures
and do not provide authenticity if an actor can rewrite both the manifest and
event. Preserve the recovery event, manifest history, and both attempt
directories together for audit.

The frozen experiment artifacts are read-only inputs to the Phase 8A research
API.

## Phase 8A — Read-only synthetic research API

Phase 8A exposes six authenticated GET endpoints for summaries from the
completed Phase 7.2 experiment:

- `GET /api/research/phase7-2/status`
- `GET /api/research/phase7-2/selection`
- `GET /api/research/phase7-2/validation`
- `GET /api/research/phase7-2/final-test`
- `GET /api/research/phase7-2/final-test/seeds`
- `GET /api/research/phase7-2/integrity`

Every request uses the shared account `get_current_user` dependency, which
uses the account database through `get_db`. If account database
initialization/readiness fails, research requests also return HTTP 503. If
artifact verification or reading fails, the router returns a separate generic
research-service HTTP 503 response.

Each request creates a fresh `Phase72ResearchStore`, verifies the frozen
artifact set, and only then reads the response data. The store expects the
required Phase 7.2 artifacts beneath the fixed project path
`.artifacts/phase7_2`; the API accepts no caller-supplied artifact path. The
artifact set must be provisioned at that path. Missing or invalid artifacts
are not replaced with generated or partial results.

The research API is read-only: it does not load PPO models, train models, run
inference, or execute backtests. It only returns stored synthetic experiment
summaries. These results must never be represented as NSE/BSE or real-market
performance, investment recommendations, or evidence of expected returns.
Account, portfolio, market-data, and historical CSV backtest functionality
remain separate from synthetic PPO research. The dashboard is teammate-owned;
it was not modified or connected as part of Phase 8A.
