# Phase 5.16 - First Controlled Synthetic PPO Training Run

**Outcome:** One synthetic-only PPO training run completed and was evaluated on
the isolated validation scenario. The first invocation stopped on an
environment constraint/turnover interaction; the defect was reproduced,
documented, fixed without changing accounting or reward formulas, covered by a
regression test, and the same experiment configuration was rerun successfully.
These results are software-pipeline evidence only, not real-market investment
performance.

## 1. Pre-training checks

All required gates passed before the successful run:

| Check | Result |
|---|---|
| PPO configuration validation | PASS; existing hyperparameters retained, CPU selected |
| Observation/action spaces | PASS; 43 observations and 7 continuous actions |
| Data provenance | PASS; each scenario reports `source_type="synthetic"` and `synthetic=true` |
| Asset order | PASS; SPY, QQQ, IEF, GLD, XLU, XLV |
| Scenario isolation | PASS; distinct scenario IDs/seeds and non-shared asset, benchmark, and VIX arrays |
| Generator configuration | PASS; same configuration hash across train, validation, and test |
| Training source | PASS; only the training environment was passed to `train_ppo()` |
| Model output location | PASS; artifacts are under `AI Finance\phase_5_16_artifacts_20261114`, outside the Git repository |
| CPU runtime | PASS; Torch reported `2.15.0+cpu`, CUDA unavailable |

Scenario seeds and IDs:

| Split | Scenario ID | Derived scenario seed |
|---|---|---:|
| Train | `train-16138347438539916964` | 16138347438539916964 |
| Validation | `validation-134183728835869882` | 134183728835869882 |
| Test | `test-11601846009883706861` | 11601846009883706861 |

The test split was inspected only for required isolation/provenance/space
preflight checks. It was not used for training, evaluation, or model selection.
The PPO random seed was `42`; the derived training scenario seed is separately
recorded in the model manifest.

Runtime versions:

| Component | Version |
|---|---|
| Python | 3.11.9 |
| Stable-Baselines3 | 2.9.2 |
| PyTorch | 2.15.0+cpu |
| Gymnasium | 1.4.0 |
| NumPy | 2.4.6 |

## 2. Exact training configuration

The successful `train_ppo()` invocation used the following experiment
settings. It repeated the same settings and seed as the initial failed
invocation; no hyperparameter tuning or additional experiment was performed.

| Setting | Value |
|---|---:|
| Requested total timesteps | 10,000 |
| PPO/training seed | 42 |
| Device | CPU |
| `n_steps` | 1024 |
| `batch_size` | 256 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| Learning rate | 0.0003 |
| Clip range | 0.2 |
| Entropy coefficient | 0.0 |
| Value coefficient | 0.5 |
| Maximum gradient norm | 0.5 |
| Policy network | two 128-unit ReLU hidden layers |
| Logging | SB3 console verbosity 1; `log_interval=1` |
| Checkpoint frequency | 5,000 callback steps |
| TensorBoard | Disabled; package not installed in the existing environment |

SB3 PPO collects complete 1,024-step rollouts. It therefore completed **10,240
environment timesteps** for the requested 10,000-step budget.

## 3. Training duration

Measured wall-clock duration for the successful `train_ppo()` call, including
model construction and learning: **159.10 seconds** (about 2 minutes 39
seconds). Validation, model serialization, and baseline runs are not included
in that duration.

## 4. Training metrics and observations

Console metrics were captured for each rollout in
[`training_console.log`](../../phase_5_16_artifacts_20261114/training_console.log).
Selected observed SB3 metrics:

| Metric | First rollout | Final rollout |
|---|---:|---:|
| `rollout/ep_len_mean` | 33 | 33 |
| `rollout/ep_rew_mean` | 0.227 | 0.243 |
| `time/total_timesteps` | 1,024 | 10,240 |
| `train/approx_kl` | — | 0.0041774595 |
| `train/clip_fraction` | — | 0.039 |
| `train/entropy_loss` | — | -9.91 |
| `train/explained_variance` | — | 0.427 |
| `train/policy_gradient_loss` | — | -0.00892 |
| `train/value_loss` | — | 0.0021 |

These are PPO diagnostics and episodic rewards on the generated training
scenario only. They are not real-market measurements and do not establish
policy quality or generalization.

## 5. Model artifact and checksum verification

The trained model and supporting artifacts were saved outside the repository:

- Model: [`trained_synthetic_ppo_seed42.zip`](../../phase_5_16_artifacts_20261114/trained_synthetic_ppo_seed42.zip)
- Manifest: [`trained_synthetic_ppo_seed42.manifest.json`](../../phase_5_16_artifacts_20261114/trained_synthetic_ppo_seed42.manifest.json)
- Experiment configuration: [`experiment_config.json`](../../phase_5_16_artifacts_20261114/experiment_config.json)
- Checkpoints: [`checkpoints/`](../../phase_5_16_artifacts_20261114/checkpoints/)
- Training result metadata: [`training_result.json`](../../phase_5_16_artifacts_20261114/training_result.json)

SHA-256:

```text
89ac44363f50cbfa57e9b2c2dd1414c3a41ff9fa360a8bb74781d945d7979930
```

The digest was independently checked with PowerShell `Get-FileHash` and matched
the manifest. `load_ppo_model()` also successfully verified the manifest and
checksum, loaded the saved model against the validation environment, and
confirmed 43-observation/7-action compatibility. The manifest records synthetic
provenance, the training seed (42), the distinct training scenario seed, asset
order, configuration, and library versions.

Checkpoints were produced at the configured 5,000 and 10,000 callback steps.
The successful run log contains no warning or traceback matches.

## 6. Validation results

The reloaded model ran one deterministic, full 33-decision episode on the
separate validation scenario (`validation-134183728835869882`), starting at
100,000 synthetic units. The per-step actions, rewards, NAV, allocations,
turnover, and costs are in
[`validation_results.json`](../../phase_5_16_artifacts_20261114/validation_results.json).

| Synthetic validation output | PPO |
|---|---:|
| Cumulative episode reward (sum of net log-NAV rewards) | -0.130203 |
| Ending NAV | 87,791.70 |
| Gross turnover sum | 7.406225 |
| One-way turnover sum | 3.703112 |
| Transaction fees | 714.91 |
| Slippage cost | 357.46 |
| Total transaction cost | 1,072.37 |
| Terminated / truncated | true / false |

Ending marked weights were SPY 24.58%, QQQ 2.96%, IEF 3.21%, GLD 3.41%,
XLU 35.92%, XLV 26.59%, and cash 3.32%. Market returns can cause marked
weights at episode end to drift from rebalance-target constraints.

Validation data was not used for training or model selection. No test-scenario
performance evaluation was run.

## 7. Baseline sanity checks

Both reference policies used a fresh environment on the same validation
scenario, the same initial NAV, costs, action conversion, and accounting rules.
The equal-weight target was reapplied each decision. The buy-and-hold reference
set the same initial target and then requested the current marked allocation;
environment caps, cash minimum, and turnover limit could still require
corrective trades.

| Synthetic validation output | Equal-weight, rebalanced | Buy-and-hold reference |
|---|---:|---:|
| Cumulative episode reward | 0.028020 | 0.020727 |
| Ending NAV | 102,841.61 | 102,094.36 |
| Gross turnover sum | 3.125463 | 1.000001 |
| One-way turnover sum | 1.562731 | 0.500000 |
| Transaction fees | 314.88 | 100.00 |
| Slippage cost | 157.44 | 50.00 |
| Total transaction cost | 472.32 | 150.00 |
| Terminated / truncated | true / false | true / false |

These are one-path synthetic software sanity checks. They do not demonstrate
real-world performance, statistical significance, or investment superiority.

## 8. Limitations, failure, and changes made

- The first training invocation failed before producing a final model. Its
  traceback is preserved in
  [`training_attempt_1_failure.log`](../../phase_5_16_artifacts_20261114/training_attempt_1_failure.log).
- The failure exposed a constraint/turnover defect: synthetic returns can
  drift marked cash below its minimum between rebalance decisions. A maximum
  turnover limit can then prevent the next applied transition from restoring
  the cash floor in one step. The previous environment incorrectly raised
  `portfolio constraint projection produced invalid weights`.
- A seeded valid-action replay reproduced the condition (pre-trade cash
  0.0197679, target cash 0.02, applied cash 0.0199693 under the unchanged
  1.0 gross-turnover cap). The environment now validates that targets satisfy
  bounds and that any turnover-limited transition from a market-drifted breach
  does not worsen it. `info["constraint_recovery"]` reports a still-pending
  recovery. The turnover limit, transaction-cost calculations, NAV accounting,
  and reward formula were not changed. The 500-action regression test passes.
- The failed attempt's elapsed time was not persisted by the first runner; no
  duration is claimed for it. The retry used the same settings, seed, and data
  configuration.
- Only one root seed and one validation scenario were evaluated. This small
  training budget and synthetic regime process cannot support robust
  generalization conclusions. Hyperparameters were not tuned.
- TensorBoard logging was unavailable; SB3 console diagnostics and checkpoints
  were captured instead.
- No historical CSVs, `.env`, market-data API, downloaded data, historical
  backtest, test-scenario evaluation, or trade execution was used.

Files changed for this phase:

- `src/rl/ppo_agent.py` — explicit training verbosity; validation result
  turnover/fee/slippage fields; separate recording of PPO and scenario seeds.
- `src/rl/ppo_config.py` — validated runtime verbosity setting.
- `src/rl/environment/synthetic_portfolio_env.py` — recoverable
  market-drift/turnover constraint handling and `constraint_recovery` status.
- `tests/rl/test_ppo_agent.py` and
  `tests/rl/test_synthetic_portfolio_env.py` — coverage for logging,
  seed/provenance, validation cost metrics, and the reproduced constraint case.
- `docs/phase_5_12_portfolio_environment_implementation.md` and
  `docs/phase_5_15_ppo_agent_integration.md` — directly related behavior and
  seed-provenance documentation.
- `docs/phase_5_16_first_synthetic_ppo_training.md` — this report.

Required RL test command, run after the code fix and before the successful
training run:

```text
python -m unittest discover -s tests\rl -v
Ran 65 tests in 46.227s
OK
```

## 9. GO/NO-GO recommendation

**NO-GO for immediate follow-on PPO tuning or strategy selection based on this
single synthetic validation episode.** The trained policy's observed synthetic
validation NAV was below both simple references on this one generated path;
that result is a diagnostic observation, not an investment-performance claim.

Proceed first with code review of the constraint-recovery change, the manifest,
and the captured training/validation artifacts. A separately approved,
pre-specified synthetic replication may then be considered. Historical-data
training, real-market backtesting, and live trading remain **NO-GO**.
