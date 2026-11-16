# Phase 5.15 - PPO Agent Integration (Plumbing Only)

**Status:** PPO construction, deterministic inference, evaluation, and
synthetic-only persistence plumbing are implemented. **No PPO training was
executed.** This does not claim strategy quality or real-market performance.

## 1. PPO architecture

The optional Stable-Baselines3 implementation is in
`src/rl/ppo_agent.py`; configuration is separated into `src/rl/ppo_config.py`.
SB3, Torch, Monitor, and checkpoint callback imports are lazy. Importing the
agent module does not construct PPO or start training.

`create_ppo_agent()` validates that its input is a Gymnasium environment with
synthetic provenance, a 43-value observation space, and a continuous
seven-value `Box` action space. It constructs SB3 PPO with `MlpPolicy`, a
Gaussian continuous-action policy, and the configured multilayer perceptron.
The existing environment action conversion, accounting, constraints, and
reward are unchanged.

## 2. Configuration

`PPOConfig` defaults follow Phase 5.11:

| Parameter | Default |
|---|---:|
| `n_steps` | 1024 |
| `batch_size` | 256 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `learning_rate` | 0.0003 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| Policy network | two 128-unit ReLU hidden layers |
| `seed` | 42 |
| `device` | `auto` |

Validation rejects invalid numeric ranges, non-positive layer sizes, and
`batch_size` values that do not divide `n_steps`. `TrainingOptions` separately
validates the timestep budget, optional tensorboard log, and paired checkpoint
directory/frequency settings.

## 3. Environment compatibility and split isolation

`build_ppo_environments(config, seed)` uses the Phase 5.14 synthetic pipeline
to create three independently seeded scenarios. Only the training environment
is wrapped in SB3 `Monitor`. Validation and test environments remain separate
unwrapped Gymnasium instances; the factory does not fit a scaler or reuse the
training data as evaluation data.

The PPO constructor accepts the seven-dimensional continuous action and
43-dimensional observation unchanged. The evaluation helper resets its
provided synthetic environment and calls deterministic prediction only. It
collects actions, observations, rewards, NAV, weights, and
terminated/truncated flags.

## 4. Training interface

`train_ppo(env, total_timesteps, config, seed, tensorboard_log, checkpoint_dir,
checkpoint_freq, ...)` validates its arguments, constructs the model, optionally
configures a checkpoint callback, calls `.learn()` when explicitly invoked,
and returns a `TrainingResult` with the model and synthetic training metadata.

This function was **not invoked** in Phase 5.15. No `.learn()` call was made.
The test for this interface inspects and validates its signature/options only;
it does not execute a training run. Training requires separate approval after
code review.

## 5. Evaluation interface

`evaluate_ppo(model, env, deterministic=True, max_steps=None)` rejects
nondeterministic evaluation, non-synthetic environments, malformed prediction
shapes, non-finite actions, and actions outside the declared action space.
It returns `EvaluationResult` containing synthetic source status, scenario ID,
seed, the initial observation, and per-step `EvaluationStep` records with
actions, observations, rewards, NAV, allocation weights, and episode flags.

Evaluation is a software interface check on synthetic scenarios only. It is
not a strategy-quality measurement, benchmark study, or investment performance
claim.

## 6. Model persistence

`save_ppo_model()` saves the SB3 zip artifact and a sibling JSON manifest.
Persistence requires explicit `source_type="synthetic"` and `synthetic=true`
provenance. The top-level seed records the PPO/training seed, while the
provenance seed records the independent synthetic scenario seed. These are
validated separately because the train/validation/test generator derives
distinct scenario seeds from the root seed. The manifest contains:

- Synthetic-only source marker and allow-listed scenario provenance.
- Generator version, scenario seed/ID, configuration hash, and generation
  configuration when supplied by the generator.
- Asset ordering, observation/action dimensions, PPO configuration, Python,
  SB3, Torch, Gymnasium, and NumPy versions.
- SHA-256 checksum of the model file and a saved-at UTC timestamp.

`load_ppo_model()` requires both files, verifies manifest version, synthetic
marker, asset/dimension compatibility, optional expected PPO configuration,
and model checksum before calling SB3 load. It can attach a separately selected
environment after synthetic compatibility validation. The manifest uses an
allow-list; `.env` values and environment variables are never read or stored.
Model artifacts should be kept outside tracked source, e.g.
`models/synthetic/`.

## 7. Tests and actual results

Added `tests/rl/test_ppo_agent.py` with nine lightweight tests covering:

- PPO and runtime configuration validation.
- Training interface signature without executing training.
- Monitor wrapping of the training environment only and distinct validation
  and test scenarios.
- Agent initialization and 43/7 dimensional compatibility.
- Finite, in-bounds, repeatable deterministic predictions.
- Structured deterministic evaluation results.
- Model save/load prediction round trip and synthetic manifest metadata.
- Invalid/missing manifest and synthetic provenance validation.
- Rejection of non-synthetic environments.

Because SB3 and Torch were not initially installed in the repository `.venv`,
the already-declared optional dependency `stable-baselines3>=2.1.0` was
installed into that virtual environment for plumbing validation. No dependency
manifest was changed. Tested versions:

```text
stable-baselines3 2.9.2
torch 2.15.0+cpu
gymnasium 1.4.0
numpy 2.4.6
```

Full required RL test command, executed from the repository root:

```text
python -m unittest discover -s tests\rl -v
Ran 63 tests in 45.427s
OK
```

Pylance syntax checks found no syntax errors in the new PPO implementation,
configuration, and test files. The tests instantiate and save/load an
untrained randomly initialized PPO model to validate plumbing; they do not
call `.learn()`. Pylance's currently selected interpreter still reports
`reportMissingImports` for Gymnasium, NumPy, SB3, and Torch; the repository
`.venv` imported all four and ran the test suite successfully.

## 8. Known limitations

- The training function is implemented but intentionally unexecuted. No trained
  checkpoint, learning curve, policy quality, or training duration is available.
- The default 1024-step rollout configuration is architectural. Tests use tiny
  construction-only settings; no production training-budget tuning was done.
- Model persistence records generator configuration/provenance when provided,
  but callers remain responsible for supplying the correct synthetic scenario
  metadata. Metadata is not independent proof of data origin.
- `load_ppo_model()` supports only these synthetic environment dimensions and
  must not be used to connect a synthetic model to live trading.
- SB3/Torch are optional dependencies. They were installed only in the local
  `.venv` for this validation; `setup.py`/`requirements.txt` were not changed.
- Historical market data remains NO-GO pending provenance, licensing,
  adjustment-methodology, and corporate-action resolution.

## 9. Training execution confirmation

No PPO training was executed. `train_ppo()` was not called; `.learn()` was not
invoked. No historical CSVs, `.env`, market-data APIs, financial downloads,
historical backtests, or trade execution were used. Future training requires a
separate approval after code review.
