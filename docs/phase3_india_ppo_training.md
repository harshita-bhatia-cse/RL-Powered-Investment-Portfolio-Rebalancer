# Phase 3: India-Specific PPO Training Preparation

Phase 3 defines a separate, guarded India historical PPO workflow. It does not
download data, generate data, train a model or modify synthetic PPO artifacts.

## Data readiness

Training is blocked unless all of the following are satisfied:

- a Phase 1 validated CSV exists;
- the Phase 1 manifest has `training_ready=true`;
- the manifest has `license_permission_status="approved"`;
- provider, adjustment method and retrieval timestamp are present;
- configured symbols match the manifest;
- train, validation and test splits are chronological;
- each environment observation is finite and inside the Gymnasium space.

The repository currently contains no real India CSV or India ingestion manifest.
Therefore data-readiness is **not confirmed** and actual India PPO training is
blocked.

## Pipeline

`src/rl/india_ppo_training.py` provides:

1. `prepare_india_ppo_training()`:
   - loads the India Phase 1/2 dataset;
   - validates Phase 1 training-ready and permission status;
   - fits the scaler on training rows only;
   - constructs separate train, validation and test environments;
   - validates observations and output paths;
   - returns metadata without creating artifacts or calling PPO `.learn()`.
2. `create_india_ppo_agent()`:
   - lazily imports optional Stable-Baselines3;
   - constructs PPO using the existing `PPOConfig`;
   - does not call `.learn()`.
3. `train_india_ppo()`:
   - requires explicit `IndiaHistoricalAuthorization`;
   - verifies the India-only output root;
   - creates the agent and calls `.learn()` only after approval;
   - writes model and manifest only under `.artifacts/india_daily_ppo/`.

The India artifact root is separate from `.artifacts/ppo`. Existing synthetic
models, manifests and diagnostics are never read or modified.

## Safeguards

- No historical training can bypass `training_ready=true`.
- Legacy `.artifacts/ppo` and any output outside `.artifacts/india_daily_ppo`
  are rejected.
- Non-empty output directories are not overwritten.
- Validation and test environments receive the unchanged training-fitted scaler.
- Test data is reserved for later final evaluation, not training or model selection.
- Corporate-action coverage remains a limitation until independently verified.

## Recommended future sequence

1. Supply approved real India CSV files and a complete Phase 1 manifest.
2. Run and approve Phase 3 preflight tests.
3. Create an explicit authorization record.
4. Train only in an empty India artifact directory.
5. Evaluate on validation data for model selection.
6. Use the reserved test partition once for final reporting.
