"""Explicitly train a synthetic PPO model for the dashboard's validation view."""

from __future__ import annotations

import os
from pathlib import Path

from src.rl.phase4_evaluation import (
    evaluate_phase4,
    save_phase4_diagnostics,
    save_phase4_results,
)
from src.rl.phase4_training import TrainingOrchestrator
from src.rl.ppo_agent import build_ppo_environments
from src.rl.ppo_config import PPOConfig, TrainingOptions


def main() -> None:
    output_dir = Path(
        os.environ.get(
            "PPO_ARTIFACT_DIR",
            str(Path(__file__).resolve().parents[1] / ".artifacts" / "ppo"),
        )
    ).expanduser()
    seed = 42
    config = PPOConfig(seed=seed, device="cpu")
    environments = build_ppo_environments(seed=seed)
    orchestrator = TrainingOrchestrator(
        config=config,
        seed=seed,
        output_dir=output_dir,
    )
    result, metadata, model_path, manifest_path = orchestrator.train(
        environments.train_env,
        options=TrainingOptions(
            total_timesteps=10_000,
            seed=seed,
            verbose=0,
        ),
        experiment_name="dashboard-synthetic-ppo-seed42",
    )
    evaluation = evaluate_phase4(result.model, environments.validation_env)
    save_phase4_results(
        output_dir,
        experiment=metadata,
        evaluation=evaluation,
    )
    diagnostics_path = save_phase4_diagnostics(output_dir, evaluation)
    print(f"Synthetic PPO model: {model_path}")
    print(f"Model manifest: {manifest_path}")
    print(f"Validation evaluation: {output_dir / 'evaluation.json'}")
    print(f"PPO diagnostics: {diagnostics_path}")
    print("Results are synthetic software checks, not real-market performance.")


if __name__ == "__main__":
    main()
