"""Explicitly train a synthetic PPO model for the dashboard's validation view."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.rl.environment.synthetic_portfolio_env import (
    DEFAULT_OBSERVATION_SCHEMA,
    DEFAULT_PERIOD_SESSIONS,
    OBSERVATION_SCHEMA_WITH_TIME,
    OBSERVATION_SCHEMA_WITHOUT_TIME,
    OBSERVATION_WARMUP_SESSIONS,
)
from src.rl.phase4_evaluation import (
    evaluate_phase4,
    save_phase4_diagnostics,
    save_phase4_observation_action_diagnostics,
    save_phase4_results,
)
from src.rl.phase4_training import TrainingOrchestrator
from src.rl.ppo_agent import build_ppo_environments
from src.rl.ppo_config import PPOConfig, TrainingOptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--randomized-training-horizons",
        action="store_true",
        help=(
            "Randomize training-only episode lengths from 11 decisions "
            "to the full training horizon."
        ),
    )
    parser.add_argument("--min-episode-decisions", type=int, default=11)
    parser.add_argument("--max-episode-decisions", type=int)
    parser.add_argument("--horizon-seed", type=int)
    parser.add_argument(
        "--observation-schema",
        choices=(OBSERVATION_SCHEMA_WITH_TIME, OBSERVATION_SCHEMA_WITHOUT_TIME),
        default=DEFAULT_OBSERVATION_SCHEMA,
        help="Select the versioned 43-feature or 42-feature synthetic observation schema.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Use a new, empty artifact directory for this experiment.",
    )
    args = parser.parse_args()

    repository_artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
    randomized_default = repository_artifacts / "ppo_randomized_horizon" / "seed42"
    no_time_default = repository_artifacts / "ppo_without_time" / "seed42"
    default_output = repository_artifacts / "ppo"
    if args.randomized_training_horizons:
        default_output = randomized_default
    elif args.observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME:
        default_output = no_time_default
    output_dir = Path(
        args.output_dir
        or os.environ.get("PPO_ARTIFACT_DIR", str(default_output))
    ).expanduser()
    legacy_output = (repository_artifacts / "ppo").resolve()
    randomized_output = (repository_artifacts / "ppo_randomized_horizon").resolve()
    resolved_output = output_dir.resolve()
    if args.randomized_training_horizons and (
        args.observation_schema != OBSERVATION_SCHEMA_WITH_TIME
    ):
        parser.error(
            "randomized training horizons and the no-time observation schema "
            "must use separate configurations"
        )
    if args.randomized_training_horizons:
        if resolved_output == legacy_output or legacy_output in resolved_output.parents:
            raise ValueError(
                "Randomized-horizon training must use a separate directory outside .artifacts/ppo"
            )
        if resolved_output.exists() and any(resolved_output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing experiment artifacts: {resolved_output}"
            )
    if args.observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME:
        protected_directories = (legacy_output, randomized_output)
        if any(
            resolved_output == protected or protected in resolved_output.parents
            for protected in protected_directories
        ):
            raise ValueError(
                "The no-time experiment must use a separate directory outside "
                ".artifacts/ppo and .artifacts/ppo_randomized_horizon"
            )
        if resolved_output.exists() and any(resolved_output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing experiment artifacts: {resolved_output}"
            )
    if not args.randomized_training_horizons and (
        args.min_episode_decisions != 11
        or args.max_episode_decisions is not None
        or args.horizon_seed is not None
    ):
        parser.error("horizon options require --randomized-training-horizons")

    seed = 42
    config = PPOConfig(seed=seed, device="cpu")
    environments = build_ppo_environments(
        seed=seed,
        randomize_training_horizons=args.randomized_training_horizons,
        min_episode_decisions=args.min_episode_decisions,
        max_episode_decisions=args.max_episode_decisions,
        horizon_seed=args.horizon_seed,
        observation_schema=args.observation_schema,
    )
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
        experiment_name=(
            "phase7-randomized-horizon-ppo-seed42"
            if args.randomized_training_horizons
            else (
                "phase7-no-time-observation-ppo-seed42"
                if args.observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME
                else "dashboard-synthetic-ppo-seed42"
            )
        ),
    )
    if args.randomized_training_horizons:
        horizon_seed = seed if args.horizon_seed is None else args.horizon_seed
        maximum_decisions = args.max_episode_decisions
        if maximum_decisions is None:
            training_config = environments.pipeline.datasets.train.config
            maximum_decisions = (
                training_config.sessions - OBSERVATION_WARMUP_SESSIONS
            ) // DEFAULT_PERIOD_SESSIONS
        metadata["training_episode_horizon"] = {
            "sampling": "uniform_integer_decisions_inclusive",
            "minimum_decisions": args.min_episode_decisions,
            "maximum_decisions": maximum_decisions,
            "horizon_seed": horizon_seed,
        }
    observation_diagnostics: dict[str, object] = {}
    evaluation = evaluate_phase4(
        result.model,
        environments.validation_env,
        observation_diagnostics_callback=observation_diagnostics.update,
    )
    save_phase4_results(
        output_dir,
        experiment=metadata,
        evaluation=evaluation,
    )
    diagnostics_path = save_phase4_diagnostics(output_dir, evaluation)
    observation_diagnostics_path = save_phase4_observation_action_diagnostics(
        output_dir, observation_diagnostics
    )
    print(f"Synthetic PPO model: {model_path}")
    print(f"Model manifest: {manifest_path}")
    print(f"Validation evaluation: {output_dir / 'evaluation.json'}")
    print(f"PPO diagnostics: {diagnostics_path}")
    print(f"PPO observation/action diagnostics: {observation_diagnostics_path}")
    print("Results are synthetic software checks, not real-market performance.")


if __name__ == "__main__":
    main()
