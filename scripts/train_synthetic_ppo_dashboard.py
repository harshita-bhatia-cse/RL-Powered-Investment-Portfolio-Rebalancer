"""Explicitly train a synthetic PPO model for the dashboard's validation view."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

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


def _non_negative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed must be a non-negative integer"
        ) from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return seed


def _validate_output_directory(
    output_dir: Path,
    *,
    legacy_output: Path,
    protected_directories: tuple[Path, ...] = (),
) -> Path:
    resolved_output = output_dir.resolve()
    resolved_legacy = legacy_output.resolve()
    if (
        resolved_output != resolved_legacy
        and resolved_legacy in resolved_output.parents
    ):
        raise ValueError(
            "Experiment outputs must not be written inside the protected "
            ".artifacts/ppo baseline directory"
        )
    if any(
        resolved_output == protected.resolve()
        or protected.resolve() in resolved_output.parents
        for protected in protected_directories
    ):
        raise ValueError("This experiment must use a separate artifact directory")
    if resolved_output.exists() and not resolved_output.is_dir():
        raise FileExistsError(
            f"Artifact output path is not a directory: {resolved_output}"
        )
    if resolved_output.is_dir() and any(
        entry.name != "training.log" or not entry.is_file()
        for entry in resolved_output.iterdir()
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifacts: {resolved_output}"
        )
    return resolved_output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed",
        type=_non_negative_seed,
        default=42,
        help="Seed PPO initialization and training randomness (default: 42).",
    )
    parser.add_argument(
        "--scenario-seed",
        type=_non_negative_seed,
        default=42,
        help="Root seed for deterministic train, validation, and test scenarios.",
    )
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
    args = parser.parse_args(argv)

    repository_artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
    randomized_default = repository_artifacts / "ppo_randomized_horizon" / "seed42"
    no_time_default = repository_artifacts / "ppo_without_time" / "seed42"
    default_output = repository_artifacts / "ppo"
    if args.randomized_training_horizons:
        default_output = randomized_default
    elif args.observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME:
        default_output = no_time_default
    output_dir = Path(
        args.output_dir or os.environ.get("PPO_ARTIFACT_DIR", str(default_output))
    ).expanduser()
    legacy_output = repository_artifacts / "ppo"
    randomized_output = (repository_artifacts / "ppo_randomized_horizon").resolve()
    if args.randomized_training_horizons and (
        args.observation_schema != OBSERVATION_SCHEMA_WITH_TIME
    ):
        parser.error(
            "randomized training horizons and the no-time observation schema "
            "must use separate configurations"
        )
    if args.randomized_training_horizons:
        protected_directories = (legacy_output,)
    elif args.observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME:
        protected_directories = (legacy_output, randomized_output)
    else:
        protected_directories = ()
    output_dir = _validate_output_directory(
        output_dir,
        legacy_output=legacy_output,
        protected_directories=protected_directories,
    )
    if not args.randomized_training_horizons and (
        args.min_episode_decisions != 11
        or args.max_episode_decisions is not None
        or args.horizon_seed is not None
    ):
        parser.error("horizon options require --randomized-training-horizons")

    training_seed = args.training_seed
    scenario_seed = args.scenario_seed
    config = PPOConfig(seed=training_seed, device="cpu")
    environments = build_ppo_environments(
        seed=scenario_seed,
        randomize_training_horizons=args.randomized_training_horizons,
        min_episode_decisions=args.min_episode_decisions,
        max_episode_decisions=args.max_episode_decisions,
        horizon_seed=args.horizon_seed,
        observation_schema=args.observation_schema,
    )
    orchestrator = TrainingOrchestrator(
        config=config,
        seed=training_seed,
        output_dir=output_dir,
    )
    result, metadata, model_path, manifest_path = orchestrator.train(
        environments.train_env,
        options=TrainingOptions(
            total_timesteps=10_000,
            seed=training_seed,
            verbose=0,
        ),
        experiment_name=(
            f"phase7-randomized-horizon-ppo-seed{training_seed}"
            if args.randomized_training_horizons
            else (
                f"phase7-no-time-observation-ppo-seed{training_seed}"
                if args.observation_schema == OBSERVATION_SCHEMA_WITHOUT_TIME
                else f"dashboard-synthetic-ppo-seed{training_seed}"
            )
        ),
    )
    metadata["training_seed"] = training_seed
    metadata["scenario_root_seed"] = scenario_seed
    metadata["scenario_splits"] = {
        split_name: {
            "scenario_id": getattr(environments.pipeline.datasets, split_name).metadata[
                "scenario_id"
            ],
            "seed": getattr(environments.pipeline.datasets, split_name).seed,
        }
        for split_name in ("train", "validation", "test")
    }
    if args.randomized_training_horizons:
        horizon_seed = scenario_seed if args.horizon_seed is None else args.horizon_seed
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
