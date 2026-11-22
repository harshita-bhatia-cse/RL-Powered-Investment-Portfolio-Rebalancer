"""Explicit synthetic PPO training orchestration for Phase 4."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym

from src.rl.ppo_agent import TrainingResult, save_ppo_model, train_ppo
from src.rl.ppo_config import PPOConfig, TrainingOptions
from src.rl.environment.synthetic_portfolio_env import (
    OBSERVATION_SCHEMA_VERSION,
    observation_size_for_schema,
)
from src.rl.phase7_experiment import (
    Phase72Run,
    build_phase72_run_metadata,
    validate_phase72_protocol,
    validate_phase72_run_directory,
    validate_phase72_run_metadata,
)
from src.rl.synthetic_pipeline import RandomizedEpisodeHorizonEnv


class HistoricalDataAuthorizationError(PermissionError):
    """Raised when historical training is attempted without explicit authorization."""


@dataclass(frozen=True)
class HistoricalDataAuthorization:
    """Explicit authorization record for a future historical training workflow."""

    authorized: bool = False
    reference: str | None = None
    approved_by: str | None = None

    def validate(self) -> None:
        if self.authorized is not True:
            raise HistoricalDataAuthorizationError(
                "Historical PPO training is blocked: data provenance and licensing "
                "are not authorized."
            )
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise HistoricalDataAuthorizationError(
                "Historical PPO training requires a non-empty authorization reference."
            )


class TrainingOrchestrator:
    """Run an explicit synthetic training and checkpoint workflow."""

    def __init__(
        self,
        *,
        config: PPOConfig,
        seed: int,
        output_dir: str | Path,
        historical_authorization: HistoricalDataAuthorization | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        config.validate()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.config = config
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.repository_root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.historical_authorization = (
            historical_authorization or HistoricalDataAuthorization()
        )

    def train(
        self,
        env: gym.Env,
        *,
        options: TrainingOptions,
        experiment_name: str = "phase4-synthetic",
        phase72_protocol: Mapping[str, Any] | None = None,
        phase72_run: Phase72Run | None = None,
    ) -> tuple[TrainingResult, dict[str, Any], Path, Path]:
        """Train only synthetic environments and save model/experiment metadata."""
        self._guard_environment(env)
        options.validate()
        if (phase72_protocol is None) != (phase72_run is None):
            raise ValueError(
                "phase72_protocol and phase72_run must be supplied together"
            )
        base_env = getattr(env, "unwrapped", env)
        run_metadata = None
        if phase72_protocol is not None and phase72_run is not None:
            run_metadata = self._validate_phase72_run(
                env,
                base_env=base_env,
                options=options,
                protocol=phase72_protocol,
                run=phase72_run,
            )
        checkpoint_dir = None
        if options.checkpoint_freq is not None:
            checkpoint_dir = str(
                Path(options.checkpoint_dir)
                if options.checkpoint_dir is not None
                else self.output_dir / "checkpoints"
            )
        result = train_ppo(
            env,
            total_timesteps=options.total_timesteps,
            config=self.config,
            seed=self.seed,
            tensorboard_log=options.tensorboard_log,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=options.checkpoint_freq,
            checkpoint_prefix=options.checkpoint_prefix,
            tb_log_name=options.tb_log_name,
            log_interval=options.log_interval,
            verbose=options.verbose,
        )

        data_metadata = dict(base_env.data_metadata)
        model_path = self.output_dir / "model"
        model_path, manifest_path = save_ppo_model(
            result.model,
            model_path,
            config=self.config,
            seed=self.seed,
            provenance=data_metadata,
            experiment_metadata=run_metadata,
        )

        metadata = {
            "experiment_name": experiment_name,
            "environment_type": type(base_env).__name__,
            "data_provenance": "synthetic",
            "historical_training_authorized": False,
            "seed": self.seed,
            "observation_schema": base_env.observation_schema,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_shape": [
                observation_size_for_schema(base_env.observation_schema)
            ],
            "ppo_config": asdict(self.config),
            "training_options": asdict(options),
            "training_metadata": result.metadata,
            "model": {
                "path": str(model_path),
                "manifest": str(manifest_path),
            },
        }
        if run_metadata is not None:
            metadata["phase72_experiment"] = run_metadata
        experiment_path = self.output_dir / "experiment.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        experiment_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result, metadata, model_path, manifest_path

    def _guard_environment(self, env: gym.Env) -> None:
        if not isinstance(env, gym.Env):
            raise TypeError("env must be a Gymnasium environment")
        base = getattr(env, "unwrapped", env)
        metadata = getattr(base, "data_metadata", {})
        source_type = metadata.get("source_type")
        synthetic = metadata.get("synthetic")

        if synthetic is True and source_type == "synthetic":
            return
        if source_type == "historical" or (
            synthetic is False
            or not getattr(base, "metadata", {}).get("synthetic", False)
        ):
            if not self.historical_authorization.authorized:
                raise HistoricalDataAuthorizationError(
                    "Historical PPO training is blocked: data provenance and licensing "
                    "have not been authorized."
                )
            self.historical_authorization.validate()
            raise HistoricalDataAuthorizationError(
                "Historical PPO training is not enabled in this Phase 4 workflow."
            )
        raise HistoricalDataAuthorizationError(
            "PPO training requires synthetic or explicitly identified historical provenance."
        )

    def _validate_phase72_run(
        self,
        env: gym.Env,
        *,
        base_env: Any,
        options: TrainingOptions,
        protocol: Mapping[str, Any],
        run: Phase72Run,
    ) -> dict[str, Any]:
        validate_phase72_protocol(protocol)
        validate_phase72_run_directory(
            protocol,
            run,
            repository_root=self.repository_root,
        )
        if run.training_seed != self.seed:
            raise ValueError("Phase 7.2 run seed does not match the orchestrator seed")
        if self.output_dir.resolve() != run.output_directory.resolve():
            raise ValueError("orchestrator output directory does not match the run plan")
        if (
            not self.output_dir.is_dir()
            or any(self.output_dir.iterdir())
        ):
            raise FileExistsError(
                "Phase 7.2 training requires its pre-reserved empty run directory"
            )
        if options.total_timesteps != protocol["shared_settings"]["training_budget"][
            "requested_total_timesteps"
        ]:
            raise ValueError("training timestep budget does not match the protocol")
        if options.seed not in (None, run.training_seed):
            raise ValueError("training options seed does not match the run seed")
        expected_metadata = build_phase72_run_metadata(
            protocol,
            configuration=run.configuration,
            training_seed=run.training_seed,
            training_provenance=base_env.data_metadata,
        )
        validate_phase72_run_metadata(
            protocol,
            expected_metadata,
            configuration=run.configuration,
            training_seed=run.training_seed,
            expected_training_provenance=base_env.data_metadata,
        )
        if json.loads(json.dumps(asdict(self.config), sort_keys=True)) != json.loads(
            json.dumps(expected_metadata["ppo_settings"], sort_keys=True)
        ):
            raise ValueError("PPO settings do not match the Phase 7.2 protocol")
        if (
            base_env.observation_schema != run.observation_schema
            or expected_metadata["observation_schema"] != run.observation_schema
        ):
            raise ValueError("training environment schema does not match the run plan")

        randomized_wrapper = _find_randomized_horizon_wrapper(env)
        if run.randomize_training_horizons:
            horizon = protocol["configurations"][run.configuration][
                "horizon_settings"
            ]
            if (
                run.configuration != "B"
                or randomized_wrapper is None
                or randomized_wrapper.min_episode_decisions
                != horizon["minimum_decisions"]
                or randomized_wrapper.max_episode_decisions
                != horizon["maximum_decisions"]
                or randomized_wrapper.horizon_seed != run.horizon_seed
            ):
                raise ValueError("training horizon wrapper does not match the protocol")
        elif randomized_wrapper is not None:
            raise ValueError("only Configuration B may use randomized training horizons")
        if run.randomize_training_horizons != (
            run.configuration == "B"
        ):
            raise ValueError("run configuration and horizon mode disagree")
        return expected_metadata


def _find_randomized_horizon_wrapper(
    env: gym.Env,
) -> RandomizedEpisodeHorizonEnv | None:
    current: Any = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, RandomizedEpisodeHorizonEnv):
            return current
        current = current.env
    return None
