"""Explicit synthetic PPO training orchestration for Phase 4."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

from src.rl.ppo_agent import TrainingResult, save_ppo_model, train_ppo
from src.rl.ppo_config import PPOConfig, TrainingOptions


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
    ) -> None:
        config.validate()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.config = config
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.historical_authorization = (
            historical_authorization or HistoricalDataAuthorization()
        )

    def train(
        self,
        env: gym.Env,
        *,
        options: TrainingOptions,
        experiment_name: str = "phase4-synthetic",
    ) -> tuple[TrainingResult, dict[str, Any], Path, Path]:
        """Train only synthetic environments and save model/experiment metadata."""
        self._guard_environment(env)
        options.validate()
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

        base_env = getattr(env, "unwrapped", env)
        data_metadata = dict(base_env.data_metadata)
        model_path = self.output_dir / "model"
        model_path, manifest_path = save_ppo_model(
            result.model,
            model_path,
            config=self.config,
            seed=self.seed,
            provenance=data_metadata,
        )

        metadata = {
            "experiment_name": experiment_name,
            "environment_type": type(base_env).__name__,
            "data_provenance": "synthetic",
            "historical_training_authorized": False,
            "seed": self.seed,
            "ppo_config": asdict(self.config),
            "training_options": asdict(options),
            "training_metadata": result.metadata,
            "model": {
                "path": str(model_path),
                "manifest": str(manifest_path),
            },
        }
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
