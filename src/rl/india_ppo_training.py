"""India historical PPO preparation and guarded training helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from src.data.india_daily_preprocessing import (
    IndiaDailyDataset,
    IndiaFeatureScaler,
    load_indian_daily_dataset,
)
from src.rl.environment.india_daily_portfolio_env import (
    IndiaDailyPortfolioEnv,
)
from src.rl.ppo_config import PPOConfig, TrainingOptions


INDIA_ARTIFACT_ROOT = ".artifacts/india_daily_ppo"


class IndiaDataNotReadyError(PermissionError):
    """Raised when India data is unprovenanced or not training-ready."""


@dataclass(frozen=True)
class IndiaHistoricalAuthorization:
    """Explicit future-training authorization record."""

    authorized: bool = False
    reference: str | None = None
    approved_by: str | None = None

    def validate(self) -> None:
        if self.authorized is not True:
            raise IndiaDataNotReadyError(
                "India PPO training is blocked: explicit training authorization is absent."
            )
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise IndiaDataNotReadyError(
                "India PPO training requires a non-empty authorization reference."
            )
        if not isinstance(self.approved_by, str) or not self.approved_by.strip():
            raise IndiaDataNotReadyError(
                "India PPO training requires an approving authority."
            )


@dataclass(frozen=True)
class IndiaPPOEnvironmentBundle:
    dataset: IndiaDailyDataset
    scaler: IndiaFeatureScaler
    train_env: gym.Env
    validation_env: gym.Env
    test_env: gym.Env


@dataclass(frozen=True)
class IndiaPPOPreparation:
    bundle: IndiaPPOEnvironmentBundle
    output_dir: Path
    manifest: dict[str, object]


def prepare_india_ppo_training(
    *,
    validated_csv: str | Path,
    manifest_json: str | Path,
    symbols: tuple[str, ...],
    seed: int,
    ppo_config: PPOConfig | None = None,
    training_options: TrainingOptions | None = None,
    output_dir: str | Path | None = None,
    initial_capital: float = 100_000.0,
    transaction_cost_rate: float = 0.001,
    slippage_bps: float = 5.0,
    repository_root: str | Path | None = None,
) -> IndiaPPOPreparation:
    """Preflight India PPO environments without instantiating or training PPO."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    selected_config = ppo_config or PPOConfig(seed=seed, device="cpu")
    selected_config.validate()
    if training_options is not None:
        training_options.validate()

    try:
        preflight_manifest = json.loads(
            Path(manifest_json).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise IndiaDataNotReadyError(
            "India PPO preflight requires a readable provenance manifest."
        ) from error
    if not isinstance(preflight_manifest, dict):
        raise IndiaDataNotReadyError(
            "India PPO preflight manifest must be a JSON object."
        )
    if preflight_manifest.get("training_ready") is not True:
        raise IndiaDataNotReadyError(
            "India data is not training-ready; provider, permission or validation provenance is insufficient."
        )
    if preflight_manifest.get("license_permission_status") != "approved":
        raise IndiaDataNotReadyError(
            "India data requires explicit approved license/permission status."
        )

    try:
        dataset = load_indian_daily_dataset(
            validated_csv,
            manifest_json=manifest_json,
            symbols=symbols,
        )
    except ValueError as error:
        try:
            readiness_manifest = json.loads(
                Path(manifest_json).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            readiness_manifest = {}
        if (
            isinstance(readiness_manifest, dict)
            and (
                readiness_manifest.get("training_ready") is not True
                or readiness_manifest.get("license_permission_status") != "approved"
            )
        ):
            raise IndiaDataNotReadyError(
                "India data is not training-ready or permission is not approved."
            ) from error
        raise
    manifest = dataset.manifest
    if manifest.get("training_ready") is not True:
        raise IndiaDataNotReadyError(
            "India data is not training-ready; provider, permission or adjustment provenance is insufficient."
        )
    if manifest.get("license_permission_status") != "approved":
        raise IndiaDataNotReadyError(
            "India data requires explicit approved license/permission status."
        )
    if not str(manifest.get("retrieval_timestamp", "")).strip():
        raise IndiaDataNotReadyError(
            "India data manifest is missing its retrieval timestamp."
        )

    train_features = dataset.features.loc[
        (dataset.features.index >= dataset.splits.train[0])
        & (dataset.features.index <= dataset.splits.train[1])
    ]
    scaler = IndiaFeatureScaler.fit(train_features)

    train_env = IndiaDailyPortfolioEnv(
        dataset,
        partition="train",
        scaler=scaler,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        slippage_bps=slippage_bps,
    )
    validation_env = IndiaDailyPortfolioEnv(
        dataset,
        partition="validation",
        scaler=scaler,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        slippage_bps=slippage_bps,
    )
    test_env = IndiaDailyPortfolioEnv(
        dataset,
        partition="test",
        scaler=scaler,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        slippage_bps=slippage_bps,
    )

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    selected_output = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else root / INDIA_ARTIFACT_ROOT
    ).resolve()
    artifact_root = (root / ".artifacts").resolve()
    india_root = (root / INDIA_ARTIFACT_ROOT).resolve()
    legacy_root = (root / ".artifacts" / "ppo").resolve()
    if selected_output == artifact_root or artifact_root not in selected_output.parents:
        raise ValueError("India PPO artifacts must be inside the repository .artifacts directory")
    if selected_output == legacy_root or legacy_root in selected_output.parents:
        raise ValueError("India PPO artifacts must not use or overwrite .artifacts/ppo")
    if selected_output != india_root:
        raise ValueError("India PPO output must use .artifacts/india_daily_ppo")
    if selected_output.exists() and any(selected_output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing India PPO artifacts: {selected_output}"
        )

    for environment in (train_env, validation_env, test_env):
        observation, _ = environment.reset(seed=seed)
        if observation.shape != environment.observation_space.shape:
            raise RuntimeError("India environment observation shape mismatch")
        if not np.isfinite(observation).all():
            raise RuntimeError("India environment reset produced non-finite observation")
        if not environment.observation_space.contains(observation):
            raise RuntimeError("India environment reset observation is outside its space")

    preparation_manifest: dict[str, object] = {
        "source_type": "india_historical_csv",
        "synthetic": False,
        "training_ready": True,
        "symbols": list(dataset.symbols),
        "seed": seed,
        "ppo_config": asdict(selected_config),
        "training_options": (
            asdict(training_options) if training_options is not None else None
        ),
        "observation_shape": [int(train_env.observation_space.shape[0])],
        "action_shape": [int(train_env.action_space.shape[0])],
        "splits": {
            "train": [dataset.splits.train[0].isoformat(), dataset.splits.train[1].isoformat()],
            "validation": [
                dataset.splits.validation[0].isoformat(),
                dataset.splits.validation[1].isoformat(),
            ],
            "test": [dataset.splits.test[0].isoformat(), dataset.splits.test[1].isoformat()],
        },
        "environment_settings": {
            "initial_capital": float(initial_capital),
            "transaction_cost_rate": float(transaction_cost_rate),
            "slippage_bps": float(slippage_bps),
        },
        "data_manifest": manifest,
        "output_directory": str(selected_output),
        "training_started": False,
    }

    bundle = IndiaPPOEnvironmentBundle(
        dataset=dataset,
        scaler=scaler,
        train_env=train_env,
        validation_env=validation_env,
        test_env=test_env,
    )
    return IndiaPPOPreparation(
        bundle=bundle,
        output_dir=selected_output,
        manifest=preparation_manifest,
    )


def create_india_ppo_agent(
    preparation: IndiaPPOEnvironmentBundle,
    *,
    config: PPOConfig,
    verbose: int = 0,
) -> Any:
    """Create an SB3 PPO agent for future training without calling learn()."""
    config.validate()
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1 or 2")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise IndiaPPOTrainingBlockedError(
            "Stable-Baselines3 is unavailable; agent creation was blocked."
        ) from error
    train_env = Monitor(preparation.train_env)
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": list(config.net_arch)},
        seed=config.seed,
        device=config.device,
        verbose=verbose,
    )
    model.india_training_ready = True
    model.india_observation_shape = preparation.train_env.observation_space.shape
    model.india_action_shape = preparation.train_env.action_space.shape
    return model


class IndiaPPOTrainingBlockedError(RuntimeError):
    """Raised when an India PPO operation is unsafe or unsupported."""


def train_india_ppo(
    preparation: IndiaPPOPreparation,
    *,
    authorization: IndiaHistoricalAuthorization,
    config: PPOConfig,
    training_options: TrainingOptions,
) -> dict[str, object]:
    """Safely start India PPO learning only after explicit authorization."""
    authorization.validate()
    config.validate()
    training_options.validate()
    if not preparation.manifest.get("training_ready"):
        raise IndiaDataNotReadyError("India training manifest is not training-ready")
    if preparation.output_dir.exists() and any(preparation.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing India PPO artifacts: {preparation.output_dir}"
        )

    model = create_india_ppo_agent(
        preparation.bundle,
        config=config,
        verbose=training_options.verbose,
    )
    model.learn(
        total_timesteps=training_options.total_timesteps,
        tb_log_name=training_options.tb_log_name,
        log_interval=training_options.log_interval,
    )
    preparation.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = preparation.output_dir / "model.zip"
    model.save(str(model_path))
    manifest = dict(preparation.manifest)
    manifest["training_started"] = True
    manifest["training_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["authorization"] = {
        "reference": authorization.reference,
        "approved_by": authorization.approved_by,
    }
    manifest["model_sha256"] = _sha256(model_path)
    manifest_path = preparation.output_dir / "model.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "model": model,
        "model_path": model_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
