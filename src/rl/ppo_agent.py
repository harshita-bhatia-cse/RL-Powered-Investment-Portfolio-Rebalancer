"""Stable-Baselines3 PPO plumbing for synthetic portfolio environments.

Importing this module does not import Stable-Baselines3, instantiate PPO, or
start training. The optional dependency is loaded only by functions that need
it. Training is performed only when ``train_ppo`` is explicitly called.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from src.rl.environment.synthetic_portfolio_env import OBSERVATION_SIZE
from src.rl.environment.portfolio_env import ASSET_UNIVERSE
from src.rl.ppo_config import PPOConfig, TrainingOptions
from src.rl.synthetic_data import GENERATOR_VERSION, SyntheticMarketConfig
from src.rl.synthetic_pipeline import SyntheticPipeline, build_synthetic_pipeline

MODEL_MANIFEST_VERSION = "1.0"
_PROVENANCE_KEYS = (
    "scenario_id",
    "seed",
    "generator_version",
    "configuration_hash",
    "generated_at_utc",
    "generation_config",
)


@dataclass(frozen=True)
class PPOEnvironmentBundle:
    """Isolated train, validation, and test datasets/environments."""

    pipeline: SyntheticPipeline
    train_env: gym.Env
    validation_env: gym.Env
    test_env: gym.Env


@dataclass(frozen=True)
class EvaluationStep:
    """One deterministic inference and resulting environment transition."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    nav: float
    weights: dict[str, float]
    gross_turnover: float
    one_way_turnover: float
    transaction_fees: float
    slippage_cost: float
    transaction_cost: float
    terminated: bool
    truncated: bool
    daily_navs: tuple[float, ...] = ()
    requested_weights: dict[str, float] | None = None
    applied_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Structured synthetic episode evaluation output."""

    source_type: str
    synthetic: bool
    scenario_id: str
    seed: int
    initial_observation: np.ndarray
    steps: tuple[EvaluationStep, ...]


@dataclass(frozen=True)
class TrainingResult:
    """Trained PPO model and runtime metadata returned by explicit training."""

    model: Any
    metadata: dict[str, object]


def build_ppo_environments(
    config: SyntheticMarketConfig | None = None,
    *,
    seed: int,
) -> PPOEnvironmentBundle:
    """Create a monitored training env and separate untouched eval envs."""
    pipeline = build_synthetic_pipeline(config, seed=seed)
    monitor_type = _monitor_type()
    train_env = monitor_type(pipeline.environments["train"])
    return PPOEnvironmentBundle(
        pipeline=pipeline,
        train_env=train_env,
        validation_env=pipeline.environments["validation"],
        test_env=pipeline.environments["test"],
    )


def create_ppo_agent(
    env: gym.Env,
    config: PPOConfig | None = None,
    *,
    seed: int | None = None,
    tensorboard_log: str | None = None,
    verbose: int = 0,
) -> Any:
    """Initialize an SB3 PPO model after checking environment dimensions."""
    selected_config = config or PPOConfig()
    selected_config.validate()
    _validate_environment(env)
    effective_seed = selected_config.seed if seed is None else _validate_seed(seed)
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2")
    kwargs = _algorithm_kwargs(
        selected_config, effective_seed, tensorboard_log, verbose
    )
    ppo_type = _ppo_type()
    return ppo_type("MlpPolicy", env, **kwargs)


def train_ppo(
    env: gym.Env,
    *,
    total_timesteps: int,
    config: PPOConfig | None = None,
    seed: int | None = None,
    tensorboard_log: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int | None = None,
    checkpoint_prefix: str = "synthetic_ppo",
    tb_log_name: str = "synthetic_ppo",
    log_interval: int = 1,
    verbose: int = 1,
) -> TrainingResult:
    """Train PPO when explicitly called; this function is not run by imports.

    This phase adds the training interface but deliberately does not call it.
    """
    selected_config = config or PPOConfig()
    selected_config.validate()
    options = TrainingOptions(
        total_timesteps=total_timesteps,
        seed=seed,
        tensorboard_log=tensorboard_log,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=checkpoint_freq,
        checkpoint_prefix=checkpoint_prefix,
        tb_log_name=tb_log_name,
        log_interval=log_interval,
        verbose=verbose,
    )
    options.validate()
    _validate_environment(env)
    effective_seed = selected_config.seed if seed is None else _validate_seed(seed)
    model = create_ppo_agent(
        env,
        selected_config,
        seed=effective_seed,
        tensorboard_log=options.tensorboard_log,
        verbose=options.verbose,
    )

    callbacks = None
    if options.checkpoint_dir is not None:
        checkpoint_path = Path(options.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        callback_type = _checkpoint_callback_type()
        callbacks = callback_type(
            save_freq=options.checkpoint_freq,
            save_path=str(checkpoint_path),
            name_prefix=options.checkpoint_prefix,
        )
    model.learn(
        total_timesteps=options.total_timesteps,
        callback=callbacks,
        tb_log_name=options.tb_log_name,
        log_interval=options.log_interval,
    )
    return TrainingResult(
        model=model,
        metadata={
            "source_type": "synthetic",
            "synthetic": True,
            "seed": effective_seed,
            "training_scenario_id": _base_environment(env).data_metadata[
                "scenario_id"
            ],
            "training_scenario_seed": _base_environment(env).data_metadata["seed"],
            "total_timesteps": options.total_timesteps,
            "ppo_config": asdict(selected_config),
            "training_options": asdict(options),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def evaluate_ppo(
    model: Any,
    env: gym.Env,
    *,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> EvaluationResult:
    """Run deterministic synthetic predictions and collect transitions."""
    if deterministic is not True:
        raise ValueError("synthetic pipeline evaluation requires deterministic=True")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer when supplied")
    _validate_environment(env)
    base_env = _base_environment(env)
    metadata = base_env.data_metadata
    if metadata.get("synthetic") is not True or metadata.get("source_type") != "synthetic":
        raise ValueError("PPO evaluation accepts synthetic environments only")

    observation, _ = env.reset(seed=metadata.get("seed"))
    initial_observation = np.array(observation, dtype=np.float32, copy=True)
    initial_observation.setflags(write=False)
    steps: list[EvaluationStep] = []
    while max_steps is None or len(steps) < max_steps:
        action, _ = model.predict(observation, deterministic=True)
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape == (1, env.action_space.shape[0]):
            action_array = action_array[0]
        if action_array.shape != env.action_space.shape or not np.isfinite(action_array).all():
            raise ValueError("model prediction must be a finite seven-value action")
        if not env.action_space.contains(action_array):
            raise ValueError("model prediction is outside the environment action space")
        next_observation, reward, terminated, truncated, info = env.step(action_array)
        daily_navs = _daily_navs(base_env, info)
        observation_copy = np.array(next_observation, dtype=np.float32, copy=True)
        action_copy = np.array(action_array, dtype=np.float32, copy=True)
        observation_copy.setflags(write=False)
        action_copy.setflags(write=False)
        steps.append(
            EvaluationStep(
                observation=observation_copy,
                action=action_copy,
                reward=float(reward),
                nav=float(info["nav"]),
                weights=dict(info["weights"]),
                requested_weights=dict(info["requested_weights"]),
                applied_weights=dict(info["applied_weights"]),
                gross_turnover=float(info["gross_turnover"]),
                one_way_turnover=float(info["one_way_turnover"]),
                transaction_fees=float(info["transaction_fees"]),
                slippage_cost=float(info["slippage_cost"]),
                transaction_cost=float(info["transaction_cost"]),
                terminated=terminated,
                truncated=truncated,
                daily_navs=daily_navs,
            )
        )
        observation = next_observation
        if terminated or truncated:
            break
    return EvaluationResult(
        source_type="synthetic",
        synthetic=True,
        scenario_id=str(metadata.get("scenario_id", "")),
        seed=int(metadata["seed"]),
        initial_observation=initial_observation,
        steps=tuple(steps),
    )


def save_ppo_model(
    model: Any,
    path: str | os.PathLike[str],
    *,
    config: PPOConfig,
    seed: int,
    provenance: Mapping[str, object],
) -> tuple[Path, Path]:
    """Save model and synthetic-only JSON manifest with a model checksum."""
    config.validate()
    normalized_seed = _validate_seed(seed)
    _validate_synthetic_provenance(provenance)
    _validate_model_dimensions(model)
    model_path, manifest_path = _model_paths(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    if not model_path.is_file():
        raise OSError(f"model save did not create expected artifact: {model_path}")
    manifest = _build_manifest(
        model_path=model_path,
        config=config,
        seed=normalized_seed,
        provenance=provenance,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_path, manifest_path


def load_ppo_model(
    path: str | os.PathLike[str],
    *,
    env: gym.Env | None = None,
    expected_config: PPOConfig | None = None,
) -> Any:
    """Validate sidecar/checksum and load a compatible synthetic PPO artifact."""
    model_path, manifest_path = _model_paths(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO model artifact not found: {model_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PPO model manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"PPO model manifest is unreadable: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("PPO model manifest must be a JSON object")
    _validate_manifest(manifest, model_path, expected_config)
    if env is not None:
        _validate_environment(env)
    model_type = _ppo_type()
    model = model_type.load(str(model_path), env=env)
    _validate_model_dimensions(model)
    return model


def _build_manifest(
    *,
    model_path: Path,
    config: PPOConfig,
    seed: int,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    try:
        import stable_baselines3
        import torch
    except ImportError as error:
        raise ImportError(
            "Stable-Baselines3 and Torch are required to save PPO model metadata"
        ) from error
    selected_provenance = {
        key: _json_compatible(provenance[key])
        for key in _PROVENANCE_KEYS
        if key in provenance
    }
    return {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "source_type": "synthetic",
        "synthetic": True,
        "seed": seed,
        "asset_order": list(ASSET_UNIVERSE),
        "observation_shape": [OBSERVATION_SIZE],
        "action_shape": [len(ASSET_UNIVERSE) + 1],
        "generator_version": str(
            provenance.get("generator_version", GENERATOR_VERSION)
        ),
        "provenance": selected_provenance,
        "ppo_config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
        },
        "model_sha256": _sha256_file(model_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _validate_manifest(
    manifest: Mapping[str, object],
    model_path: Path,
    expected_config: PPOConfig | None,
) -> None:
    if manifest.get("manifest_version") != MODEL_MANIFEST_VERSION:
        raise ValueError("unsupported PPO model manifest version")
    if manifest.get("source_type") != "synthetic" or manifest.get("synthetic") is not True:
        raise ValueError("model artifact is not marked synthetic-only")
    if manifest.get("asset_order") != list(ASSET_UNIVERSE):
        raise ValueError("model manifest asset order is incompatible")
    if manifest.get("observation_shape") != [OBSERVATION_SIZE]:
        raise ValueError("model manifest observation shape is incompatible")
    if manifest.get("action_shape") != [len(ASSET_UNIVERSE) + 1]:
        raise ValueError("model manifest action shape is incompatible")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("synthetic") not in (None, True):
        raise ValueError("model provenance metadata must remain synthetic")
    _validate_seed(manifest.get("seed"))
    if provenance.get("seed") is not None:
        _validate_seed(provenance.get("seed"))
    expected_checksum = manifest.get("model_sha256")
    if not isinstance(expected_checksum, str) or expected_checksum != _sha256_file(model_path):
        raise ValueError("PPO model checksum does not match its manifest")
    stored_config = manifest.get("ppo_config")
    if not isinstance(stored_config, dict):
        raise ValueError("PPO model manifest has no valid configuration")
    if expected_config is not None:
        expected_config.validate()
        if stored_config != _json_compatible(asdict(expected_config)):
            raise ValueError("PPO model configuration does not match expected_config")


def _validate_synthetic_provenance(
    provenance: Mapping[str, object],
) -> None:
    if provenance.get("source_type") != "synthetic" or provenance.get("synthetic") is not True:
        raise ValueError("model persistence requires synthetic-only provenance")
    if provenance.get("seed") is not None:
        _validate_seed(provenance.get("seed"))
    for key in _PROVENANCE_KEYS:
        value = provenance.get(key)
        if key == "generation_config":
            if value is not None and not isinstance(value, Mapping):
                raise ValueError("generation_config provenance must be a mapping")
            if value is not None and not set(value).issubset(
                {
                    "sessions",
                    "correlation_matrix",
                    "regimes",
                    "transition_matrix",
                    "initial_regime_probabilities",
                    "initial_vix_level",
                    "vix_mean_reversion_speed",
                    "vix_min_level",
                    "vix_max_level",
                    "vix_log_level_clip",
                }
            ):
                raise ValueError("generation_config contains unsupported fields")
            if value is not None:
                try:
                    _json_compatible(dict(value))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "generation_config must be JSON serializable"
                    ) from error
            continue
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"provenance field {key!r} must be a JSON scalar")


def _model_paths(path: str | os.PathLike[str]) -> tuple[Path, Path]:
    model_path = Path(path)
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    return model_path, model_path.with_suffix(".manifest.json")


def _validate_environment(env: gym.Env) -> None:
    if not isinstance(env, gym.Env):
        raise TypeError("env must be a Gymnasium environment")
    observation_shape = getattr(env.observation_space, "shape", None)
    action_shape = getattr(env.action_space, "shape", None)
    if observation_shape != (OBSERVATION_SIZE,):
        raise ValueError("environment must expose a 43-dimensional observation space")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("environment must expose a continuous Box action space")
    if action_shape != (len(ASSET_UNIVERSE) + 1,):
        raise ValueError("environment must expose a seven-dimensional action space")
    base_env = _base_environment(env)
    if (
        getattr(base_env, "metadata", {}).get("synthetic") is not True
        or getattr(base_env, "data_metadata", {}).get("synthetic") is not True
        or getattr(base_env, "data_metadata", {}).get("source_type") != "synthetic"
    ):
        raise ValueError("PPO environment must use synthetic data")


def _validate_model_dimensions(model: Any) -> None:
    if getattr(getattr(model, "observation_space", None), "shape", None) != (
        OBSERVATION_SIZE,
    ):
        raise ValueError("PPO model must have a 43-dimensional observation space")
    if getattr(getattr(model, "action_space", None), "shape", None) != (
        len(ASSET_UNIVERSE) + 1,
    ):
        raise ValueError("PPO model must have a seven-dimensional action space")
    if not isinstance(getattr(model, "action_space", None), gym.spaces.Box):
        raise ValueError("PPO model must use a continuous Box action space")


def _base_environment(env: gym.Env) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    return unwrapped


def _daily_navs(env: Any, info: Mapping[str, Any]) -> tuple[float, ...]:
    """Reconstruct the exact daily marked NAV path for one rebalance interval."""
    start = int(info["session_start"])
    end = int(info["session_end"])
    returns = np.asarray(env.asset_returns[start:end], dtype=np.float64)
    if returns.ndim != 2 or len(returns) != end - start or len(returns) == 0:
        raise ValueError("evaluation transition has an invalid daily return interval")

    weight_map = info["applied_weights"]
    weights = np.asarray(
        [*(weight_map[name] for name in ASSET_UNIVERSE), weight_map["CASH"]],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("evaluation transition has invalid applied weights")

    nav = float(info["post_cost_nav"])
    asset_growth = np.ones(returns.shape[1], dtype=np.float64)
    daily_navs: list[float] = []
    for session_returns in returns:
        asset_growth *= 1.0 + session_returns
        nav = float(
            info["post_cost_nav"]
            * (np.dot(weights[:-1], asset_growth) + weights[-1])
        )
        if not np.isfinite(nav) or nav <= 0.0:
            raise FloatingPointError("daily PPO evaluation NAV is invalid")
        daily_navs.append(nav)

    if not np.isclose(daily_navs[-1], float(info["nav"]), rtol=1e-10, atol=1e-7):
        raise RuntimeError("daily PPO evaluation NAV does not match the environment")
    return tuple(daily_navs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_compatible(value: object) -> object:
    return json.loads(json.dumps(value))


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _algorithm_kwargs(
    config: PPOConfig,
    seed: int,
    tensorboard_log: str | None,
    verbose: int,
) -> dict[str, Any]:
    from torch.nn import ReLU

    return {
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "learning_rate": config.learning_rate,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": config.max_grad_norm,
        "seed": seed,
        "device": config.device,
        "tensorboard_log": tensorboard_log,
        "verbose": verbose,
        "policy_kwargs": {
            "net_arch": list(config.net_arch),
            "activation_fn": ReLU,
        },
    }


def _ppo_type() -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError(
            "Install the project's optional 'ml' dependencies to use PPO"
        ) from error
    return PPO


def _monitor_type() -> Any:
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise ImportError(
            "Install the project's optional 'ml' dependencies to wrap PPO environments"
        ) from error
    return Monitor


def _checkpoint_callback_type() -> Any:
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as error:
        raise ImportError(
            "Install the project's optional 'ml' dependencies for checkpoint callbacks"
        ) from error
    return CheckpointCallback
