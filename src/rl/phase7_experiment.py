"""Safe Phase 7.2 synthetic PPO experiment protocol and run planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from src.rl.environment.synthetic_portfolio_env import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_ASSET_WEIGHT,
    DEFAULT_MAX_TURNOVER,
    DEFAULT_MIN_CASH_WEIGHT,
    DEFAULT_OBSERVATION_SCHEMA,
    DEFAULT_PERIOD_SESSIONS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSACTION_COST_RATE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_WITHOUT_TIME,
    OBSERVATION_WARMUP_SESSIONS,
    observation_size_for_schema,
)
from src.rl.ppo_config import PPOConfig
from src.rl.synthetic_data import (
    GENERATOR_VERSION,
    RegimeConfig,
    SyntheticMarketConfig,
)

PROTOCOL_ID = "phase7_2_synthetic_ppo_comparison"
PROTOCOL_VERSION = 1
CONFIGURATIONS = ("A", "B", "C")
TRAINING_SEEDS = (101, 202, 303, 404, 505)
VALIDATION_ROOTS = tuple(range(7201, 7221))
RESERVED_TEST_ROOTS = tuple(range(9201, 9221))
_PROVENANCE_FIELDS = (
    "scenario_id",
    "seed",
    "generator_version",
    "configuration_hash",
    "generated_at_utc",
    "generation_config",
)


@dataclass(frozen=True)
class Phase72Run:
    """One validated configuration/seed output directory."""

    configuration: str
    training_seed: int
    output_directory: Path
    observation_schema: str
    randomize_training_horizons: bool
    horizon_seed: int | None
    reservation_token: str | None = None


def default_protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "phase7_2_experiment_protocol.json"
    )


def load_phase72_protocol(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and fully validate the checked-in Phase 7.2 protocol."""
    protocol_path = default_protocol_path() if path is None else Path(path)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"experiment protocol is unreadable: {protocol_path}") from error
    if not isinstance(protocol, dict):
        raise ValueError("experiment protocol must be a JSON object")
    validate_phase72_protocol(protocol)
    return protocol


def validate_phase72_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject protocol drift, invalid seeds, and unmatched A/B/C settings."""
    _require_keys(
        protocol,
        {"protocol_id", "protocol_version", "seed_roles", "shared_settings", "configurations"},
        "protocol",
    )
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unsupported experiment protocol id")
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported experiment protocol version")

    seed_roles = protocol["seed_roles"]
    _require_keys(
        seed_roles,
        {
            "training_seeds",
            "validation_scenario_roots",
            "reserved_test_scenario_roots",
        },
        "seed_roles",
    )
    role_values = {
        "training_seeds": _seed_list(seed_roles["training_seeds"], "training_seeds"),
        "validation_scenario_roots": _seed_list(
            seed_roles["validation_scenario_roots"],
            "validation_scenario_roots",
        ),
        "reserved_test_scenario_roots": _seed_list(
            seed_roles["reserved_test_scenario_roots"],
            "reserved_test_scenario_roots",
        ),
    }
    role_sets = [set(values) for values in role_values.values()]
    if any(
        role_sets[left].intersection(role_sets[right])
        for left in range(len(role_sets))
        for right in range(left + 1, len(role_sets))
    ):
        raise ValueError("training, validation, and test seed roles must not overlap")
    if tuple(role_values["training_seeds"]) != TRAINING_SEEDS:
        raise ValueError("training seed set does not match the Phase 7.2 protocol")
    if tuple(role_values["validation_scenario_roots"]) != VALIDATION_ROOTS:
        raise ValueError("validation roots do not match the Phase 7.2 protocol")
    if tuple(role_values["reserved_test_scenario_roots"]) != RESERVED_TEST_ROOTS:
        raise ValueError("test roots do not match the Phase 7.2 protocol")

    shared = protocol["shared_settings"]
    _require_keys(
        shared,
        {
            "observation_schema_version",
            "ppo_settings",
            "training_budget",
            "synthetic_generator",
            "environment",
            "evaluation",
        },
        "shared_settings",
    )
    if shared["observation_schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("protocol observation schema version is unsupported")
    _validate_ppo_settings(shared["ppo_settings"], role_values["training_seeds"][0])
    _validate_training_budget(shared["training_budget"])
    generator = _synthetic_config_from_json(shared["synthetic_generator"])
    if shared["synthetic_generator"].get("generator_version") != GENERATOR_VERSION:
        raise ValueError("protocol synthetic generator version is incompatible")
    _validate_environment_settings(shared["environment"])
    _validate_evaluation_settings(
        shared["evaluation"],
        generator=generator,
        environment=shared["environment"],
        validation_count=len(role_values["validation_scenario_roots"]),
        test_count=len(role_values["reserved_test_scenario_roots"]),
    )
    _validate_configurations(
        protocol["configurations"],
        role_values["training_seeds"],
        generator=generator,
    )


def build_phase72_run_matrix(
    protocol: Mapping[str, Any],
    *,
    output_root: str | Path,
    repository_root: str | Path,
) -> tuple[Phase72Run, ...]:
    """Plan all 15 run paths and validate the entire matrix before use."""
    validate_phase72_protocol(protocol)
    repo = Path(repository_root).resolve()
    artifacts_root = (repo / ".artifacts").resolve()
    resolved_root = Path(output_root).resolve()
    try:
        relative_root = resolved_root.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("experiment output root must be inside the repository .artifacts directory") from error
    if not relative_root.parts:
        raise ValueError("experiment output root must be a dedicated .artifacts subdirectory")

    protected_model_dir = (artifacts_root / "ppo").resolve()
    if (
        resolved_root == protected_model_dir
        or protected_model_dir in resolved_root.parents
    ):
        raise ValueError("Phase 7.2 may never use the existing .artifacts/ppo directory")
    if resolved_root.exists() and not resolved_root.is_dir():
        raise FileExistsError(f"experiment output root is not a directory: {resolved_root}")
    if resolved_root.is_dir() and any(resolved_root.iterdir()):
        raise FileExistsError(
            f"refusing to reuse a non-empty experiment output root: {resolved_root}"
        )

    runs: list[Phase72Run] = []
    configurations = protocol["configurations"]
    training_seeds = protocol["seed_roles"]["training_seeds"]
    for configuration in CONFIGURATIONS:
        specification = configurations[configuration]
        for training_seed in training_seeds:
            horizon_settings = specification["horizon_settings"]
            horizon_seed = (
                horizon_settings["sampler_seed_by_training_seed"][str(training_seed)]
                if horizon_settings is not None
                else None
            )
            run_directory = (
                resolved_root
                / f"config_{configuration}"
                / f"seed_{training_seed}"
            ).resolve()
            if (
                run_directory == protected_model_dir
                or protected_model_dir in run_directory.parents
            ):
                raise ValueError("a Phase 7.2 run path overlaps .artifacts/ppo")
            for parent in run_directory.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"output path parent is not a directory: {parent}")
            if run_directory.exists() and (
                not run_directory.is_dir() or any(run_directory.iterdir())
            ):
                raise FileExistsError(
                    f"refusing to use a non-empty or non-directory run path: {run_directory}"
                )
            runs.append(
                Phase72Run(
                    configuration=configuration,
                    training_seed=training_seed,
                    output_directory=run_directory,
                    observation_schema=specification["observation_schema"],
                    randomize_training_horizons=specification[
                        "randomize_training_horizons"
                    ],
                    horizon_seed=horizon_seed,
                )
            )
    if len({run.output_directory for run in runs}) != len(runs):
        raise ValueError("experiment matrix contains duplicate output paths")
    return tuple(runs)


def reserve_phase72_output_directories(
    protocol: Mapping[str, Any],
    *,
    output_root: str | Path,
    repository_root: str | Path,
) -> tuple[Phase72Run, ...]:
    """Preflight every run path, then create all unique empty run directories."""
    runs = build_phase72_run_matrix(
        protocol,
        output_root=output_root,
        repository_root=repository_root,
    )
    existing = [run.output_directory for run in runs if run.output_directory.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to reserve an existing run directory: {existing[0]}"
        )
    for run in runs:
        run.output_directory.mkdir(parents=True, exist_ok=False)
    reservation_token = uuid4().hex
    return tuple(
        replace(run, reservation_token=reservation_token)
        for run in runs
    )


def validate_phase72_run_directory(
    protocol: Mapping[str, Any],
    run: Phase72Run,
    *,
    repository_root: str | Path,
) -> None:
    """Verify a run plan still points to its unique protected-path-safe location."""
    validate_phase72_protocol(protocol)
    if not isinstance(run, Phase72Run) or not run.reservation_token:
        raise ValueError("Phase 7.2 training requires a preflight-reserved run plan")
    if run.configuration not in CONFIGURATIONS:
        raise ValueError("run plan has an unknown configuration")
    if run.training_seed not in protocol["seed_roles"]["training_seeds"]:
        raise ValueError("run plan seed is not in the protocol")
    specification = protocol["configurations"][run.configuration]
    horizon_settings = specification["horizon_settings"]
    expected_horizon_seed = (
        horizon_settings["sampler_seed_by_training_seed"][str(run.training_seed)]
        if horizon_settings is not None
        else None
    )
    if (
        run.observation_schema != specification["observation_schema"]
        or run.randomize_training_horizons
        is not specification["randomize_training_horizons"]
        or run.horizon_seed != expected_horizon_seed
    ):
        raise ValueError("run plan settings do not match the protocol")
    if len(run.output_directory.parents) < 2:
        raise ValueError("run plan output directory has an invalid structure")
    output_root = run.output_directory.parents[1].resolve()
    expected_directory = (
        output_root
        / f"config_{run.configuration}"
        / f"seed_{run.training_seed}"
    ).resolve()
    if run.output_directory.resolve() != expected_directory:
        raise ValueError("run plan output path does not match its configuration and seed")

    artifacts_root = (Path(repository_root).resolve() / ".artifacts").resolve()
    try:
        relative_root = output_root.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("run output root must be inside repository .artifacts") from error
    if not relative_root.parts:
        raise ValueError("run output root must be a dedicated .artifacts subdirectory")
    protected_model_dir = (artifacts_root / "ppo").resolve()
    if (
        output_root == protected_model_dir
        or protected_model_dir in output_root.parents
        or expected_directory == protected_model_dir
        or protected_model_dir in expected_directory.parents
    ):
        raise ValueError("Phase 7.2 may never use the existing .artifacts/ppo directory")


def build_phase72_run_metadata(
    protocol: Mapping[str, Any],
    *,
    configuration: str,
    training_seed: int,
    training_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the exact protocol record to pair with a run and model manifest."""
    validate_phase72_protocol(protocol)
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown experiment configuration: {configuration!r}")
    if training_seed not in protocol["seed_roles"]["training_seeds"]:
        raise ValueError("training_seed is not in the protocol training seed set")
    if not isinstance(training_provenance, Mapping):
        raise ValueError("training provenance must be a mapping")

    specification = protocol["configurations"][configuration]
    normalized_provenance = _normalize_json(training_provenance)
    if (
        normalized_provenance.get("source_type") != "synthetic"
        or normalized_provenance.get("synthetic") is not True
        or not str(normalized_provenance.get("scenario_id", "")).startswith("train-")
        or normalized_provenance.get("generator_version")
        != protocol["shared_settings"]["synthetic_generator"]["generator_version"]
        or _generation_config_only(
            normalized_provenance.get("generation_config")
        )
        != _generation_config_only(
            protocol["shared_settings"]["synthetic_generator"]
        )
    ):
        raise ValueError("training provenance does not match the protocol synthetic training split")
    if not isinstance(normalized_provenance.get("configuration_hash"), str):
        raise ValueError("training provenance must include a configuration hash")
    _require_seed(normalized_provenance.get("seed"), "training data seed")
    expected_training_data_seed = int(
        np.random.SeedSequence(training_seed, spawn_key=(0,)).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
    if (
        normalized_provenance["seed"] != expected_training_data_seed
        or normalized_provenance["scenario_id"]
        != f"train-{expected_training_data_seed}"
    ):
        raise ValueError("training provenance does not match the configured root seed")
    if any(field not in normalized_provenance for field in _PROVENANCE_FIELDS):
        raise ValueError("training provenance is missing manifest provenance fields")
    normalized_provenance = {
        "source_type": "synthetic",
        "synthetic": True,
        **{
            field: normalized_provenance[field]
            for field in _PROVENANCE_FIELDS
        },
    }

    ppo_settings = dict(protocol["shared_settings"]["ppo_settings"])
    ppo_settings.pop("seed_role")
    ppo_settings["seed"] = training_seed
    horizon_settings = specification["horizon_settings"]
    if horizon_settings is not None:
        horizon_settings = {
            "sampling": horizon_settings["sampling"],
            "minimum_decisions": horizon_settings["minimum_decisions"],
            "maximum_decisions": horizon_settings["maximum_decisions"],
            "sampler_seed": horizon_settings["sampler_seed_by_training_seed"][
                str(training_seed)
            ],
        }
    metadata = {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "configuration": configuration,
        "training_seed": training_seed,
        "training_scenario_root_seed": training_seed,
        "training_budget": dict(protocol["shared_settings"]["training_budget"]),
        "observation_schema": specification["observation_schema"],
        "observation_schema_version": protocol["shared_settings"][
            "observation_schema_version"
        ],
        "observation_shape": [
            observation_size_for_schema(specification["observation_schema"])
        ],
        "ppo_settings": ppo_settings,
        "synthetic_data_provenance": normalized_provenance,
        "horizon_settings": horizon_settings,
    }
    return metadata


def validate_phase72_run_metadata(
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    configuration: str,
    training_seed: int,
    expected_training_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Reject missing or inconsistent protocol/run metadata before training."""
    if not isinstance(metadata, Mapping):
        raise ValueError("run metadata must be a mapping")
    provenance = (
        expected_training_provenance
        if expected_training_provenance is not None
        else metadata.get("synthetic_data_provenance")
    )
    if not isinstance(provenance, Mapping):
        raise ValueError("run metadata is missing synthetic data provenance")
    expected = build_phase72_run_metadata(
        protocol,
        configuration=configuration,
        training_seed=training_seed,
        training_provenance=provenance,
    )
    if _normalize_json(metadata) != _normalize_json(expected):
        raise ValueError("run metadata is missing required fields or differs from the protocol")


def validate_experiment_manifest_metadata(
    metadata: Mapping[str, Any],
    *,
    seed: int,
    observation_schema: str,
    ppo_settings: Mapping[str, Any],
    manifest_provenance: Mapping[str, Any],
) -> None:
    """Check that the run record agrees with model-manifest fields."""
    if not isinstance(metadata, Mapping):
        raise ValueError("experiment manifest metadata must be a mapping")
    configuration = metadata.get("configuration")
    expected_schema = (
        OBSERVATION_SCHEMA_WITHOUT_TIME
        if configuration == "C"
        else DEFAULT_OBSERVATION_SCHEMA
    )
    if configuration not in CONFIGURATIONS or observation_schema != expected_schema:
        raise ValueError("experiment configuration and model observation schema disagree")
    if metadata.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("experiment manifest metadata has an unsupported protocol id")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("experiment manifest metadata has an unsupported protocol version")
    if metadata.get("training_seed") != seed:
        raise ValueError("experiment metadata training seed does not match model manifest")
    if seed not in TRAINING_SEEDS:
        raise ValueError("experiment metadata seed is outside the Phase 7.2 seed set")
    if metadata.get("training_scenario_root_seed") != seed:
        raise ValueError("experiment metadata scenario root seed does not match training seed")
    if metadata.get("observation_schema") != observation_schema:
        raise ValueError("experiment metadata schema does not match model manifest")
    if metadata.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("experiment metadata observation schema version is invalid")
    if metadata.get("observation_shape") != [
        observation_size_for_schema(observation_schema)
    ]:
        raise ValueError("experiment metadata observation shape is invalid")
    if _normalize_json(metadata.get("ppo_settings")) != _normalize_json(ppo_settings):
        raise ValueError("experiment PPO settings do not match model manifest")
    budget = metadata.get("training_budget")
    if not isinstance(budget, Mapping):
        raise ValueError("experiment metadata is missing its training budget")
    _require_keys(budget, {"requested_total_timesteps"}, "training_budget")
    _require_seed(budget.get("requested_total_timesteps"), "requested_total_timesteps")
    if budget["requested_total_timesteps"] != 10000:
        raise ValueError("experiment metadata training budget is inconsistent")

    provenance = metadata.get("synthetic_data_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("experiment metadata is missing synthetic data provenance")
    expected_provenance = {
        "source_type": "synthetic",
        "synthetic": True,
        **_normalize_json(manifest_provenance),
    }
    if _normalize_json(provenance) != _normalize_json(expected_provenance):
        raise ValueError("experiment data provenance does not match model manifest")
    if configuration == "B":
        horizon = metadata.get("horizon_settings")
        if not isinstance(horizon, Mapping):
            raise ValueError("Configuration B manifest is missing horizon settings")
        _require_keys(
            horizon,
            {"sampling", "minimum_decisions", "maximum_decisions", "sampler_seed"},
            "horizon_settings",
        )
        if (
            horizon.get("sampling") != "uniform_integer_decisions_inclusive"
            or horizon.get("minimum_decisions") != 11
            or horizon.get("maximum_decisions") != 33
            or horizon.get("sampler_seed") != seed
        ):
            raise ValueError("Configuration B horizon settings are inconsistent")
        _require_seed(horizon.get("sampler_seed"), "horizon sampler seed")
    elif metadata.get("horizon_settings") is not None:
        raise ValueError("only Configuration B may include horizon settings")


def _validate_configurations(
    configurations: Any,
    training_seeds: list[int],
    *,
    generator: SyntheticMarketConfig,
) -> None:
    if not isinstance(configurations, Mapping) or set(configurations) != set(CONFIGURATIONS):
        raise ValueError("protocol must define exactly configurations A, B, and C")
    expected = {
        "A": (DEFAULT_OBSERVATION_SCHEMA, False),
        "B": (DEFAULT_OBSERVATION_SCHEMA, True),
        "C": (OBSERVATION_SCHEMA_WITHOUT_TIME, False),
    }
    available_decisions = (
        generator.sessions - OBSERVATION_WARMUP_SESSIONS
    ) // DEFAULT_PERIOD_SESSIONS
    for name in CONFIGURATIONS:
        specification = configurations[name]
        _require_keys(
            specification,
            {"observation_schema", "randomize_training_horizons", "horizon_settings"},
            f"configuration {name}",
        )
        schema, randomized = expected[name]
        if (
            specification["observation_schema"] != schema
            or specification["randomize_training_horizons"] is not randomized
        ):
            raise ValueError(f"configuration {name} has an invalid schema or horizon mode")
        observation_size_for_schema(specification["observation_schema"])
        horizon = specification["horizon_settings"]
        if name != "B":
            if horizon is not None:
                raise ValueError(f"configuration {name} must not define horizon settings")
            continue
        _require_keys(
            horizon,
            {
                "sampling",
                "minimum_decisions",
                "maximum_decisions",
                "sampler_seed_by_training_seed",
            },
            "B horizon_settings",
        )
        minimum = horizon["minimum_decisions"]
        maximum = horizon["maximum_decisions"]
        if (
            horizon["sampling"] != "uniform_integer_decisions_inclusive"
            or minimum != 11
            or maximum != available_decisions
            or minimum > maximum
        ):
            raise ValueError("Configuration B horizon bounds do not match the protocol")
        sampler_seeds = horizon["sampler_seed_by_training_seed"]
        if not isinstance(sampler_seeds, Mapping) or set(sampler_seeds) != {
            str(seed) for seed in training_seeds
        }:
            raise ValueError("B must record one horizon sampler seed per training seed")
        for training_seed in training_seeds:
            if sampler_seeds[str(training_seed)] != training_seed:
                raise ValueError("B horizon sampler seeds must match training seeds")


def _validate_ppo_settings(settings: Any, seed: int) -> None:
    if not isinstance(settings, Mapping):
        raise ValueError("protocol PPO settings must be an object")
    values = dict(settings)
    if values.pop("seed_role", None) != "training_seed":
        raise ValueError("PPO seed role must be training_seed")
    values["seed"] = seed
    if "net_arch" in values:
        values["net_arch"] = tuple(values["net_arch"])
    try:
        config = PPOConfig(**values)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("protocol PPO settings are invalid") from error
    expected = asdict(PPOConfig(seed=seed, device="cpu"))
    if _normalize_json(asdict(config)) != _normalize_json(expected):
        raise ValueError("protocol PPO settings differ from the shared Phase 7.2 settings")


def _validate_training_budget(budget: Any) -> None:
    if not isinstance(budget, Mapping):
        raise ValueError("training_budget must be an object")
    _require_keys(budget, {"requested_total_timesteps"}, "training_budget")
    timesteps = budget["requested_total_timesteps"]
    _require_seed(timesteps, "requested_total_timesteps")
    if timesteps <= 0:
        raise ValueError("requested_total_timesteps must be positive")


def _validate_environment_settings(settings: Any) -> None:
    expected = {
        "warmup_sessions": OBSERVATION_WARMUP_SESSIONS,
        "period_sessions": DEFAULT_PERIOD_SESSIONS,
        "initial_capital": DEFAULT_INITIAL_CAPITAL,
        "max_asset_weight": DEFAULT_MAX_ASSET_WEIGHT,
        "min_cash_weight": DEFAULT_MIN_CASH_WEIGHT,
        "max_turnover": DEFAULT_MAX_TURNOVER,
        "transaction_cost_rate": DEFAULT_TRANSACTION_COST_RATE,
        "slippage_bps": DEFAULT_SLIPPAGE_BPS,
        "vix_level_scale": 100.0,
    }
    if _normalize_json(settings) != _normalize_json(expected):
        raise ValueError("protocol environment settings differ from current defaults")


def _validate_evaluation_settings(
    settings: Any,
    *,
    generator: SyntheticMarketConfig,
    environment: Mapping[str, Any],
    validation_count: int,
    test_count: int,
) -> None:
    if not isinstance(settings, Mapping):
        raise ValueError("evaluation settings must be an object")
    _require_keys(
        settings,
        {
            "deterministic",
            "evaluation_sessions",
            "rebalance_decisions",
            "validation_scenario_count",
            "reserved_test_scenario_count",
            "benchmark_inputs",
        },
        "evaluation",
    )
    evaluation_sessions = generator.sessions - environment["warmup_sessions"]
    period = environment["period_sessions"]
    if (
        settings["deterministic"] is not True
        or evaluation_sessions <= 0
        or evaluation_sessions % period != 0
        or settings["evaluation_sessions"] != evaluation_sessions
        or settings["rebalance_decisions"] != evaluation_sessions // period
        or settings["validation_scenario_count"] != validation_count
        or settings["reserved_test_scenario_count"] != test_count
        or settings["benchmark_inputs"]
        != "same synthetic scenario returns, capital, costs, and period as PPO"
    ):
        raise ValueError("protocol evaluation assumptions are inconsistent")


def _synthetic_config_from_json(settings: Any) -> SyntheticMarketConfig:
    if not isinstance(settings, Mapping):
        raise ValueError("synthetic_generator must be an object")
    values = dict(settings)
    values.pop("generator_version", None)
    try:
        values["correlation_matrix"] = tuple(
            tuple(row) for row in values["correlation_matrix"]
        )
        values["transition_matrix"] = tuple(
            tuple(row) for row in values["transition_matrix"]
        )
        values["initial_regime_probabilities"] = tuple(
            values["initial_regime_probabilities"]
        )
        values["regimes"] = tuple(
            RegimeConfig(
                name=regime["name"],
                annualized_log_returns=tuple(regime["annualized_log_returns"]),
                annualized_volatilities=tuple(regime["annualized_volatilities"]),
                vix_long_run_level=regime["vix_long_run_level"],
                vix_annualized_log_volatility=regime[
                    "vix_annualized_log_volatility"
                ],
            )
            for regime in values["regimes"]
        )
        config = SyntheticMarketConfig(**values)
        config.validate()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("protocol synthetic generator settings are invalid") from error
    return config


def _generation_config_only(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: _normalize_json(item)
        for key, item in value.items()
        if key != "generator_version"
    }


def _seed_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    seeds = [_require_seed(seed, label) for seed in value]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} must not contain duplicates")
    return seeds


def _require_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_json(item) for item in value]
    return value
