"""Reproducible synthetic ETF, benchmark, and volatility-index scenarios."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from src.rl.environment.portfolio_env import ASSET_UNIVERSE, BENCHMARK_SYMBOL

GENERATOR_VERSION = "1.0"
TRADING_DAYS_PER_YEAR = 252
REGIME_NAMES = ("bull", "bear", "sideways")
GENERATED_SERIES = (*ASSET_UNIVERSE, BENCHMARK_SYMBOL)
_LOG_RETURN_LIMIT = 20.0
_CORRELATION_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RegimeConfig:
    """Annualized synthetic log-return and VIX parameters for one regime."""

    name: Literal["bull", "bear", "sideways"]
    annualized_log_returns: tuple[float, ...]
    annualized_volatilities: tuple[float, ...]
    vix_long_run_level: float
    vix_annualized_log_volatility: float


def _default_regimes() -> tuple[RegimeConfig, ...]:
    return (
        RegimeConfig(
            name="bull",
            annualized_log_returns=(0.12, 0.16, 0.025, 0.04, 0.07, 0.08, 0.10),
            annualized_volatilities=(0.18, 0.27, 0.07, 0.18, 0.16, 0.15, 0.17),
            vix_long_run_level=16.0,
            vix_annualized_log_volatility=0.35,
        ),
        RegimeConfig(
            name="bear",
            annualized_log_returns=(-0.18, -0.25, 0.04, -0.08, -0.10, -0.08, -0.20),
            annualized_volatilities=(0.35, 0.48, 0.12, 0.32, 0.28, 0.25, 0.32),
            vix_long_run_level=38.0,
            vix_annualized_log_volatility=0.55,
        ),
        RegimeConfig(
            name="sideways",
            annualized_log_returns=(0.04, 0.05, 0.025, 0.02, 0.04, 0.05, 0.04),
            annualized_volatilities=(0.16, 0.23, 0.06, 0.15, 0.14, 0.13, 0.15),
            vix_long_run_level=21.0,
            vix_annualized_log_volatility=0.30,
        ),
    )


def _default_correlation_matrix() -> tuple[tuple[float, ...], ...]:
    matrix = np.full((len(GENERATED_SERIES), len(GENERATED_SERIES)), 0.25)
    np.fill_diagonal(matrix, 1.0)
    return tuple(tuple(float(value) for value in row) for row in matrix)


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Configuration for a seeded, regime-switching synthetic scenario.

    Return drifts and volatilities are annualized log-return parameters for the
    six assets followed by the synthetic benchmark. The correlation matrix
    follows that same seven-series order and is shared across regimes.
    """

    sessions: int = 756
    correlation_matrix: tuple[tuple[float, ...], ...] = field(
        default_factory=_default_correlation_matrix
    )
    regimes: tuple[RegimeConfig, ...] = field(default_factory=_default_regimes)
    transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.94, 0.02, 0.04),
        (0.04, 0.92, 0.04),
        (0.08, 0.08, 0.84),
    )
    initial_regime_probabilities: tuple[float, ...] = (0.50, 0.20, 0.30)
    initial_vix_level: float = 20.0
    vix_mean_reversion_speed: float = 0.08
    vix_min_level: float = 5.0
    vix_max_level: float = 100.0
    vix_log_level_clip: float = 5.0

    def validate(self) -> None:
        """Validate all dimensions, parameter ranges, and probability matrices."""
        if (
            isinstance(self.sessions, bool)
            or not isinstance(self.sessions, int)
            or self.sessions <= 0
        ):
            raise ValueError("sessions must be a positive integer")
        _require_finite_positive("initial_vix_level", self.initial_vix_level)
        _require_finite_positive(
            "vix_mean_reversion_speed", self.vix_mean_reversion_speed
        )
        _require_finite_positive("vix_min_level", self.vix_min_level)
        _require_finite_positive("vix_max_level", self.vix_max_level)
        _require_finite_positive("vix_log_level_clip", self.vix_log_level_clip)
        if self.vix_min_level >= self.vix_max_level:
            raise ValueError("vix_min_level must be less than vix_max_level")
        if not self.vix_min_level <= self.initial_vix_level <= self.vix_max_level:
            raise ValueError(
                "initial_vix_level must be between the configured VIX bounds"
            )
        correlation = _finite_matrix(
            "correlation_matrix",
            self.correlation_matrix,
            (len(GENERATED_SERIES), len(GENERATED_SERIES)),
        )
        if not np.allclose(correlation, correlation.T, atol=1e-12, rtol=0.0):
            raise ValueError("correlation_matrix must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("correlation_matrix must have a unit diagonal")
        if np.any(correlation < -1.0) or np.any(correlation > 1.0):
            raise ValueError("correlation_matrix entries must be between -1 and 1")
        if float(np.linalg.eigvalsh(correlation).min()) < -_CORRELATION_TOLERANCE:
            raise ValueError("correlation_matrix must be positive semidefinite")

        if tuple(regime.name for regime in self.regimes) != REGIME_NAMES:
            raise ValueError(f"regimes must be ordered as {REGIME_NAMES}")
        for regime in self.regimes:
            _validate_regime(regime)
            if not self.vix_min_level <= regime.vix_long_run_level <= self.vix_max_level:
                raise ValueError(
                    f"{regime.name} VIX target must be within configured VIX bounds"
                )

        transitions = _finite_matrix(
            "transition_matrix",
            self.transition_matrix,
            (len(REGIME_NAMES), len(REGIME_NAMES)),
        )
        _validate_probability_matrix(transitions, "transition_matrix")
        initial = _finite_vector(
            "initial_regime_probabilities",
            self.initial_regime_probabilities,
            len(REGIME_NAMES),
        )
        _validate_probability_vector(initial, "initial_regime_probabilities")


@dataclass(frozen=True)
class SyntheticDataset:
    """Aligned synthetic data accepted directly by ``SyntheticPortfolioEnv``."""

    asset_returns: np.ndarray
    benchmark_returns: np.ndarray
    vix_levels: np.ndarray
    asset_names: tuple[str, ...]
    series_names: tuple[str, ...]
    regime_labels: tuple[str, ...]
    metadata: dict[str, object]
    config: SyntheticMarketConfig
    seed: int

    def __post_init__(self) -> None:
        self.config.validate()
        _validate_seed(self.seed)
        if self.asset_names != ASSET_UNIVERSE:
            raise ValueError("asset_names must match the environment's fixed ordering")
        if self.series_names != GENERATED_SERIES:
            raise ValueError("series_names must match the fixed asset/benchmark ordering")
        if len(self.regime_labels) != self.config.sessions or any(
            label not in REGIME_NAMES for label in self.regime_labels
        ):
            raise ValueError("regime_labels must align with sessions and known regimes")
        sessions = len(self.regime_labels)
        if self.asset_returns.shape != (sessions, len(self.asset_names)):
            raise ValueError("asset_returns shape does not match labels and asset_names")
        if self.benchmark_returns.shape != (sessions,):
            raise ValueError("benchmark_returns are not aligned with regime labels")
        if self.vix_levels.shape != (sessions,):
            raise ValueError("vix_levels are not aligned with regime labels")
        if (
            self.metadata.get("source_type") != "synthetic"
            or self.metadata.get("synthetic") is not True
        ):
            raise ValueError("dataset metadata must identify synthetic data")
        if self.metadata.get("seed") != self.seed:
            raise ValueError("metadata seed must match dataset seed")
        if self.metadata.get("generation_config") != asdict(self.config):
            raise ValueError("metadata configuration must match dataset config")
        if not np.isfinite(self.asset_returns).all():
            raise ValueError("asset_returns must be finite")
        if not np.isfinite(self.benchmark_returns).all():
            raise ValueError("benchmark_returns must be finite")
        if np.any(self.asset_returns <= -1.0) or np.any(self.benchmark_returns <= -1.0):
            raise ValueError("simple returns must be greater than -1")
        if not np.isfinite(self.vix_levels).all() or np.any(self.vix_levels <= 0.0):
            raise ValueError("vix_levels must be finite and positive")

    def environment_kwargs(self) -> dict[str, object]:
        """Return the matching input arguments for ``SyntheticPortfolioEnv``."""
        return {
            "asset_returns": self.asset_returns,
            "benchmark_returns": self.benchmark_returns,
            "vix_levels": self.vix_levels,
            "data_metadata": self.metadata,
        }


@dataclass(frozen=True)
class SyntheticScenarioSplits:
    """Independent synthetic train, validation, and test scenarios."""

    train: SyntheticDataset
    validation: SyntheticDataset
    test: SyntheticDataset


def generate_synthetic_market(
    config: SyntheticMarketConfig | None = None,
    *,
    seed: int,
    scenario_id: str = "scenario",
) -> SyntheticDataset:
    """Generate aligned synthetic simple returns and a positive VIX level path.

    A Markov chain selects a bull, bear, or sideways regime each session. Given
    the regime, the seven log returns (six ETFs plus benchmark) are sampled
    jointly from a multivariate normal distribution using annualized drift,
    volatility, and correlation parameters.
    """
    selected_config = config or SyntheticMarketConfig()
    selected_config.validate()
    normalized_seed = _validate_seed(seed)
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")

    rng = np.random.default_rng(normalized_seed)
    regimes = selected_config.regimes
    initial_probabilities = np.asarray(
        selected_config.initial_regime_probabilities, dtype=np.float64
    )
    transition_matrix = np.asarray(selected_config.transition_matrix, dtype=np.float64)
    correlation = np.asarray(selected_config.correlation_matrix, dtype=np.float64)
    regime_ids = np.empty(selected_config.sessions, dtype=np.int8)
    regime_ids[0] = rng.choice(len(REGIME_NAMES), p=initial_probabilities)
    for session in range(1, selected_config.sessions):
        regime_ids[session] = rng.choice(
            len(REGIME_NAMES), p=transition_matrix[regime_ids[session - 1]]
        )

    log_returns = np.empty(
        (selected_config.sessions, len(GENERATED_SERIES)), dtype=np.float64
    )
    vix_levels = np.empty(selected_config.sessions, dtype=np.float64)
    previous_log_vix = math.log(selected_config.initial_vix_level)
    correlation_eigenvalues, correlation_eigenvectors = np.linalg.eigh(correlation)
    correlation_root = correlation_eigenvectors @ np.diag(
        np.sqrt(np.maximum(correlation_eigenvalues, 0.0))
    )

    for session, regime_id in enumerate(regime_ids):
        regime = regimes[int(regime_id)]
        annual_volatilities = np.asarray(
            regime.annualized_volatilities, dtype=np.float64
        )
        daily_scale = annual_volatilities / math.sqrt(TRADING_DAYS_PER_YEAR)
        covariance_root = daily_scale[:, None] * correlation_root
        standard_normal = rng.standard_normal(len(GENERATED_SERIES))
        sampled_log_return = (
            np.asarray(regime.annualized_log_returns, dtype=np.float64)
            / TRADING_DAYS_PER_YEAR
            + covariance_root @ standard_normal
        )
        log_returns[session] = np.clip(
            sampled_log_return, -_LOG_RETURN_LIMIT, _LOG_RETURN_LIMIT
        )

        vix_target = math.log(regime.vix_long_run_level)
        vix_noise = (
            regime.vix_annualized_log_volatility
            / math.sqrt(TRADING_DAYS_PER_YEAR)
            * rng.standard_normal()
        )
        vix_noise = float(
            np.clip(
                vix_noise,
                -selected_config.vix_log_level_clip,
                selected_config.vix_log_level_clip,
            )
        )
        previous_log_vix += (
            selected_config.vix_mean_reversion_speed
            * (vix_target - previous_log_vix)
            + vix_noise
        )
        previous_log_vix = float(
            np.clip(
                previous_log_vix,
                math.log(selected_config.vix_min_level),
                math.log(selected_config.vix_max_level),
            )
        )
        vix_levels[session] = math.exp(previous_log_vix)

    simple_returns = np.expm1(log_returns)
    if not np.isfinite(simple_returns).all() or np.any(simple_returns <= -1.0):
        raise FloatingPointError("generated log returns did not produce valid simple returns")

    asset_returns = simple_returns[:, : len(ASSET_UNIVERSE)].copy()
    benchmark_returns = simple_returns[:, -1].copy()
    _mark_read_only(asset_returns, benchmark_returns, vix_levels)
    config_payload = _config_to_json(selected_config)
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata: dict[str, object] = {
        "source_type": "synthetic",
        "synthetic": True,
        "generator_version": GENERATOR_VERSION,
        "scenario_id": scenario_id,
        "seed": normalized_seed,
        "configuration_hash": config_hash,
        "generation_config": config_payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_names": list(GENERATED_SERIES),
        "asset_names": list(ASSET_UNIVERSE),
        "regime_names": list(REGIME_NAMES),
        "return_method": "regime-switching multivariate normal log returns",
        "vix_method": "regime-targeted mean-reverting log-level process",
    }
    return SyntheticDataset(
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        vix_levels=vix_levels,
        asset_names=ASSET_UNIVERSE,
        series_names=GENERATED_SERIES,
        regime_labels=tuple(REGIME_NAMES[int(value)] for value in regime_ids),
        metadata=metadata,
        config=selected_config,
        seed=normalized_seed,
    )


def generate_synthetic_splits(
    config: SyntheticMarketConfig | None = None,
    *,
    seed: int,
    sessions: int | None = None,
) -> SyntheticScenarioSplits:
    """Create independent deterministic train/validation/test scenarios.

    Each split gets a seed derived from the root seed and a stable split index.
    These are independent simulated paths, not chronological partitions of a
    shared path and not historical time periods.
    """
    selected_config = config or SyntheticMarketConfig()
    selected_config.validate()
    normalized_seed = _validate_seed(seed)
    if sessions is not None:
        selected_config = replace(selected_config, sessions=sessions)
        selected_config.validate()

    split_datasets: dict[str, SyntheticDataset] = {}
    for split_index, split_name in enumerate(("train", "validation", "test")):
        split_seed_sequence = np.random.SeedSequence(
            normalized_seed, spawn_key=(split_index,)
        )
        split_seed = int(split_seed_sequence.generate_state(1, dtype=np.uint64)[0])
        split_datasets[split_name] = generate_synthetic_market(
            selected_config,
            seed=split_seed,
            scenario_id=f"{split_name}-{split_seed}",
        )
    return SyntheticScenarioSplits(
        train=split_datasets["train"],
        validation=split_datasets["validation"],
        test=split_datasets["test"],
    )


def _validate_regime(regime: RegimeConfig) -> None:
    if regime.name not in REGIME_NAMES:
        raise ValueError(f"unsupported regime name: {regime.name}")
    _finite_vector(
        f"{regime.name}.annualized_log_returns",
        regime.annualized_log_returns,
        len(GENERATED_SERIES),
    )
    volatilities = _finite_vector(
        f"{regime.name}.annualized_volatilities",
        regime.annualized_volatilities,
        len(GENERATED_SERIES),
    )
    if np.any(volatilities < 0.0):
        raise ValueError(f"{regime.name} volatilities must be non-negative")
    _require_finite_positive(
        f"{regime.name}.vix_long_run_level", regime.vix_long_run_level
    )
    _require_finite_nonnegative(
        f"{regime.name}.vix_annualized_log_volatility",
        regime.vix_annualized_log_volatility,
    )


def _finite_matrix(
    name: str,
    values: tuple[tuple[float, ...], ...],
    shape: tuple[int, int],
) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return matrix


def _finite_vector(name: str, values: tuple[float, ...], size: int) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must have shape ({size},) and finite values")
    return vector


def _validate_probability_matrix(matrix: np.ndarray, name: str) -> None:
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(f"{name} entries must be between 0 and 1")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} rows must sum to 1")


def _validate_probability_vector(vector: np.ndarray, name: str) -> None:
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"{name} entries must be between 0 and 1")
    if not math.isclose(float(vector.sum()), 1.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to 1")


def _require_finite_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _require_finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer between 0 and 2**64 - 1")
    if seed < 0 or seed > np.iinfo(np.uint64).max:
        raise ValueError("seed must be between 0 and 2**64 - 1")
    return int(seed)


def _config_to_json(config: SyntheticMarketConfig) -> dict[str, object]:
    return asdict(config)


def _mark_read_only(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.setflags(write=False)
