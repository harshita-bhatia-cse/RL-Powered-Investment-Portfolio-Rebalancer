"""Pydantic response contracts for stored synthetic Phase 7.2 research results."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


SyntheticProvenance = Literal["stored synthetic Phase 7.2 experiment results"]
ConfigurationId = Literal["A", "B", "C"]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
StrictInteger = Annotated[int, Field(strict=True)]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

_PROVENANCE: SyntheticProvenance = "stored synthetic Phase 7.2 experiment results"
_PROVENANCE_DESCRIPTION = (
    "Stored synthetic Phase 7.2 experiment results only; not NSE/BSE performance, "
    "live trading results, or investment recommendations."
)


class ResearchResponseModel(BaseModel):
    """Shared strict response configuration and synthetic-only provenance."""

    model_config = ConfigDict(extra="forbid")

    synthetic_only: Literal[True] = Field(
        default=True,
        description="These results are synthetic experiment outputs only.",
    )
    provenance: SyntheticProvenance = Field(
        default=_PROVENANCE,
        description=_PROVENANCE_DESCRIPTION,
    )


class ResearchStatusResponse(ResearchResponseModel):
    """The fields returned by Phase72ResearchStore.read_status()."""

    format: Literal["phase72-run-manifest-v2"]
    manifest_status: Literal["test_complete"]
    protocol_id: Literal["phase7_2_synthetic_ppo_comparison"]
    protocol_version: StrictInteger
    protocol_sha256: Sha256Digest
    run_count: StrictInteger
    trained_run_count: StrictInteger
    validation_result_count: StrictInteger
    final_test_result_count: StrictInteger
    selected_configuration: Literal["A"]
    final_test_scenario_count: StrictInteger
    integrity_verified: Literal[True]


class ValidationSeedAggregateResponse(BaseModel):
    """One seed's stored validation aggregate for a configuration."""

    model_config = ConfigDict(extra="forbid")

    training_seed: StrictInteger
    mean_validation_sharpe: FiniteFloat
    mean_maximum_drawdown: FiniteFloat
    mean_maximum_drawdown_magnitude: FiniteFloat
    mean_turnover: FiniteFloat


class ValidationConfigurationAggregateResponse(BaseModel):
    """Stored configuration-level validation metrics and seed aggregates."""

    model_config = ConfigDict(extra="forbid")

    mean_validation_sharpe: FiniteFloat
    mean_maximum_drawdown: FiniteFloat
    mean_maximum_drawdown_magnitude: FiniteFloat
    mean_turnover: FiniteFloat
    seed_aggregates: list[ValidationSeedAggregateResponse]


class SelectionRuleResponse(BaseModel):
    """Frozen selection rule stored in selection.json."""

    model_config = ConfigDict(extra="forbid")

    primary: Literal["highest mean validation Sharpe ratio"]
    aggregation: Literal[
        "mean over 20 roots per seed, then mean over all five seeds"
    ]
    tie_breakers: list[
        Literal[
            "lowest mean maximum drawdown magnitude",
            "lowest mean portfolio turnover",
            "lexicographically smallest configuration id",
        ]
    ]
    drawdown_note: Literal[
        "The evaluation metric is non-positive; minimize its absolute "
        "magnitude to prefer a smaller drawdown."
    ]


class PrimaryTieBreakerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_value: FiniteFloat
    candidates_within_tolerance: list[ConfigurationId]


class DrawdownTieBreakerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_magnitude: FiniteFloat
    candidates_within_tolerance: list[ConfigurationId]


class TurnoverTieBreakerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_value: FiniteFloat
    candidates_within_tolerance: list[ConfigurationId]


class ConfigurationIdTieBreakerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected: ConfigurationId
    remaining_candidates: list[ConfigurationId]


class TieBreakerOutcomesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_metric: PrimaryTieBreakerResponse
    drawdown_tie_breaker: DrawdownTieBreakerResponse
    turnover_tie_breaker: TurnoverTieBreakerResponse
    configuration_id_tie_breaker: ConfigurationIdTieBreakerResponse


class ResearchSelectionResponse(ResearchResponseModel):
    """The fields returned by Phase72ResearchStore.read_selection()."""

    format: Literal["phase72-selection-v1"]
    protocol_id: Literal["phase7_2_synthetic_ppo_comparison"]
    protocol_version: StrictInteger
    protocol_sha256: Sha256Digest
    selected_configuration: Literal["A"]
    selection_sha256: Sha256Digest
    selection_rule: SelectionRuleResponse
    selection_tolerance: FiniteFloat
    training_seeds_retained: list[StrictInteger]
    configuration_metrics: dict[
        ConfigurationId,
        ValidationConfigurationAggregateResponse,
    ]
    tie_breaker_outcomes: TieBreakerOutcomesResponse


class ValidationAggregatesResponse(ResearchResponseModel):
    """The fields returned by Phase72ResearchStore.read_validation_aggregates()."""

    configuration_metrics: dict[
        ConfigurationId,
        ValidationConfigurationAggregateResponse,
    ]


class FinalTestAggregateResponse(BaseModel):
    """Stored aggregate for the final-test result set."""

    model_config = ConfigDict(extra="forbid")

    mean_maximum_drawdown: FiniteFloat
    mean_sharpe_ratio: FiniteFloat
    mean_turnover: FiniteFloat
    scenario_count: StrictInteger


class FinalTestOverallResponse(ResearchResponseModel):
    """The fields returned by Phase72ResearchStore.read_final_test_overall()."""

    selected_configuration: Literal["A"]
    scenario_count: StrictInteger
    seed_count: StrictInteger
    overall_aggregate: FinalTestAggregateResponse


class FinalTestSeedAggregateItem(BaseModel):
    """One stored seed aggregate returned by the research store."""

    model_config = ConfigDict(extra="forbid")

    training_seed: StrictInteger
    scenario_count: StrictInteger
    mean_maximum_drawdown: FiniteFloat
    mean_sharpe_ratio: FiniteFloat
    mean_turnover: FiniteFloat


class FinalTestSeedAggregatesResponse(ResearchResponseModel):
    """Provenance envelope for the list returned by read_final_test_seed_aggregates()."""

    seed_aggregates: list[FinalTestSeedAggregateItem]


class ArtifactIntegritySummaryResponse(ResearchStatusResponse):
    """Path-free integrity summary returned by Phase72ResearchStore.verify()."""
