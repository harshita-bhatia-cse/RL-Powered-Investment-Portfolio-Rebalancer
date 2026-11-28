"""Authenticated read-only API for frozen synthetic Phase 7.2 research results."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ValidationError

from src.api.phase8_research_schemas import (
    ArtifactIntegritySummaryResponse,
    FinalTestOverallResponse,
    FinalTestSeedAggregatesResponse,
    ResearchSelectionResponse,
    ResearchStatusResponse,
    ValidationAggregatesResponse,
)
from src.auth.dependencies import get_current_user
from src.services.phase7_2_research_store import (
    Phase72ResearchStore,
    Phase72ResearchStoreError,
)


logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
router = APIRouter(
    prefix="/api/research/phase7-2",
    tags=["synthetic research"],
)


def _verified_read(
    response_model: type[ResponseModel],
    read: Callable[[Phase72ResearchStore], Any] | None = None,
) -> ResponseModel:
    """Verify a fresh store before reading, returning only a generic failure."""
    try:
        store = Phase72ResearchStore()
        verified_summary = store.verify()
        result = verified_summary if read is None else read(store)
        return response_model.model_validate(result)
    except (
        Phase72ResearchStoreError,
        OSError,
        UnicodeError,
        ValidationError,
    ) as error:
        logger.error(
            "Phase 7.2 research artifact read failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Stored synthetic Phase 7.2 research artifacts are unavailable "
                "or could not be verified."
            ),
        ) from error


@router.get("/status", response_model=ResearchStatusResponse)
def research_status(
    _user: object = Depends(get_current_user),
) -> ResearchStatusResponse:
    """Return verified status for stored synthetic Phase 7.2 results."""
    return _verified_read(
        ResearchStatusResponse,
        lambda store: store.read_status(),
    )


@router.get("/selection", response_model=ResearchSelectionResponse)
def research_selection(
    _user: object = Depends(get_current_user),
) -> ResearchSelectionResponse:
    """Return the frozen synthetic validation selection."""
    return _verified_read(
        ResearchSelectionResponse,
        lambda store: store.read_selection(),
    )


@router.get("/validation", response_model=ValidationAggregatesResponse)
def research_validation_aggregates(
    _user: object = Depends(get_current_user),
) -> ValidationAggregatesResponse:
    """Return stored synthetic validation aggregates."""
    return _verified_read(
        ValidationAggregatesResponse,
        lambda store: store.read_validation_aggregates(),
    )


@router.get("/final-test", response_model=FinalTestOverallResponse)
def research_final_test_overall(
    _user: object = Depends(get_current_user),
) -> FinalTestOverallResponse:
    """Return the stored synthetic final-test summary."""
    return _verified_read(
        FinalTestOverallResponse,
        lambda store: store.read_final_test_overall(),
    )


@router.get(
    "/final-test/seeds",
    response_model=FinalTestSeedAggregatesResponse,
)
def research_final_test_seed_aggregates(
    _user: object = Depends(get_current_user),
) -> FinalTestSeedAggregatesResponse:
    """Return stored per-seed synthetic final-test aggregates."""
    return _verified_read(
        FinalTestSeedAggregatesResponse,
        lambda store: {
            "seed_aggregates": store.read_final_test_seed_aggregates()
        },
    )


@router.get("/integrity", response_model=ArtifactIntegritySummaryResponse)
def research_integrity(
    _user: object = Depends(get_current_user),
) -> ArtifactIntegritySummaryResponse:
    """Return a path-free summary from fresh artifact verification."""
    return _verified_read(ArtifactIntegritySummaryResponse)
