"""Profile evaluation API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from job_market_intelligence.api.schemas import (
    CategoryPredictionResponse,
    ProfileEvaluationApiResponse,
    ProfileEvaluationRequest,
    ProfileSourceItem,
)
from job_market_intelligence.api.services import build_profile_evaluator
from job_market_intelligence.config.settings import Settings
from job_market_intelligence.rag.profile_evaluator import (
    ProfileEvaluationResponse as DomainProfileEvaluationResponse,
)
from job_market_intelligence.rag.profile_evaluator import (
    ProfileEvaluator,
)
from job_market_intelligence.rag.prompts import estimate_classifier_certainty

router = APIRouter(
    prefix="/profile",
    tags=["profile"],
)


async def get_profile_evaluator(request: Request) -> ProfileEvaluator:
    """Provide a shared profile evaluator, creating it on first use."""
    evaluator = getattr(request.app.state, "profile_evaluator", None)
    if evaluator is not None:
        return evaluator

    settings = await get_settings(request)
    try:
        evaluator = build_profile_evaluator(settings)
    except Exception as error:
        request.app.state.profile_evaluator = None
        request.app.state.profile_evaluator_error = error
        detail = (
            "Profile evaluator is not available. "
            "Check that Qdrant and Ollama are running and that local model "
            "artifacts are available."
        )
        detail = f"{detail} Startup error: {error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from error

    request.app.state.profile_evaluator = evaluator
    request.app.state.profile_evaluator_error = None
    return evaluator


ProfileEvaluatorDep = Annotated[ProfileEvaluator, Depends(get_profile_evaluator)]


async def get_settings(request: Request) -> Settings:
    """Provide API settings loaded during app startup."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API settings are not available.",
        )
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/evaluate", response_model=ProfileEvaluationApiResponse)
async def evaluate_profile(
    request: ProfileEvaluationRequest,
    evaluator: ProfileEvaluatorDep,
    settings: SettingsDep,
) -> ProfileEvaluationApiResponse:
    """Evaluate a profile using the classifier signal plus RAG market evidence."""
    try:
        response = evaluator.evaluate(
            profile_text=request.profile_text,
            top_k=request.top_k or settings.profile_top_k,
            include_category_prediction=True,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Profile evaluation dependencies are not available. "
                "Check that Qdrant and Ollama are running and that local model "
                "artifacts are available."
            ),
        ) from error

    if response.category_prediction is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile evaluator did not return a category prediction.",
        )

    return _to_api_response(response)


def _to_api_response(
    response: DomainProfileEvaluationResponse,
) -> ProfileEvaluationApiResponse:
    """Convert the domain profile evaluation response to the API schema."""
    category_prediction = response.category_prediction
    if category_prediction is None:
        raise ValueError("Category prediction is required for API responses.")

    return ProfileEvaluationApiResponse(
        profile_text=response.profile_text,
        category_prediction=CategoryPredictionResponse(
            label=category_prediction.label,
            confidence=category_prediction.confidence,
            certainty=estimate_classifier_certainty(category_prediction.confidence),
            model_type=category_prediction.model_type,
        ),
        evaluation=response.evaluation,
        sources=[
            ProfileSourceItem(
                job_id=document.job_id,
                title=document.title,
                job_category=document.job_category,
                score=document.score,
                description=document.description,
            )
            for document in response.documents
        ],
    )
