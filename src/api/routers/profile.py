"""Profile evaluation API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    CategoryPredictionResponse,
    ProfileEvaluationApiResponse,
    ProfileEvaluationRequest,
    ProfileSourceItem,
)
from src.models.profile_classifier import JobCategoryClassifier
from src.rag.profile_evaluator import (
    ProfileEvaluationResponse as DomainProfileEvaluationResponse,
)
from src.rag.profile_evaluator import (
    ProfileEvaluator,
)
from src.rag.prompts import estimate_classifier_certainty

router = APIRouter(
    prefix="/profile",
    tags=["profile"],
)


async def get_profile_evaluator() -> ProfileEvaluator:
    """Provide the profile evaluator used by profile endpoints."""
    return ProfileEvaluator(category_classifier=JobCategoryClassifier())


ProfileEvaluatorDep = Annotated[ProfileEvaluator, Depends(get_profile_evaluator)]


@router.post("/evaluate", response_model=ProfileEvaluationApiResponse)
async def evaluate_profile(
    request: ProfileEvaluationRequest,
    evaluator: ProfileEvaluatorDep,
) -> ProfileEvaluationApiResponse:
    """Evaluate a profile using the classifier signal plus RAG market evidence."""
    try:
        response = evaluator.evaluate(
            profile_text=request.profile_text,
            top_k=request.top_k,
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
            )
            for document in response.documents
        ],
    )
