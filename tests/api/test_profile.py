"""Tests for profile API route functions."""

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.routers.profile import evaluate_profile
from src.api.schemas import ProfileEvaluationRequest
from src.models.profile_classifier import CategoryPrediction
from src.rag.profile_evaluator import ProfileEvaluationResponse as DomainProfileEvaluationResponse
from src.rag.retriever import RetrievedDocument


def sample_document() -> RetrievedDocument:
    """Return one retrieved document for profile API tests."""
    return RetrievedDocument(
        job_id=123,
        title="Data Engineer",
        job_category="Data Engineering",
        description="Python, SQL and Airflow role.",
        description_preview="Python and SQL role.",
        score=0.82,
        payload={"job_id": 123},
    )


class FakeProfileEvaluator:
    """Fake profile evaluator for API contract tests."""

    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None

    def evaluate(
        self,
        profile_text: str,
        top_k: int = 5,
        include_category_prediction: bool = True,
    ) -> DomainProfileEvaluationResponse:
        """Return a deterministic profile evaluation."""
        self.last_call = {
            "profile_text": profile_text,
            "top_k": top_k,
            "include_category_prediction": include_category_prediction,
        }
        return DomainProfileEvaluationResponse(
            evaluation="Refuerza Airflow y modelado de datos.",
            profile_text=profile_text.strip(),
            target_role=None,
            category_prediction=CategoryPrediction(
                label="Data Engineering",
                confidence=0.74,
                model_type="random_forest",
            ),
            documents=[sample_document()],
            prompt="prompt",
        )


class FailingProfileEvaluator:
    """Fake profile evaluator that simulates unavailable dependencies."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def evaluate(
        self,
        profile_text: str,
        top_k: int = 5,
        include_category_prediction: bool = True,
    ) -> DomainProfileEvaluationResponse:
        """Raise the configured error."""
        raise self.error


def test_profile_evaluate_returns_structured_response() -> None:
    """The profile endpoint should expose classifier signal and RAG sources."""
    evaluator = FakeProfileEvaluator()
    request = ProfileEvaluationRequest(
        profile_text="Python, SQL, pandas and Airflow.",
        top_k=3,
    )

    response = asyncio.run(evaluate_profile(request, evaluator))

    assert evaluator.last_call == {
        "profile_text": "Python, SQL, pandas and Airflow.",
        "top_k": 3,
        "include_category_prediction": True,
    }
    assert response.model_dump() == {
        "profile_text": "Python, SQL, pandas and Airflow.",
        "category_prediction": {
            "label": "Data Engineering",
            "confidence": 0.74,
            "certainty": "alta",
            "model_type": "random_forest",
        },
        "evaluation": "Refuerza Airflow y modelado de datos.",
        "sources": [
            {
                "job_id": 123,
                "title": "Data Engineer",
                "job_category": "Data Engineering",
                "score": 0.82,
            }
        ],
    }


@pytest.mark.parametrize("profile_text", ["", "   "])
def test_profile_evaluation_request_rejects_empty_profile(profile_text: str) -> None:
    """The API request schema should reject blank profile text."""
    with pytest.raises(ValidationError):
        ProfileEvaluationRequest(profile_text=profile_text)


def test_profile_evaluate_maps_validation_errors_to_400() -> None:
    """Domain validation errors should become explicit client errors."""
    request = ProfileEvaluationRequest(profile_text="Python and SQL.")
    evaluator = FailingProfileEvaluator(ValueError("top_k must be greater than 0."))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(evaluate_profile(request, evaluator))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "top_k must be greater than 0."


def test_profile_evaluate_maps_runtime_errors_to_503() -> None:
    """External dependency failures should become service unavailable errors."""
    request = ProfileEvaluationRequest(profile_text="Python and SQL.")
    evaluator = FailingProfileEvaluator(
        RuntimeError("Could not generate a response with Ollama.")
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(evaluate_profile(request, evaluator))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Could not generate a response with Ollama."


def test_profile_evaluate_maps_unexpected_dependency_errors_to_503() -> None:
    """Unexpected Qdrant/Ollama failures should avoid generic 500 responses."""
    request = ProfileEvaluationRequest(profile_text="Python and SQL.")
    evaluator = FailingProfileEvaluator(ConnectionError("Qdrant is unavailable."))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(evaluate_profile(request, evaluator))

    assert exc_info.value.status_code == 503
    assert "Qdrant and Ollama" in str(exc_info.value.detail)
