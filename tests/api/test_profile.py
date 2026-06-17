"""Tests for profile API endpoints."""

import pytest
from fastapi.testclient import TestClient

from job_market_intelligence.api.main import app
from job_market_intelligence.api.routers import profile as profile_router
from job_market_intelligence.api.routers.profile import (
    get_profile_evaluator,
    get_settings,
)
from job_market_intelligence.config.settings import Settings
from job_market_intelligence.models.profile_classifier import CategoryPrediction
from job_market_intelligence.rag.profile_evaluator import (
    ProfileEvaluationResponse as DomainProfileEvaluationResponse,
)
from job_market_intelligence.rag.retriever import RetrievedDocument


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
        _ = (profile_text, top_k, include_category_prediction)
        raise self.error


def override_profile_evaluator(evaluator: FakeProfileEvaluator) -> None:
    """Register the fake profile evaluator dependency."""
    app.dependency_overrides[get_profile_evaluator] = lambda: evaluator


def test_profile_evaluate_returns_response_with_test_client(
    api_client: TestClient,
) -> None:
    """Profile evaluation should expose classifier signal and RAG sources."""
    evaluator = FakeProfileEvaluator()
    override_profile_evaluator(evaluator)

    response = api_client.post(
        "/profile/evaluate",
        json={
            "profile_text": "Python, SQL, pandas and Airflow.",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert evaluator.last_call == {
        "profile_text": "Python, SQL, pandas and Airflow.",
        "top_k": 3,
        "include_category_prediction": True,
    }
    assert response.json() == {
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
                "description": "Python, SQL and Airflow role.",
            }
        ],
    }


def test_profile_evaluate_uses_configured_default_top_k(
    api_client: TestClient,
) -> None:
    """The endpoint should use settings.profile_top_k when top_k is omitted."""
    evaluator = FakeProfileEvaluator()
    override_profile_evaluator(evaluator)
    app.dependency_overrides[get_settings] = lambda: Settings(profile_top_k=7)

    response = api_client.post(
        "/profile/evaluate",
        json={"profile_text": "Python, SQL, pandas and Airflow."},
    )

    assert response.status_code == 200
    assert evaluator.last_call == {
        "profile_text": "Python, SQL, pandas and Airflow.",
        "top_k": 7,
        "include_category_prediction": True,
    }


def test_profile_evaluate_returns_422_for_blank_profile(
    api_client: TestClient,
) -> None:
    """Request validation errors should be exposed as HTTP 422 responses."""
    override_profile_evaluator(FakeProfileEvaluator())

    response = api_client.post("/profile/evaluate", json={"profile_text": "   "})

    assert response.status_code == 422
    assert "Profile text cannot be empty." in response.text


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (ValueError("top_k must be greater than 0."), 400, "top_k must be greater"),
        (
            RuntimeError("Could not generate a response with Ollama."),
            503,
            "Could not generate a response with Ollama.",
        ),
        (ConnectionError("Qdrant is unavailable."), 503, "Qdrant and Ollama"),
    ],
)
def test_profile_evaluate_maps_domain_errors(
    api_client: TestClient,
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    """Domain and dependency errors should become explicit HTTP responses."""
    app.dependency_overrides[get_profile_evaluator] = lambda: FailingProfileEvaluator(
        error
    )

    response = api_client.post(
        "/profile/evaluate",
        json={"profile_text": "Python and SQL."},
    )

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_profile_evaluator_is_built_lazily_and_cached(
    api_client: TestClient,
    monkeypatch,
) -> None:
    """The real dependency should build the evaluator once and reuse it."""
    evaluator = FakeProfileEvaluator()
    build_calls = 0

    def fake_builder(settings: Settings) -> FakeProfileEvaluator:
        nonlocal build_calls
        build_calls += 1
        assert settings is app.state.settings
        return evaluator

    monkeypatch.setattr(profile_router, "build_profile_evaluator", fake_builder)

    first_response = api_client.post(
        "/profile/evaluate",
        json={"profile_text": "Python and SQL."},
    )
    second_response = api_client.post(
        "/profile/evaluate",
        json={"profile_text": "Python and SQL."},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert build_calls == 1
    assert app.state.profile_evaluator is evaluator
    assert app.state.profile_evaluator_error is None


def test_profile_evaluator_build_error_returns_503(
    api_client: TestClient,
    monkeypatch,
) -> None:
    """Dependency startup failures should be explicit service errors."""

    def fake_builder(_settings: Settings) -> FakeProfileEvaluator:
        raise ConnectionError("Qdrant is unavailable.")

    monkeypatch.setattr(profile_router, "build_profile_evaluator", fake_builder)

    response = api_client.post(
        "/profile/evaluate",
        json={"profile_text": "Python and SQL."},
    )

    assert response.status_code == 503
    assert "Qdrant is unavailable" in response.json()["detail"]
    assert app.state.profile_evaluator is None
    assert isinstance(app.state.profile_evaluator_error, ConnectionError)
