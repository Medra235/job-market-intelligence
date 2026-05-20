"""Tests for RAG chatbot API route functions."""

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from job_market_intelligence.api.main import app, health_check
from job_market_intelligence.api.routers.chat import chat_with_rag, get_chat_rag_chain
from job_market_intelligence.api.schemas import ChatRequest
from job_market_intelligence.config.settings import Settings
from job_market_intelligence.rag.generator import GenerationConfig
from job_market_intelligence.rag.rag_chain import JobMarketRagChain
from job_market_intelligence.rag.retriever import RetrievedDocument


def sample_document() -> RetrievedDocument:
    """Return one retrieved document for chatbot API tests."""
    return RetrievedDocument(
        job_id=456,
        title="Machine Learning Engineer",
        job_category="AI/ML",
        description="Python, model deployment and MLOps role.",
        description_preview="Python and MLOps role.",
        score=0.91,
        payload={"job_id": 456},
    )


class FakeRetriever:
    """Fake retriever for API contract tests."""

    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """Return deterministic retrieved documents."""
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        return [sample_document()]


class FakeGenerator:
    """Fake generator for API contract tests."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Return a deterministic answer."""
        self.last_prompt = prompt
        return "Prioriza Python, deployment y MLOps con evidencia del corpus."


class FailingRetriever(FakeRetriever):
    """Fake retriever that simulates unavailable dependencies."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """Raise the configured error."""
        raise self.error


class FakeAppState:
    """Minimal app state compatible with FastAPI request.app.state."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.chat_rag_chain = None
        self.chat_rag_chain_error = None


class FakeRequest:
    """Minimal request object exposing app.state for dependency tests."""

    def __init__(self) -> None:
        self.app = type("FakeApp", (), {"state": FakeAppState()})()


def build_fake_chain(
    retriever: FakeRetriever | None = None,
    generator: FakeGenerator | None = None,
) -> JobMarketRagChain:
    """Build a RAG chain with fake dependencies."""
    return JobMarketRagChain(
        retriever=retriever or FakeRetriever(),
        generator=generator or FakeGenerator(),
    )


def test_chat_returns_answer_and_sources() -> None:
    """The chat endpoint should expose the generated answer and RAG sources."""
    retriever = FakeRetriever()
    generator = FakeGenerator()
    chain = build_fake_chain(retriever=retriever, generator=generator)
    request = ChatRequest(
        message="What should I learn for ML Engineer roles?",
        top_k=3,
    )

    response = asyncio.run(chat_with_rag(request, chain, Settings()))

    assert retriever.last_call == {
        "query": "What should I learn for ML Engineer roles?",
        "top_k": 3,
        "score_threshold": None,
    }
    assert response.model_dump() == {
        "message": "What should I learn for ML Engineer roles?",
        "answer": "Prioriza Python, deployment y MLOps con evidencia del corpus.",
        "sources": [
            {
                "job_id": 456,
                "title": "Machine Learning Engineer",
                "job_category": "AI/ML",
                "score": 0.91,
                "description_preview": "Python and MLOps role.",
            }
        ],
    }


def test_chat_uses_configured_default_top_k() -> None:
    """The chat endpoint should use settings when top_k is omitted."""
    retriever = FakeRetriever()
    chain = build_fake_chain(retriever=retriever)
    request = ChatRequest(message="Compare backend and data engineering skills.")
    settings = Settings(max_context_documents=7)

    asyncio.run(chat_with_rag(request, chain, settings))

    assert retriever.last_call == {
        "query": "Compare backend and data engineering skills.",
        "top_k": 7,
        "score_threshold": None,
    }


def test_chat_includes_request_history_without_mutating_cached_chain() -> None:
    """Client-provided history should be scoped to the current request only."""
    generator = FakeGenerator()
    chain = build_fake_chain(generator=generator)
    request = ChatRequest(
        message="And what should come next?",
        history=[
            {
                "role": "user",
                "content": "I know Python and SQL.",
            },
            {
                "role": "assistant",
                "content": "That is a good foundation for data roles.",
            },
        ],
    )

    asyncio.run(chat_with_rag(request, chain, Settings()))

    assert generator.last_prompt is not None
    assert "Historial de conversación previo" in generator.last_prompt
    assert "usuario: I know Python and SQL." in generator.last_prompt
    assert "asistente: That is a good foundation for data roles." in generator.last_prompt
    assert chain.memory.messages == []


@pytest.mark.parametrize("message", ["", "   "])
def test_chat_request_rejects_empty_message(message: str) -> None:
    """The API request schema should reject blank user messages."""
    with pytest.raises(ValidationError):
        ChatRequest(message=message)


def test_chat_request_rejects_empty_history_content() -> None:
    """The API request schema should reject blank history messages."""
    with pytest.raises(ValidationError):
        ChatRequest(
            message="What should I learn?",
            history=[{"role": "user", "content": "   "}],
        )


def test_chat_route_is_registered_without_test_client() -> None:
    """The FastAPI app should register the chat endpoint."""
    paths = {route.path for route in app.routes}

    assert "/chat" in paths


def test_health_route_is_registered_without_test_client() -> None:
    """The FastAPI app should expose a lightweight health endpoint."""
    paths = {route.path for route in app.routes}

    assert "/health" in paths


def test_health_check_returns_ok_status() -> None:
    """The health endpoint should not depend on external services."""
    response = asyncio.run(health_check())

    assert response == {"status": "ok"}


def test_get_chat_rag_chain_reuses_cached_chain() -> None:
    """The dependency should reuse the chain stored in app state."""
    request = FakeRequest()
    chain = build_fake_chain()
    request.app.state.chat_rag_chain = chain

    response = asyncio.run(get_chat_rag_chain(request))

    assert response is chain


def test_get_chat_rag_chain_builds_chain_lazily(monkeypatch) -> None:
    """The dependency should build and cache the RAG chain on first use."""
    request = FakeRequest()
    chain = build_fake_chain()

    def fake_builder(settings: Settings) -> JobMarketRagChain:
        assert settings is request.app.state.settings
        return chain

    monkeypatch.setattr(
        "job_market_intelligence.api.routers.chat.build_chat_rag_chain",
        fake_builder,
    )

    response = asyncio.run(get_chat_rag_chain(request))

    assert response is chain
    assert request.app.state.chat_rag_chain is chain
    assert request.app.state.chat_rag_chain_error is None


def test_get_chat_rag_chain_maps_build_errors_to_503(monkeypatch) -> None:
    """Dependency startup failures should be explicit service errors."""
    request = FakeRequest()

    def fake_builder(settings: Settings) -> JobMarketRagChain:
        raise ConnectionError("Qdrant is unavailable.")

    monkeypatch.setattr(
        "job_market_intelligence.api.routers.chat.build_chat_rag_chain",
        fake_builder,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_chat_rag_chain(request))

    assert exc_info.value.status_code == 503
    assert "Qdrant is unavailable" in str(exc_info.value.detail)
    assert request.app.state.chat_rag_chain is None
    assert isinstance(request.app.state.chat_rag_chain_error, ConnectionError)


def test_chat_maps_validation_errors_to_400() -> None:
    """Domain validation errors should become explicit client errors."""
    request = ChatRequest(message="What should I learn?")
    chain = build_fake_chain(retriever=FailingRetriever(ValueError("bad query")))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat_with_rag(request, chain, Settings()))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "bad query"


def test_chat_maps_runtime_errors_to_503() -> None:
    """External dependency failures should become service unavailable errors."""
    request = ChatRequest(message="What should I learn?")
    chain = build_fake_chain(
        retriever=FailingRetriever(RuntimeError("Could not reach Qdrant."))
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat_with_rag(request, chain, Settings()))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Could not reach Qdrant."


def test_chat_maps_unexpected_dependency_errors_to_503() -> None:
    """Unexpected Qdrant/Ollama failures should avoid generic 500 responses."""
    request = ChatRequest(message="What should I learn?")
    chain = build_fake_chain(retriever=FailingRetriever(ConnectionError("down")))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat_with_rag(request, chain, Settings()))

    assert exc_info.value.status_code == 503
    assert "Qdrant and Ollama" in str(exc_info.value.detail)
