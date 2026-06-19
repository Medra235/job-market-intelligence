"""Tests for RAG chatbot API endpoints."""

import pytest
from fastapi.testclient import TestClient

from job_market_intelligence.api.main import app
from job_market_intelligence.api.routers import chat as chat_router
from job_market_intelligence.api.routers.chat import get_chat_rag_chain, get_settings
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
        _ = (system_prompt, config)
        self.last_prompt = prompt
        return "Prioriza Python, deployment y MLOps con evidencia del corpus."


class FakeQueryRewriter:
    """Fake query rewriter for API contract tests."""

    def __init__(self, rewritten_query: str | None = None) -> None:
        self.rewritten_query = rewritten_query
        self.calls = []

    def rewrite(self, query: str, conversation_history) -> str:
        """Return a deterministic retrieval query."""
        self.calls.append(
            {
                "query": query,
                "conversation_history": list(conversation_history),
            }
        )
        return self.rewritten_query or query


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
        _ = (query, top_k, score_threshold)
        raise self.error


def build_fake_chain(
    retriever: FakeRetriever | None = None,
    generator: FakeGenerator | None = None,
    query_rewriter: FakeQueryRewriter | None = None,
) -> JobMarketRagChain:
    """Build a RAG chain with fake dependencies."""
    return JobMarketRagChain(
        retriever=retriever or FakeRetriever(),
        generator=generator or FakeGenerator(),
        query_rewriter=query_rewriter or FakeQueryRewriter(),
    )


def override_chat_chain(chain: JobMarketRagChain) -> None:
    """Register the fake chat chain dependency."""
    app.dependency_overrides[get_chat_rag_chain] = lambda: chain


def test_health_check_returns_ok(api_client: TestClient) -> None:
    """The health endpoint should work through FastAPI routing."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources(api_client: TestClient) -> None:
    """The chat endpoint should expose the generated answer and RAG sources."""
    retriever = FakeRetriever()
    chain = build_fake_chain(retriever=retriever)
    override_chat_chain(chain)

    response = api_client.post(
        "/chat",
        json={
            "message": "What should I learn for ML Engineer roles?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert retriever.last_call == {
        "query": "What should I learn for ML Engineer roles?",
        "top_k": 3,
        "score_threshold": None,
    }
    assert response.json() == {
        "message": "What should I learn for ML Engineer roles?",
        "retrieval_query": "What should I learn for ML Engineer roles?",
        "answer": "Prioriza Python, deployment y MLOps con evidencia del corpus.",
        "sources": [
            {
                "job_id": 456,
                "title": "Machine Learning Engineer",
                "job_category": "AI/ML",
                "score": 0.91,
                "description": "Python, model deployment and MLOps role.",
                "description_preview": "Python and MLOps role.",
            }
        ],
    }


def test_chat_uses_configured_default_top_k(api_client: TestClient) -> None:
    """The endpoint should use settings.max_context_documents when top_k is omitted."""
    retriever = FakeRetriever()
    chain = build_fake_chain(retriever=retriever)
    override_chat_chain(chain)
    app.dependency_overrides[get_settings] = lambda: Settings(max_context_documents=7)

    response = api_client.post(
        "/chat",
        json={"message": "Compare backend and data engineering skills."},
    )

    assert response.status_code == 200
    assert retriever.last_call == {
        "query": "Compare backend and data engineering skills.",
        "top_k": 7,
        "score_threshold": None,
    }


def test_chat_uses_request_history_without_mutating_cached_chain(
    api_client: TestClient,
) -> None:
    """Client-provided history should be scoped to the current request only."""
    retriever = FakeRetriever()
    generator = FakeGenerator()
    query_rewriter = FakeQueryRewriter(
        rewritten_query="What should come next after Python and SQL for data roles?"
    )
    chain = build_fake_chain(
        retriever=retriever,
        generator=generator,
        query_rewriter=query_rewriter,
    )
    override_chat_chain(chain)

    response = api_client.post(
        "/chat",
        json={
            "message": "And what should come next?",
            "history": [
                {
                    "role": "user",
                    "content": "I know Python and SQL.",
                },
                {
                    "role": "assistant",
                    "content": "That is a good foundation for data roles.",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["retrieval_query"] == (
        "What should come next after Python and SQL for data roles?"
    )
    assert retriever.last_call["query"] == response.json()["retrieval_query"]
    assert generator.last_prompt is not None
    assert "Historial de conversación previo" in generator.last_prompt
    assert "usuario: I know Python and SQL." in generator.last_prompt
    assert "asistente: That is a good foundation for data roles." in generator.last_prompt
    assert chain.memory.messages == []
    assert len(query_rewriter.calls[0]["conversation_history"]) == 2


def test_chat_returns_422_for_blank_message(api_client: TestClient) -> None:
    """Request validation errors should be exposed as HTTP 422 responses."""
    override_chat_chain(build_fake_chain())

    response = api_client.post("/chat", json={"message": "   "})

    assert response.status_code == 422
    assert "Message cannot be empty." in response.text


def test_chat_returns_422_for_blank_history_message(api_client: TestClient) -> None:
    """History validation errors should be exposed as HTTP 422 responses."""
    override_chat_chain(build_fake_chain())

    response = api_client.post(
        "/chat",
        json={
            "message": "What should I learn?",
            "history": [{"role": "user", "content": "   "}],
        },
    )

    assert response.status_code == 422
    assert "Chat message content cannot be empty." in response.text


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (ValueError("bad query"), 400, "bad query"),
        (RuntimeError("Could not reach Qdrant."), 503, "Could not reach Qdrant."),
        (ConnectionError("down"), 503, "Qdrant and Ollama"),
    ],
)
def test_chat_maps_domain_errors(
    api_client: TestClient,
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    """Domain and dependency errors should become explicit HTTP responses."""
    chain = build_fake_chain(retriever=FailingRetriever(error))
    override_chat_chain(chain)

    response = api_client.post("/chat", json={"message": "What should I learn?"})

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_chat_chain_is_built_lazily_and_cached(
    api_client: TestClient,
    monkeypatch,
) -> None:
    """The real dependency should build the RAG chain once and reuse it."""
    chain = build_fake_chain()
    build_calls = 0

    def fake_builder(settings: Settings) -> JobMarketRagChain:
        nonlocal build_calls
        build_calls += 1
        assert settings is app.state.settings
        return chain

    monkeypatch.setattr(chat_router, "build_chat_rag_chain", fake_builder)

    first_response = api_client.post("/chat", json={"message": "What should I learn?"})
    second_response = api_client.post(
        "/chat",
        json={"message": "What should I learn?"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert build_calls == 1
    assert app.state.chat_rag_chain is chain
    assert app.state.chat_rag_chain_error is None


def test_chat_chain_build_error_returns_503(
    api_client: TestClient,
    monkeypatch,
) -> None:
    """Dependency startup failures should be explicit service errors."""

    def fake_builder(_settings: Settings) -> JobMarketRagChain:
        raise ConnectionError("Qdrant is unavailable.")

    monkeypatch.setattr(chat_router, "build_chat_rag_chain", fake_builder)

    response = api_client.post("/chat", json={"message": "What should I learn?"})

    assert response.status_code == 503
    assert "Qdrant is unavailable" in response.json()["detail"]
    assert app.state.chat_rag_chain is None
    assert isinstance(app.state.chat_rag_chain_error, ConnectionError)
