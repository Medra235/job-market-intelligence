"""Tests for the Streamlit frontend API client."""

import httpx
import pytest

from job_market_intelligence.frontend.api_client import (
    ApiClientError,
    JobMarketApiClient,
)


def test_api_client_builds_expected_profile_request() -> None:
    """Profile evaluation should be delegated to the backend API."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = request.read().decode()
        return httpx.Response(
            status_code=200,
            json={
                "category_prediction": {"label": "Data Engineering"},
                "evaluation": "Keep building SQL skills.",
                "sources": [],
            },
        )

    client = JobMarketApiClient(
        base_url="http://api.test/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.evaluate_profile(
        profile_text="Python, SQL and pandas.",
        top_k=3,
    )

    assert response["evaluation"] == "Keep building SQL skills."
    assert captured_request["url"] == "http://api.test/profile/evaluate"
    assert '"profile_text":"Python, SQL and pandas."' in captured_request["payload"]
    assert '"top_k":3' in captured_request["payload"]


def test_api_client_sends_chat_history() -> None:
    """The chat client should send user history without local RAG logic."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["payload"] = request.read().decode()
        return httpx.Response(
            status_code=200,
            json={
                "message": "What next?",
                "answer": "Prioritize projects.",
                "sources": [],
            },
        )

    client = JobMarketApiClient(
        base_url="http://api.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.chat(
        message="What next?",
        history=[{"role": "user", "content": "I know Python."}],
        top_k=2,
    )

    assert response["answer"] == "Prioritize projects."
    assert '"message":"What next?"' in captured_request["payload"]
    assert '"history":[{"role":"user","content":"I know Python."}]' in (
        captured_request["payload"]
    )
    assert '"top_k":2' in captured_request["payload"]


def test_api_client_sends_market_limit_as_query_param() -> None:
    """Market filters should be sent as query parameters."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        return httpx.Response(status_code=200, json=[])

    client = JobMarketApiClient(
        base_url="http://api.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.get_category_distribution(limit=20)

    assert response == []
    assert captured_request["url"] == "http://api.test/market/categories?limit=20"


def test_api_client_wraps_backend_error_detail() -> None:
    """HTTP errors should expose the FastAPI detail message."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            json={"detail": "RAG chatbot is not available."},
        )

    client = JobMarketApiClient(
        base_url="http://api.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ApiClientError) as error:
        client.chat(message="Hello", history=[])

    assert str(error.value) == "RAG chatbot is not available."
    assert error.value.status_code == 503
