"""Tests for the reusable RAG retriever."""

from types import SimpleNamespace

import pytest

from job_market_intelligence.rag.retriever import QdrantRetriever, RetrievedDocument


class FakeVector:
    """Minimal vector object compatible with sentence-transformers output."""

    def tolist(self) -> list[float]:
        """Return a plain list as expected by qdrant-client."""
        return [0.1, 0.2, 0.3]


class FakeModel:
    """Fake embedding model that records the encoded query."""

    def __init__(self) -> None:
        self.encoded_text: str | None = None

    def encode(self, text: str, show_progress_bar: bool = False) -> FakeVector:
        """Return a deterministic vector without loading a real model."""
        self.encoded_text = text
        return FakeVector()


class FakeClient:
    """Fake Qdrant client that records query parameters."""

    def __init__(self) -> None:
        self.last_call: dict | None = None

    def query_points(self, **kwargs):
        """Return one scored point with a realistic payload."""
        self.last_call = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.75,
                    payload={
                        "job_id": 123,
                        "title": "Data Engineer",
                        "job_category": "Data Engineering",
                        "description": "Full Python, SQL and Airflow role.",
                        "description_preview": "Python, SQL and Airflow role.",
                    },
                )
            ]
        )


def test_retrieved_document_from_qdrant_point() -> None:
    """Convert a Qdrant scored point into the project retrieval schema."""
    point = SimpleNamespace(
        score=0.82,
        payload={
            "job_id": 456,
            "title": "Frontend Developer",
            "job_category": "Frontend",
            "description": "Full React and TypeScript role.",
            "description_preview": "React and TypeScript role.",
        },
    )

    document = RetrievedDocument.from_qdrant_point(point)

    assert document.job_id == 456
    assert document.title == "Frontend Developer"
    assert document.job_category == "Frontend"
    assert document.description == "Full React and TypeScript role."
    assert document.description_preview == "React and TypeScript role."
    assert document.score == 0.82
    assert document.payload == point.payload


def test_retriever_queries_qdrant_with_expected_parameters() -> None:
    """The retriever should encode the query and request payload-only results."""
    model = FakeModel()
    client = FakeClient()
    retriever = QdrantRetriever(
        collection_name="job_postings",
        client=client,
        model=model,
    )

    results = retriever.retrieve(
        query="python sql airflow aws",
        top_k=3,
        score_threshold=0.4,
    )

    assert model.encoded_text == "python sql airflow aws"
    assert client.last_call == {
        "collection_name": "job_postings",
        "query": [0.1, 0.2, 0.3],
        "limit": 3,
        "with_payload": True,
        "with_vectors": False,
        "score_threshold": 0.4,
    }
    assert results == [
        RetrievedDocument(
            job_id=123,
            title="Data Engineer",
            job_category="Data Engineering",
            description="Full Python, SQL and Airflow role.",
            description_preview="Python, SQL and Airflow role.",
            score=0.75,
            payload={
                "job_id": 123,
                "title": "Data Engineer",
                "job_category": "Data Engineering",
                "description": "Full Python, SQL and Airflow role.",
                "description_preview": "Python, SQL and Airflow role.",
            },
        )
    ]


@pytest.mark.parametrize("query", ["", "   "])
def test_retriever_rejects_empty_queries(query: str) -> None:
    """Empty queries should fail before reaching Qdrant."""
    retriever = QdrantRetriever(client=FakeClient(), model=FakeModel())

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve(query=query)


def test_retriever_rejects_invalid_top_k() -> None:
    """top_k must be positive."""
    retriever = QdrantRetriever(client=FakeClient(), model=FakeModel())

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.retrieve(query="data engineering", top_k=0)
