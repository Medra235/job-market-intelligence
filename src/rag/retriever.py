"""Reusable Qdrant retriever for the job-postings RAG corpus."""

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.rag.index_corpus import (
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_QDRANT_URL,
    get_embedding_model,
    get_qdrant_client,
)


@dataclass(frozen=True)
class RetrievedDocument:
    """Retrieved job posting returned by the vector store."""

    job_id: int | None
    title: str
    job_category: str
    description_preview: str
    score: float
    payload: dict[str, Any]

    @classmethod
    def from_qdrant_point(cls, point) -> "RetrievedDocument":
        """Build a retrieved document from a Qdrant scored point."""
        payload = point.payload or {}
        return cls(
            job_id=payload.get("job_id"),
            title=payload.get("title", ""),
            job_category=payload.get("job_category", ""),
            description_preview=payload.get("description_preview", ""),
            score=point.score,
            payload=payload,
        )


class QdrantRetriever:
    """Semantic retriever backed by sentence-transformers and Qdrant."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        local_files_only: bool = False,
        client: QdrantClient | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.model = model or get_embedding_model(
            model_name, local_files_only=local_files_only
        )
        self.client = client or get_qdrant_client(qdrant_url)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the most semantically similar job postings for a query."""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_vector = self.model.encode(query, show_progress_bar=False).tolist()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        return [
            RetrievedDocument.from_qdrant_point(point) for point in response.points
        ]


def retrieve(
    query: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    local_files_only: bool = False,
    score_threshold: float | None = None,
) -> list[RetrievedDocument]:
    """Convenience function for one-off retrieval calls."""
    retriever = QdrantRetriever(
        collection_name=collection_name,
        model_name=model_name,
        qdrant_url=qdrant_url,
        local_files_only=local_files_only,
    )
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
    )
