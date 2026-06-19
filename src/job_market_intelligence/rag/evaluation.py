"""Shared helpers for RAG evaluation scripts."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from job_market_intelligence.config.settings import settings

DEFAULT_GOLDEN_SET_PATH = Path("data/evaluation/rag_golden_set.yaml")


@dataclass(frozen=True)
class GoldenQuery:
    """One manual retrieval evaluation case."""

    query_id: str
    query: str
    expected_categories: list[str]
    expected_keywords: list[str]
    standalone_query: str | None = None

    @property
    def retrieval_query(self) -> str:
        """Return the query used for retrieval evaluation."""
        return self.standalone_query or self.query


def load_golden_set(path: Path) -> tuple[list[GoldenQuery], int]:
    """Load golden queries and default top_k from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Golden set must be a YAML mapping.")

    metadata = data.get("metadata", {})
    default_top_k = int(metadata.get("default_top_k", settings.max_context_documents))
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("Golden set must contain a non-empty 'queries' list.")

    queries = []
    seen_ids = set()
    for raw_query in raw_queries:
        query_id = str(raw_query["id"]).strip()
        if query_id in seen_ids:
            raise ValueError(f"Duplicate golden-set query id: {query_id}")
        seen_ids.add(query_id)

        queries.append(
            GoldenQuery(
                query_id=query_id,
                query=str(raw_query["query"]).strip(),
                standalone_query=(
                    str(raw_query["standalone_query"]).strip()
                    if raw_query.get("standalone_query")
                    else None
                ),
                expected_categories=[
                    str(category).strip()
                    for category in raw_query.get("expected_categories", [])
                ],
                expected_keywords=[
                    str(keyword).strip()
                    for keyword in raw_query.get("expected_keywords", [])
                ],
            )
        )

    return queries, default_top_k
