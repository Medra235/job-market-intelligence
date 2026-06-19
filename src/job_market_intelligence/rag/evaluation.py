"""Shared helpers for RAG evaluation scripts."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.rag_chain import RagResponse
from job_market_intelligence.rag.retriever import RetrievedDocument

DEFAULT_GOLDEN_SET_PATH = Path("data/evaluation/rag_golden_set.yaml")
DEFAULT_REVIEW_QUERY_IDS = [
    "data_engineering_learning_path",
    "devops_platform_skills",
    "ai_ml_engineer_skills",
    "backend_api_development",
    "follow_up_mlops_deployment",
]


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


@dataclass(frozen=True)
class RelevanceMatch:
    """Rule-based relevance decision for a retrieved document."""

    is_relevant: bool
    category_match: bool
    matched_keywords: list[str]


@dataclass(frozen=True)
class QueryEvaluation:
    """Retrieval metrics for one query."""

    golden_query: GoldenQuery
    top_k: int
    returned_count: int
    relevant_count: int
    precision_at_k: float
    hit_at_k: int
    mrr_at_k: float


def normalize_text(text: str) -> str:
    """Normalize text for simple case-insensitive matching."""
    return " ".join(text.casefold().split())


def evaluate_document_relevance(
    document: RetrievedDocument,
    expected_categories: list[str],
    expected_keywords: list[str],
) -> RelevanceMatch:
    """Evaluate document relevance with category and keyword proxy signals."""
    normalized_categories = {
        normalize_text(category) for category in expected_categories if category
    }
    category_match = normalize_text(document.job_category) in normalized_categories

    searchable_text = normalize_text(
        " ".join(
            [
                document.title,
                document.job_category,
                document.description,
                document.description_preview,
            ]
        )
    )
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword and normalize_text(keyword) in searchable_text
    ]
    return RelevanceMatch(
        is_relevant=category_match or bool(matched_keywords),
        category_match=category_match,
        matched_keywords=matched_keywords,
    )


def calculate_metrics(
    golden_query: GoldenQuery,
    retrieved_documents: list[RetrievedDocument],
    top_k: int,
) -> QueryEvaluation:
    """Calculate Precision@k, Hit@k, and MRR@k for one query."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    relevant_ranks = [
        rank
        for rank, document in enumerate(retrieved_documents[:top_k], start=1)
        if evaluate_document_relevance(
            document=document,
            expected_categories=golden_query.expected_categories,
            expected_keywords=golden_query.expected_keywords,
        ).is_relevant
    ]
    relevant_count = len(relevant_ranks)
    return QueryEvaluation(
        golden_query=golden_query,
        top_k=top_k,
        returned_count=len(retrieved_documents[:top_k]),
        relevant_count=relevant_count,
        precision_at_k=relevant_count / top_k,
        hit_at_k=int(bool(relevant_ranks)),
        mrr_at_k=(1 / relevant_ranks[0]) if relevant_ranks else 0.0,
    )


def select_queries(
    queries: list[GoldenQuery],
    requested_ids: list[str] | None = None,
    include_all: bool = False,
    limit: int | None = None,
) -> list[GoldenQuery]:
    """Select golden-set queries for manual review."""
    query_by_id = {query.query_id: query for query in queries}
    requested_ids = requested_ids or []

    if requested_ids:
        missing_ids = [
            query_id for query_id in requested_ids if query_id not in query_by_id
        ]
        if missing_ids:
            raise ValueError(f"Unknown golden-set query id(s): {', '.join(missing_ids)}")
        selected = [query_by_id[query_id] for query_id in requested_ids]
    elif include_all:
        selected = queries
    else:
        selected = [
            query_by_id[query_id]
            for query_id in DEFAULT_REVIEW_QUERY_IDS
            if query_id in query_by_id
        ]

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No golden-set queries selected.")
    return selected


def format_sources(documents: list[RetrievedDocument]) -> str:
    """Format retrieved sources for manual review."""
    if not documents:
        return "No retrieved sources."

    sections = []
    for index, document in enumerate(documents, start=1):
        description = document.description or document.description_preview or ""
        sections.append(
            "\n".join(
                [
                    f"{index}. **{document.title or '<missing title>'}**",
                    f"   - Category: {document.job_category or '<missing>'}",
                    f"   - Score: {document.score:.4f}",
                    f"   - Job ID: {document.job_id or '<missing>'}",
                    "   - Description:",
                    "",
                    f"     {description.strip() or '<missing description>'}",
                ]
            )
        )
    return "\n\n".join(sections)


def build_review_markdown(
    cases: list[tuple[GoldenQuery, RagResponse]],
    top_k: int,
    llm_model: str,
) -> str:
    """Build the Markdown review file from generated RAG responses."""
    lines = [
        "# RAG Manual Generation Review",
        "",
        f"- LLM model: `{llm_model}`",
        f"- Retrieval top_k: `{top_k}`",
        "- Scores: `0 = poor`, `1 = partial`, `2 = good`",
        "",
        "Manual criteria:",
        "",
        "- Context relevance: retrieved sources contain useful evidence for the query.",
        "- Faithfulness: the answer is supported by the retrieved sources.",
        "- Answer relevance: the answer directly addresses the user's goal.",
        "",
    ]

    for golden_query, response in cases:
        lines.extend([f"## {golden_query.query_id}", "", "### User Query", ""])
        lines.extend([golden_query.query, ""])
        if response.query != golden_query.query:
            lines.extend(["### Query Used By RAG", "", response.query, ""])
        lines.extend(
            [
                "### Answer",
                "",
                response.answer,
                "",
                "### Retrieved Sources",
                "",
                format_sources(response.documents),
                "",
                "### Manual Scores",
                "",
                "- Context relevance: _/2",
                "- Faithfulness: _/2",
                "- Answer relevance: _/2",
                "",
            ]
        )

    return "\n".join(lines)
