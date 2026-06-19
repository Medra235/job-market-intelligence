"""Evaluate RAG retrieval against a small manual golden set.

Usage example:
    uv run python scripts/rag/evaluate_retrieval.py
    uv run python scripts/rag/evaluate_retrieval.py --top-k 3 --local-files-only
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    GoldenQuery,
    load_golden_set,
)
from job_market_intelligence.rag.retriever import QdrantRetriever, RetrievedDocument

DEFAULT_METRICS_OUTPUT_PATH = Path("reports/rag/retrieval_metrics.csv")


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retrieval evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
        help="Path to the YAML golden set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of documents to retrieve. Defaults to the golden-set value.",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--model-name",
        default=settings.embedding_model,
        help="Sentence-transformers embedding model name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url,
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional minimum similarity score for returned results.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT_PATH,
        help="CSV output path for query-level metrics.",
    )
    return parser.parse_args()


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


def evaluate_queries(
    queries: list[GoldenQuery],
    retriever: QdrantRetriever,
    top_k: int,
    score_threshold: float | None = None,
) -> list[QueryEvaluation]:
    """Retrieve documents and calculate metrics for all golden-set queries."""
    evaluations = []
    for golden_query in queries:
        documents = retriever.retrieve(
            query=golden_query.retrieval_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        evaluations.append(
            calculate_metrics(
                golden_query=golden_query,
                retrieved_documents=documents,
                top_k=top_k,
            )
        )
    return evaluations


def write_metrics_csv(evaluations: list[QueryEvaluation], output_path: Path) -> None:
    """Write query-level retrieval metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "query",
                "retrieval_query",
                "top_k",
                "returned_count",
                "relevant_count",
                "precision_at_k",
                "hit_at_k",
                "mrr_at_k",
            ],
        )
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(
                {
                    "query_id": evaluation.golden_query.query_id,
                    "query": evaluation.golden_query.query,
                    "retrieval_query": evaluation.golden_query.retrieval_query,
                    "top_k": evaluation.top_k,
                    "returned_count": evaluation.returned_count,
                    "relevant_count": evaluation.relevant_count,
                    "precision_at_k": round(evaluation.precision_at_k, 4),
                    "hit_at_k": evaluation.hit_at_k,
                    "mrr_at_k": round(evaluation.mrr_at_k, 4),
                }
            )


def print_summary(evaluations: list[QueryEvaluation], metrics_output: Path) -> None:
    """Print a compact terminal summary."""
    print("\nRAG retrieval evaluation completed.")
    print(f"Queries: {len(evaluations)}")
    print(f"Mean Precision@k: {mean(e.precision_at_k for e in evaluations):.4f}")
    print(f"Mean Hit@k: {mean(e.hit_at_k for e in evaluations):.4f}")
    print(f"Mean MRR@k: {mean(e.mrr_at_k for e in evaluations):.4f}")
    print(f"Metrics written to: {metrics_output}")


def main() -> None:
    """Run retrieval evaluation from the command line."""
    args = parse_args()
    queries, default_top_k = load_golden_set(args.golden_set)
    top_k = args.top_k or default_top_k

    retriever = QdrantRetriever(
        collection_name=args.collection,
        model_name=args.model_name,
        qdrant_url=args.qdrant_url,
        local_files_only=args.local_files_only,
    )
    evaluations = evaluate_queries(
        queries=queries,
        retriever=retriever,
        top_k=top_k,
        score_threshold=args.score_threshold,
    )
    write_metrics_csv(evaluations, args.metrics_output)
    print_summary(evaluations, args.metrics_output)


if __name__ == "__main__":
    main()
