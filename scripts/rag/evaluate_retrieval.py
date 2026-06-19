"""Evaluate RAG retrieval against a small manual golden set.

Usage example:
    uv run python scripts/rag/evaluate_retrieval.py
    uv run python scripts/rag/evaluate_retrieval.py --top-k 3 --local-files-only
"""

import argparse
import csv
from pathlib import Path
from statistics import mean

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    GoldenQuery,
    QueryEvaluation,
    calculate_metrics,
    load_golden_set,
)
from job_market_intelligence.rag.retriever import QdrantRetriever

DEFAULT_METRICS_OUTPUT_PATH = Path("reports/rag/retrieval_metrics.csv")


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
