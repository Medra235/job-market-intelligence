"""Run manual semantic-search checks against the Qdrant collection.

Usage examples:
    .venv/bin/python src/rag/test_retrieval.py --query "python sql airflow aws"
    .venv/bin/python src/rag/test_retrieval.py --query "react frontend typescript" --top-k 3
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.rag.index_corpus import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_QDRANT_URL,
)
from src.rag.retriever import retrieve  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retrieval testing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        required=True,
        help="Text query to test against the indexed corpus.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve from Qdrant.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name to query.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Sentence-transformers embedding model name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
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
    return parser.parse_args()


def print_results(results) -> None:
    """Print the retrieved documents in a readable terminal format."""
    if not results:
        print("No results returned.")
        return

    print("\nTop retrieved documents:\n")
    for index, document in enumerate(results, start=1):
        print(f"{index}. score={document.score:.4f}")
        print(f"   title: {document.title or '<missing>'}")
        print(f"   category: {document.job_category or '<missing>'}")
        print(f"   job_id: {document.job_id or '<missing>'}")
        print(f"   preview: {document.description_preview or '<missing preview>'}")
        print()


def main() -> None:
    """Run one semantic query against the indexed job-posting collection."""
    args = parse_args()
    results = retrieve(
        query=args.query,
        top_k=args.top_k,
        collection_name=args.collection,
        model_name=args.model_name,
        qdrant_url=args.qdrant_url,
        local_files_only=args.local_files_only,
        score_threshold=args.score_threshold,
    )
    print_results(results)


if __name__ == "__main__":
    main()
