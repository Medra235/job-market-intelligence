"""Run a manual multi-turn RAG check.

Usage example:
    uv run python scripts/rag/check_rag_chain.py --query "What should I learn for Data Engineering?"
    uv run python scripts/rag/check_rag_chain.py --query "What should I learn for Data Engineering?" --follow-up-query "And for ML Engineer?"
"""

import argparse

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.generator import (
    GenerationConfig,
    LlamaCppGenerator,
    TextGenerator,
)
from job_market_intelligence.rag.rag_chain import JobMarketRagChain
from job_market_intelligence.rag.retriever import QdrantRetriever

DEFAULT_DESCRIPTION_PREVIEW_CHARS = 240


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manual RAG check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="User query to answer.")
    parser.add_argument(
        "--follow-up-query",
        action="append",
        default=[],
        help=(
            "Optional follow-up query. "
            "Repeat the flag to simulate a multi-turn conversation."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top_k.")
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help="Sentence-transformers embedding model name.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="llama.cpp model name. Defaults to the configured model.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url,
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--llamacpp-url",
        default=settings.llamacpp_url,
        help="llama.cpp server base URL.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--max-context-documents",
        type=int,
        default=5,
        help="Maximum retrieved documents included in the LLM prompt.",
    )
    parser.add_argument(
        "--max-preview-chars",
        type=int,
        default=700,
        help="Maximum characters per retrieved description preview.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1536,
        help="Maximum generated tokens.",
    )
    return parser.parse_args()


def build_generator(args: argparse.Namespace) -> TextGenerator:
    """Build the local llama.cpp generator for the smoke test."""
    model_name = args.llm_model or settings.llamacpp_model
    return LlamaCppGenerator(
        model_name=model_name,
        base_url=args.llamacpp_url,
    )


def print_turn_response(turn_number: int, query: str, response) -> None:
    """Print one conversation turn in a readable terminal format."""
    print(f"\nTurn {turn_number}: {query}")
    if response.retrieval_query != response.query:
        print(f"Retrieval query: {response.retrieval_query}")
    print(response.answer)
    print("\nRetrieved documents:")
    for index, document in enumerate(response.documents, start=1):
        description = (
            document.description or document.description_preview or "<missing>"
        )
        if len(description) > DEFAULT_DESCRIPTION_PREVIEW_CHARS:
            description = (
                f"{description[:DEFAULT_DESCRIPTION_PREVIEW_CHARS].rstrip()}..."
            )
        print(
            f"{index}. score={document.score:.4f} | "
            f"{document.title} | {document.job_category} | job_id={document.job_id}"
        )
        print(f"   description: {description}")


def main() -> None:
    """Run one single-turn RAG request."""
    args = parse_args()
    retriever = QdrantRetriever(
        collection_name=args.collection,
        model_name=args.embedding_model,
        qdrant_url=args.qdrant_url,
        local_files_only=args.local_files_only,
    )
    generator = build_generator(args)
    chain = JobMarketRagChain(
        retriever=retriever,
        generator=generator,
        generation_config=GenerationConfig(max_tokens=args.max_tokens),
        max_context_documents=args.max_context_documents,
        max_preview_chars=args.max_preview_chars,
    )

    turn_queries = [args.query, *args.follow_up_query]
    for turn_number, query in enumerate(turn_queries, start=1):
        response = chain.answer(query=query, top_k=args.top_k)
        print_turn_response(turn_number, query, response)


if __name__ == "__main__":
    main()
