"""Generate a Markdown file for manual RAG generation review.

Usage examples:
    uv run python scripts/rag/generate_manual_review.py
    uv run python scripts/rag/generate_manual_review.py --query-id backend_api_development
    uv run python scripts/rag/generate_manual_review.py --all
"""

import argparse
from pathlib import Path

from job_market_intelligence.config.settings import settings
from job_market_intelligence.rag.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    GoldenQuery,
    build_review_markdown,
    load_golden_set,
    select_queries,
)
from job_market_intelligence.rag.generator import (
    GenerationConfig,
    LlamaCppGenerator,
    TextGenerator,
)
from job_market_intelligence.rag.rag_chain import JobMarketRagChain, RagResponse
from job_market_intelligence.rag.retriever import QdrantRetriever

DEFAULT_OUTPUT_PATH = Path("reports/rag/manual_generation_review.md")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manual review generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
        help="Path to the YAML golden set.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Golden-set query id to include. Repeat the flag for multiple ids.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate review blocks for all golden-set queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected queries to run.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Retriever top_k.")
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
        default=settings.max_context_documents,
        help="Maximum retrieved documents included in the LLM prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1536,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Markdown output path.",
    )
    return parser.parse_args()


def build_generator(args: argparse.Namespace) -> TextGenerator:
    """Build the local llama.cpp generator for manual review."""
    model_name = get_model_name(args)
    return LlamaCppGenerator(
        model_name=model_name,
        base_url=args.llamacpp_url,
    )


def get_model_name(args: argparse.Namespace) -> str:
    """Return the effective llama.cpp model name."""
    return args.llm_model or settings.llamacpp_model


def build_rag_chain(args: argparse.Namespace) -> JobMarketRagChain:
    """Build the configured RAG chain used for manual review."""
    retriever = QdrantRetriever(
        collection_name=args.collection,
        model_name=args.embedding_model,
        qdrant_url=args.qdrant_url,
        local_files_only=args.local_files_only,
    )
    generator = build_generator(args)
    return JobMarketRagChain(
        retriever=retriever,
        generator=generator,
        generation_config=GenerationConfig(max_tokens=args.max_tokens),
        max_context_documents=args.max_context_documents,
        max_preview_chars=settings.max_preview_chars,
    )


def generate_review_cases(
    queries: list[GoldenQuery],
    chain: JobMarketRagChain,
    top_k: int,
) -> list[tuple[GoldenQuery, RagResponse]]:
    """Run the RAG chain for selected golden queries."""
    cases = []
    for query in queries:
        chain.clear_memory()
        for message in query.conversation_history:
            if message["role"] == "user":
                chain.memory.history.add_user_message(message["content"])
            else:
                chain.memory.history.add_ai_message(message["content"])
        response = chain.answer(query=query.query, top_k=top_k)
        cases.append((query, response))
    return cases


def main() -> None:
    """Generate a Markdown file with answers and sources for manual scoring."""
    args = parse_args()
    queries, default_top_k = load_golden_set(args.golden_set)
    selected_queries = select_queries(
        queries=queries,
        requested_ids=args.query_id,
        include_all=args.all,
        limit=args.limit,
    )
    top_k = args.top_k or default_top_k
    chain = build_rag_chain(args)
    cases = generate_review_cases(selected_queries, chain=chain, top_k=top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_review_markdown(
            cases,
            top_k=top_k,
            llm_model=get_model_name(args),
        ),
        encoding="utf-8",
    )
    print(f"Manual review written to: {args.output}")
    print(f"Generated review cases: {len(cases)}")


if __name__ == "__main__":
    main()
