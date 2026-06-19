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
    load_golden_set,
)
from job_market_intelligence.rag.generator import GenerationConfig, OllamaGenerator
from job_market_intelligence.rag.rag_chain import JobMarketRagChain, RagResponse
from job_market_intelligence.rag.retriever import QdrantRetriever, RetrievedDocument

DEFAULT_OUTPUT_PATH = Path("reports/rag/manual_generation_review.md")
DEFAULT_REVIEW_QUERY_IDS = [
    "data_engineering_learning_path",
    "devops_platform_skills",
    "ai_ml_engineer_skills",
    "backend_api_development",
    "follow_up_mlops_deployment",
]


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
        default=settings.ollama_model,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url,
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--ollama-url",
        default=settings.ollama_url,
        help="Ollama base URL.",
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
        "--num-predict",
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
        missing_ids = [query_id for query_id in requested_ids if query_id not in query_by_id]
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


def build_rag_chain(args: argparse.Namespace) -> JobMarketRagChain:
    """Build the current Ollama-backed RAG chain used for manual review."""
    retriever = QdrantRetriever(
        collection_name=args.collection,
        model_name=args.embedding_model,
        qdrant_url=args.qdrant_url,
        local_files_only=args.local_files_only,
    )
    generator = OllamaGenerator(
        model_name=args.llm_model,
        base_url=args.ollama_url,
    )
    return JobMarketRagChain(
        retriever=retriever,
        generator=generator,
        generation_config=GenerationConfig(num_predict=args.num_predict),
        max_context_documents=args.max_context_documents,
        max_preview_chars=settings.max_preview_chars,
    )


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


def generate_review_cases(
    queries: list[GoldenQuery],
    chain: JobMarketRagChain,
    top_k: int,
) -> list[tuple[GoldenQuery, RagResponse]]:
    """Run the RAG chain for selected golden queries."""
    cases = []
    for query in queries:
        chain.clear_memory()
        response = chain.answer(query=query.retrieval_query, top_k=top_k)
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
        build_review_markdown(cases, top_k=top_k, llm_model=args.llm_model),
        encoding="utf-8",
    )
    print(f"Manual review written to: {args.output}")
    print(f"Generated review cases: {len(cases)}")


if __name__ == "__main__":
    main()
