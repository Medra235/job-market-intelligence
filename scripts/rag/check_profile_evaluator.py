"""Run a manual profile evaluation check.

Usage example:
    uv run python scripts/rag/check_profile_evaluator.py --profile "Python, SQL, pandas, basic AWS" --target-role "Data Engineer"
"""

import argparse
from job_market_intelligence.config.settings import settings
from job_market_intelligence.models.profile_classifier import JobCategoryClassifier
from job_market_intelligence.rag.generator import GenerationConfig, OllamaGenerator
from job_market_intelligence.rag.profile_evaluator import ProfileEvaluator
from job_market_intelligence.rag.retriever import QdrantRetriever


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manual profile evaluation check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Profile, CV, or skills text.")
    parser.add_argument(
        "--target-role",
        default=None,
        help=(
            "Optional user-declared target role used to guide retrieval and "
            "evaluation; it is not the classifier output."
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
        "--num-predict",
        type=int,
        default=1024,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Add the classical job-category prediction as an auxiliary signal.",
    )
    parser.add_argument(
        "--classifier-model-type",
        default=settings.classifier_model_type,
        help="Local model folder under models/ used for classification.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one profile evaluation request."""
    args = parse_args()
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
    category_classifier = (
        JobCategoryClassifier(model_type=args.classifier_model_type)
        if args.use_classifier
        else None
    )
    evaluator = ProfileEvaluator(
        retriever=retriever,
        generator=generator,
        category_classifier=category_classifier,
        generation_config=GenerationConfig(num_predict=args.num_predict),
        max_context_documents=args.max_context_documents,
        max_preview_chars=args.max_preview_chars,
    )
    response = evaluator.evaluate(
        profile_text=args.profile,
        target_role=args.target_role,
        top_k=args.top_k,
    )

    print(response.evaluation)
    if response.category_prediction:
        confidence = (
            f" | confidence={response.category_prediction.confidence:.4f}"
            if response.category_prediction.confidence is not None
            else ""
        )
        print(
            "\nClassifier signal: "
            f"{response.category_prediction.label} "
            f"({response.category_prediction.model_type})"
            f"{confidence}"
        )
    print("\nRetrieved documents:")
    for index, document in enumerate(response.documents, start=1):
        print(
            f"{index}. score={document.score:.4f} | "
            f"{document.title} | {document.job_category} | job_id={document.job_id}"
        )


if __name__ == "__main__":
    main()
