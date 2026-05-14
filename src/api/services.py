"""Factory helpers for FastAPI runtime services."""

from src.config.settings import Settings
from src.models.profile_classifier import JobCategoryClassifier
from src.rag.generator import GenerationConfig, OllamaGenerator
from src.rag.profile_evaluator import ProfileEvaluator
from src.rag.retriever import QdrantRetriever


def build_profile_evaluator(settings: Settings) -> ProfileEvaluator:
    """Build the shared profile evaluator from API runtime settings."""
    retriever = QdrantRetriever(
        collection_name=settings.qdrant_collection,
        model_name=settings.embedding_model,
        qdrant_url=settings.qdrant_url,
        local_files_only=settings.embedding_local_files_only,
    )
    generator = OllamaGenerator(
        model_name=settings.ollama_model,
        base_url=settings.ollama_url,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    category_classifier = JobCategoryClassifier(
        model_type=settings.classifier_model_type,
    )
    return ProfileEvaluator(
        retriever=retriever,
        generator=generator,
        category_classifier=category_classifier,
        generation_config=GenerationConfig(num_predict=settings.profile_num_predict),
        max_context_documents=settings.max_context_documents,
        max_preview_chars=settings.max_preview_chars,
    )
