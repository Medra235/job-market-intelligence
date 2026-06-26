"""Factory helpers for FastAPI runtime services."""

from job_market_intelligence.config.settings import Settings
from job_market_intelligence.models.profile_classifier import JobCategoryClassifier
from job_market_intelligence.rag.generator import (
    GenerationConfig,
    LlamaCppGenerator,
    TextGenerator,
)
from job_market_intelligence.rag.profile_evaluator import ProfileEvaluator
from job_market_intelligence.rag.rag_chain import JobMarketRagChain
from job_market_intelligence.rag.retriever import QdrantRetriever


def build_llm_generator(settings: Settings) -> TextGenerator:
    """Build the local llama.cpp generator."""
    return LlamaCppGenerator(
        model_name=settings.llamacpp_model,
        base_url=settings.llamacpp_url,
        timeout_seconds=settings.llamacpp_timeout_seconds,
    )


def build_qdrant_retriever(settings: Settings) -> QdrantRetriever:
    """Build the Qdrant retriever from API runtime settings."""
    return QdrantRetriever(
        collection_name=settings.qdrant_collection,
        model_name=settings.embedding_model,
        qdrant_url=settings.qdrant_url,
        local_files_only=settings.embedding_local_files_only,
    )


def build_profile_evaluator(settings: Settings) -> ProfileEvaluator:
    """Build the shared profile evaluator from API runtime settings."""
    retriever = build_qdrant_retriever(settings)
    generator = build_llm_generator(settings)
    category_classifier = JobCategoryClassifier(
        model_type=settings.classifier_model_type,
    )
    return ProfileEvaluator(
        retriever=retriever,
        generator=generator,
        category_classifier=category_classifier,
        generation_config=GenerationConfig(max_tokens=settings.profile_max_tokens),
        max_context_documents=settings.max_context_documents,
        max_preview_chars=settings.max_preview_chars,
    )


def build_chat_rag_chain(settings: Settings) -> JobMarketRagChain:
    """Build the shared RAG chatbot dependencies from API runtime settings."""
    retriever = build_qdrant_retriever(settings)
    generator = build_llm_generator(settings)
    return JobMarketRagChain(
        retriever=retriever,
        generator=generator,
        max_context_documents=settings.max_context_documents,
        max_preview_chars=settings.max_preview_chars,
    )
