"""Tests for API service factories."""

from job_market_intelligence.api import services
from job_market_intelligence.config.settings import Settings
from job_market_intelligence.rag.generator import GenerationConfig
from job_market_intelligence.rag.profile_evaluator import ProfileEvaluator
from job_market_intelligence.rag.rag_chain import JobMarketRagChain


class FakeGenerator:
    """Small stand-in for the llama.cpp generator."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, system_prompt=None, config=None) -> str:
        _ = (prompt, system_prompt, config)
        return "generated"


class FakeRetriever:
    """Small stand-in for the Qdrant retriever."""

    def __init__(
        self,
        collection_name: str,
        model_name: str,
        qdrant_url: str,
        local_files_only: bool,
    ) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self.qdrant_url = qdrant_url
        self.local_files_only = local_files_only

    def retrieve(self, query: str, top_k: int = 5, score_threshold=None):
        _ = (query, top_k, score_threshold)
        return []


class FakeClassifier:
    """Small stand-in for the persisted profile classifier."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type


def test_build_llm_generator_uses_runtime_settings(monkeypatch) -> None:
    """The LLM factory should pass configured llama.cpp settings."""
    monkeypatch.setattr(services, "LlamaCppGenerator", FakeGenerator)
    settings = Settings(
        llamacpp_model="demo-model",
        llamacpp_url="http://llamacpp:8080",
        llamacpp_timeout_seconds=42.0,
    )

    generator = services.build_llm_generator(settings)

    assert isinstance(generator, FakeGenerator)
    assert generator.model_name == "demo-model"
    assert generator.base_url == "http://llamacpp:8080"
    assert generator.timeout_seconds == 42.0


def test_build_qdrant_retriever_uses_runtime_settings(monkeypatch) -> None:
    """The retriever factory should pass configured Qdrant and embedding settings."""
    monkeypatch.setattr(services, "QdrantRetriever", FakeRetriever)
    settings = Settings(
        qdrant_collection="demo_collection",
        embedding_model="/models/embeddings/demo",
        qdrant_url="http://qdrant:6333",
        embedding_local_files_only=True,
    )

    retriever = services.build_qdrant_retriever(settings)

    assert isinstance(retriever, FakeRetriever)
    assert retriever.collection_name == "demo_collection"
    assert retriever.model_name == "/models/embeddings/demo"
    assert retriever.qdrant_url == "http://qdrant:6333"
    assert retriever.local_files_only is True


def test_build_profile_evaluator_wires_shared_dependencies(monkeypatch) -> None:
    """The profile factory should compose retriever, generator, and classifier."""
    monkeypatch.setattr(services, "QdrantRetriever", FakeRetriever)
    monkeypatch.setattr(services, "LlamaCppGenerator", FakeGenerator)
    monkeypatch.setattr(services, "JobCategoryClassifier", FakeClassifier)
    settings = Settings(
        classifier_model_type="random_forest",
        profile_max_tokens=256,
        max_context_documents=4,
        max_preview_chars=500,
    )

    evaluator = services.build_profile_evaluator(settings)

    assert isinstance(evaluator, ProfileEvaluator)
    assert isinstance(evaluator.retriever, FakeRetriever)
    assert isinstance(evaluator.generator, FakeGenerator)
    assert isinstance(evaluator.category_classifier, FakeClassifier)
    assert evaluator.category_classifier.model_type == "random_forest"
    assert evaluator.generation_config == GenerationConfig(max_tokens=256)
    assert evaluator.max_context_documents == 4
    assert evaluator.max_preview_chars == 500


def test_build_chat_rag_chain_wires_shared_dependencies(monkeypatch) -> None:
    """The chat factory should compose retriever and generator."""
    monkeypatch.setattr(services, "QdrantRetriever", FakeRetriever)
    monkeypatch.setattr(services, "LlamaCppGenerator", FakeGenerator)
    settings = Settings(max_context_documents=6, max_preview_chars=350)

    chain = services.build_chat_rag_chain(settings)

    assert isinstance(chain, JobMarketRagChain)
    assert isinstance(chain.retriever, FakeRetriever)
    assert isinstance(chain.generator, FakeGenerator)
    assert chain.max_context_documents == 6
    assert chain.max_preview_chars == 350
