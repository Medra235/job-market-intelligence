"""Tests for conversational query rewriting."""

from langchain_core.chat_history import InMemoryChatMessageHistory

from job_market_intelligence.rag.query_rewriter import OllamaQueryRewriter


class FakeRewriteGenerator:
    """Fake generator for query rewriter tests."""

    def __init__(self, answer: str = "", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls = []

    def generate(self, prompt: str, system_prompt=None, config=None) -> str:
        """Return the configured answer or raise the configured error."""
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "config": config,
            }
        )
        if self.error:
            raise self.error
        return self.answer


def make_history():
    """Build one prior conversation turn."""
    history = InMemoryChatMessageHistory()
    history.add_user_message("I am interested in Machine Learning Engineer roles.")
    history.add_ai_message("Focus on Python, ML fundamentals, and production ML.")
    return history.messages


def test_query_rewriter_skips_llm_without_history() -> None:
    """Standalone first turns should use the original query directly."""
    generator = FakeRewriteGenerator(answer="rewritten")
    rewriter = OllamaQueryRewriter(generator=generator)

    rewritten_query = rewriter.rewrite(
        query="What skills should I learn for Data Engineering?",
        conversation_history=[],
    )

    assert rewritten_query == "What skills should I learn for Data Engineering?"
    assert generator.calls == []


def test_query_rewriter_returns_clean_standalone_query() -> None:
    """The rewriter should clean common labels from model output."""
    generator = FakeRewriteGenerator(
        answer=(
            "Standalone retrieval query: What deployment and MLOps skills "
            "are useful for Machine Learning Engineer roles?"
        )
    )
    rewriter = OllamaQueryRewriter(generator=generator)

    rewritten_query = rewriter.rewrite(
        query="What should I learn for deployment?",
        conversation_history=make_history(),
    )

    assert (
        rewritten_query
        == (
            "What deployment and MLOps skills are useful for Machine Learning "
            "Engineer roles?"
        )
    )
    assert "Conversation history" in generator.calls[0]["prompt"]
    assert "What should I learn for deployment?" in generator.calls[0]["prompt"]


def test_query_rewriter_falls_back_to_original_query_on_generation_error() -> None:
    """Generation errors should not make the RAG chain unavailable."""
    generator = FakeRewriteGenerator(error=RuntimeError("Ollama unavailable"))
    rewriter = OllamaQueryRewriter(generator=generator)

    rewritten_query = rewriter.rewrite(
        query="What should I learn for deployment?",
        conversation_history=make_history(),
    )

    assert rewritten_query == "What should I learn for deployment?"
