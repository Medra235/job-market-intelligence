"""Tests for the single-turn RAG chain."""

from job_market_intelligence.rag.rag_chain import JobMarketRagChain
from job_market_intelligence.rag.memory import ConversationMemory
from job_market_intelligence.rag.retriever import RetrievedDocument


class FakeRetriever:
    """Fake retriever returning deterministic documents."""

    def __init__(self) -> None:
        self.last_call = None
        self.calls = []

    def retrieve(self, query: str, top_k: int = 5, score_threshold=None):
        """Return one retrieved document and record call arguments."""
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        self.calls.append(self.last_call)
        return [
            RetrievedDocument(
                job_id=1,
                title="Data Engineer",
                job_category="Data Engineering",
                description="Python and Airflow role with SQL pipelines.",
                description_preview="Python and Airflow role.",
                score=0.7,
                payload={"job_id": 1},
            )
        ]


class FakeGenerator:
    """Fake generator returning a deterministic answer."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.last_call = None
        self.prompts = []
        self.answers = answers or ["Aprende Python, SQL y Airflow."]

    def generate(self, prompt: str, system_prompt=None, config=None) -> str:
        """Return an answer and record generation inputs."""
        self.last_call = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "config": config,
        }
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("FakeGenerator exhausted answers.")
        return self.answers.pop(0)


class FakeQueryRewriter:
    """Fake query rewriter returning deterministic retrieval queries."""

    def __init__(self, rewritten_queries: list[str] | None = None) -> None:
        self.calls = []
        self.rewritten_queries = rewritten_queries or []

    def rewrite(self, query: str, conversation_history) -> str:
        """Record the rewrite inputs and return the next configured query."""
        self.calls.append(
            {
                "query": query,
                "conversation_history": list(conversation_history),
            }
        )
        if self.rewritten_queries:
            return self.rewritten_queries.pop(0)
        return query


def test_rag_chain_retrieves_builds_prompt_and_generates_answer() -> None:
    """The chain should connect retriever, prompt builder, and generator."""
    retriever = FakeRetriever()
    generator = FakeGenerator()
    chain = JobMarketRagChain(
        retriever=retriever,
        generator=generator,
        max_context_documents=1,
        max_preview_chars=100,
    )

    response = chain.answer(
        query="What should I learn for Data Engineering?",
        top_k=3,
        score_threshold=0.4,
    )

    assert retriever.last_call == {
        "query": "What should I learn for Data Engineering?",
        "top_k": 3,
        "score_threshold": 0.4,
    }
    assert "Data Engineer" in generator.last_call["prompt"]
    assert "What should I learn for Data Engineering?" in generator.last_call["prompt"]
    assert response.answer == "Aprende Python, SQL y Airflow."
    assert response.query == "What should I learn for Data Engineering?"
    assert response.retrieval_query == "What should I learn for Data Engineering?"
    assert response.documents[0].title == "Data Engineer"


def test_rag_chain_includes_conversation_memory_on_follow_up() -> None:
    """A second answer should reuse earlier turns as conversational context."""
    retriever = FakeRetriever()
    query_rewriter = FakeQueryRewriter(
        rewritten_queries=[
            "What should I learn for Data Engineering?",
            (
                "What analytics and BI skills complement Data Engineering "
                "roles?"
            ),
        ]
    )
    generator = FakeGenerator(
        answers=[
            "Aprende Python, SQL y Airflow.",
            "También conviene reforzar análisis y reporting.",
        ]
    )
    chain = JobMarketRagChain(
        retriever=retriever,
        generator=generator,
        memory=ConversationMemory(max_messages=8),
        query_rewriter=query_rewriter,
        max_context_documents=1,
        max_preview_chars=100,
    )

    first_response = chain.answer(query="What should I learn for Data Engineering?")
    second_response = chain.answer(query="And for the analytics side?")

    assert first_response.answer == "Aprende Python, SQL y Airflow."
    assert second_response.answer == "También conviene reforzar análisis y reporting."
    assert second_response.query == "And for the analytics side?"
    assert (
        second_response.retrieval_query
        == "What analytics and BI skills complement Data Engineering roles?"
    )
    assert retriever.calls[1]["query"] == second_response.retrieval_query
    assert "Previous conversation history" in generator.prompts[1]
    assert "What should I learn for Data Engineering?" in generator.prompts[1]
    assert "Aprende Python, SQL y Airflow." in generator.prompts[1]
    assert "And for the analytics side?" in generator.prompts[1]


def test_rag_chain_rejects_empty_query() -> None:
    """Empty queries should fail before retrieval."""
    chain = JobMarketRagChain(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    try:
        chain.answer(query=" ")
    except ValueError as exc:
        assert "Query cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query.")
