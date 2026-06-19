"""Tests for the RAG manual review Markdown generator."""

import pytest

from job_market_intelligence.rag.evaluation import (
    GoldenQuery,
    build_review_markdown,
    select_queries,
)
from job_market_intelligence.rag.rag_chain import RagResponse
from job_market_intelligence.rag.retriever import RetrievedDocument


def make_query(query_id: str) -> GoldenQuery:
    """Build a minimal golden query."""
    return GoldenQuery(
        query_id=query_id,
        query=f"Question for {query_id}",
        expected_categories=[],
        expected_keywords=[],
    )


def make_response(query: str = "Question for data_engineering") -> RagResponse:
    """Build a deterministic RAG response."""
    document = RetrievedDocument(
        job_id=123,
        title="Data Engineer",
        job_category="Data Engineering",
        description="Python, SQL and Airflow pipelines.",
        description_preview="Python, SQL and Airflow pipelines.",
        score=0.82,
        payload={"job_id": 123},
    )
    return RagResponse(
        answer="Prioriza Python, SQL y Airflow.",
        query=query,
        retrieval_query=query,
        documents=[document],
        prompt="prompt",
    )


def test_select_queries_preserves_requested_order() -> None:
    """Explicit query ids should be selected in the requested order."""
    queries = [make_query("first"), make_query("second")]

    selected = select_queries(queries, requested_ids=["second", "first"])

    assert [query.query_id for query in selected] == ["second", "first"]


def test_select_queries_rejects_unknown_ids() -> None:
    """Unknown query ids should fail clearly."""
    queries = [make_query("known")]

    with pytest.raises(ValueError, match="Unknown golden-set query id"):
        select_queries(queries, requested_ids=["missing"])


def test_build_review_markdown_includes_answer_sources_and_score_slots() -> None:
    """The review Markdown should include generated content and manual fields."""
    query = make_query("data_engineering")
    response = make_response()

    markdown = build_review_markdown(
        cases=[(query, response)],
        top_k=5,
        llm_model="qwen3.5:9b",
    )

    assert "# RAG Manual Generation Review" in markdown
    assert "## data_engineering" in markdown
    assert "Prioriza Python, SQL y Airflow." in markdown
    assert "**Data Engineer**" in markdown
    assert "Python, SQL and Airflow pipelines." in markdown
    assert "### User Query" in markdown
    assert "### Query Used By RAG" not in markdown
    assert "- Context relevance: _/2" in markdown
    assert "- Faithfulness: _/2" in markdown
    assert "- Answer relevance: _/2" in markdown
    assert "Notes:" not in markdown


def test_build_review_markdown_shows_rag_query_only_when_different() -> None:
    """Standalone rewritten queries should be visible in the review file."""
    query = make_query("follow_up")
    response = make_response()
    response = RagResponse(
        answer=response.answer,
        query=query.query,
        retrieval_query="Standalone retrieval query",
        documents=response.documents,
        prompt=response.prompt,
    )

    markdown = build_review_markdown(
        cases=[(query, response)],
        top_k=5,
        llm_model="qwen3.5:9b",
    )

    assert "### User Query" in markdown
    assert "### Query Used By RAG" in markdown
    assert "Standalone retrieval query" in markdown
