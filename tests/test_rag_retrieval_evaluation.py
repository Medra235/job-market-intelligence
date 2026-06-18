"""Tests for the RAG retrieval evaluation helpers."""

from pathlib import Path

import pytest

from job_market_intelligence.rag.retriever import RetrievedDocument
from scripts.rag.evaluate_retrieval import (
    GoldenQuery,
    calculate_metrics,
    evaluate_document_relevance,
    load_golden_set,
)


def make_document(
    title: str,
    category: str,
    description: str,
    score: float = 0.7,
) -> RetrievedDocument:
    """Build a retrieved document for evaluation tests."""
    return RetrievedDocument(
        job_id=1,
        title=title,
        job_category=category,
        description=description,
        description_preview=description[:80],
        score=score,
        payload={"job_id": 1},
    )


def test_document_relevance_matches_expected_category() -> None:
    """A matching category should be enough proxy evidence."""
    document = make_document(
        title="Analytics Engineer",
        category="Data Analytics",
        description="Build reporting models for stakeholders.",
    )

    relevance = evaluate_document_relevance(
        document=document,
        expected_categories=["Data Analytics"],
        expected_keywords=["Power BI"],
    )

    assert relevance.is_relevant is True
    assert relevance.category_match is True
    assert relevance.matched_keywords == []


def test_document_relevance_matches_expected_keywords() -> None:
    """Keyword evidence in the description should mark the document relevant."""
    document = make_document(
        title="Platform Engineer",
        category="Software Engineer",
        description="Own Kubernetes, Docker, Terraform and CI/CD automation.",
    )

    relevance = evaluate_document_relevance(
        document=document,
        expected_categories=["DevOps/SRE"],
        expected_keywords=["Kubernetes", "Terraform"],
    )

    assert relevance.is_relevant is True
    assert relevance.category_match is False
    assert relevance.matched_keywords == ["Kubernetes", "Terraform"]


def test_calculate_metrics_uses_top_k_ranking() -> None:
    """Precision@k, Hit@k, and MRR@k should use relevant ranks."""
    golden_query = GoldenQuery(
        query_id="data_engineering",
        query="What should I learn for Data Engineering?",
        expected_categories=["Data Engineering"],
        expected_keywords=["Airflow"],
    )
    documents = [
        make_document(
            title="Frontend Developer",
            category="Frontend",
            description="React and TypeScript role.",
        ),
        make_document(
            title="Data Engineer",
            category="Data Engineering",
            description="Python, SQL and Airflow pipelines.",
        ),
        make_document(
            title="Backend Developer",
            category="Backend",
            description="Build REST APIs.",
        ),
    ]

    evaluation = calculate_metrics(
        golden_query=golden_query,
        retrieved_documents=documents,
        top_k=3,
    )

    assert evaluation.relevant_count == 1
    assert evaluation.precision_at_k == pytest.approx(1 / 3)
    assert evaluation.hit_at_k == 1
    assert evaluation.mrr_at_k == pytest.approx(1 / 2)


def test_load_golden_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Golden-set query ids must be unique."""
    golden_set = tmp_path / "golden_set.yaml"
    golden_set.write_text(
        """
metadata:
  default_top_k: 5
queries:
  - id: duplicate
    query: "First query"
  - id: duplicate
    query: "Second query"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate golden-set query id"):
        load_golden_set(golden_set)
