"""Tests for RAG prompt construction."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.prompts import (
    NO_CONTEXT_NOTICE,
    SYSTEM_PROMPT,
    build_rag_prompt,
    format_conversation_history,
    format_retrieved_documents,
)
from src.rag.retriever import RetrievedDocument


def sample_document() -> RetrievedDocument:
    """Return one realistic retrieved document for prompt tests."""
    return RetrievedDocument(
        job_id=123,
        title="Data Engineer",
        job_category="Data Engineering",
        description="Python, SQL, Airflow and AWS are required for this role.",
        description_preview="Short preview that should not be used when description exists.",
        score=0.71234,
        payload={"job_id": 123},
    )


def test_system_prompt_sets_spanish_grounded_behavior() -> None:
    """The system prompt should define language and grounding rules."""
    assert "Answer in Spanish" in SYSTEM_PROMPT
    assert "retrieved job postings" in SYSTEM_PROMPT
    assert "Do not invent" in SYSTEM_PROMPT
    assert "Do not promise employability" in SYSTEM_PROMPT
    assert "Be practical, specific, and complete" in SYSTEM_PROMPT
    assert "skills relevantes, ordenadas por prioridad" in SYSTEM_PROMPT
    assert "Limit the skills section to the 4 most important items" in (
        SYSTEM_PROMPT.replace("\n", " ")
    )
    assert "Be precise about evidence strength" in SYSTEM_PROMPT
    assert "not as a broad market trend" in SYSTEM_PROMPT.replace("\n", " ")


def test_format_retrieved_documents_includes_payload_fields() -> None:
    """Retrieved documents should be formatted as compact evidence."""
    formatted = format_retrieved_documents([sample_document()])

    assert "[1] score=0.7123" in formatted
    assert "title: Data Engineer" in formatted
    assert "category: Data Engineering" in formatted
    assert "job_id: 123" in formatted
    assert "Python, SQL, Airflow and AWS are required for this role." in formatted
    assert "Short preview that should not be used" not in formatted


def test_format_retrieved_documents_truncates_full_description() -> None:
    """Long descriptions should be trimmed before entering the LLM prompt."""
    document = RetrievedDocument(
        job_id=123,
        title="Data Engineer",
        job_category="Data Engineering",
        description="A" * 20,
        description_preview="short preview",
        score=0.7,
        payload={"job_id": 123},
    )

    formatted = format_retrieved_documents([document], max_preview_chars=10)

    assert "AAAAAAAAAA..." in formatted
    assert "short preview" not in formatted


def test_format_retrieved_documents_falls_back_to_preview() -> None:
    """Older payloads without description should still be usable."""
    document = RetrievedDocument(
        job_id=321,
        title="ML Engineer",
        job_category="ML",
        description="",
        description_preview="Model training, deployment and monitoring.",
        score=0.9,
        payload={"job_id": 321},
    )

    formatted = format_retrieved_documents([document], max_preview_chars=100)

    assert "Model training, deployment and monitoring." in formatted


def test_format_retrieved_documents_handles_empty_context() -> None:
    """Empty context should be explicit instead of silently omitted."""
    assert format_retrieved_documents([]) == NO_CONTEXT_NOTICE


def test_format_conversation_history_includes_recent_messages() -> None:
    """Conversation history should be formatted as readable dialogue."""
    formatted = format_conversation_history(
        [
            HumanMessage(content="What should I learn for Data Engineering?"),
            AIMessage(content="Python and SQL are core skills."),
        ]
    )

    assert "usuario: What should I learn for Data Engineering?" in formatted
    assert "asistente: Python and SQL are core skills." in formatted


def test_build_rag_prompt_includes_conversation_history() -> None:
    """The final prompt should include prior dialogue when available."""
    prompt = build_rag_prompt(
        query="And for the analytics side?",
        documents=[sample_document()],
        conversation_history=[
            HumanMessage(content="What should I learn for Data Engineering?"),
            AIMessage(content="Python and SQL are core skills."),
        ],
    )

    assert "Historial de conversación previo" in prompt
    assert "usuario: What should I learn for Data Engineering?" in prompt
    assert "asistente: Python and SQL are core skills." in prompt
    assert "And for the analytics side?" in prompt


def test_build_rag_prompt_includes_context_and_query() -> None:
    """The final prompt should combine context and the user question."""
    prompt = build_rag_prompt(
        query="What should I learn for Data Engineering?",
        documents=[sample_document()],
    )

    assert "Contexto recuperado de ofertas de empleo" in prompt
    assert "Data Engineer" in prompt
    assert "What should I learn for Data Engineering?" in prompt
    assert "evidencia principal" in prompt
    assert "3 recomendaciones accionables, ordenadas por prioridad" in prompt
    assert "No conviertas una evidencia puntual en una tendencia general" in prompt
    assert "Mantén cada punto breve" in prompt


@pytest.mark.parametrize("query", ["", "   "])
def test_build_rag_prompt_rejects_empty_query(query: str) -> None:
    """Empty user queries should fail early."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        build_rag_prompt(query=query, documents=[sample_document()])
