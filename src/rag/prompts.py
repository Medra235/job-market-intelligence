"""Prompt templates for the job-market RAG assistant."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from src.rag.retriever import RetrievedDocument

DEFAULT_MAX_CONTEXT_DOCUMENTS = 5
DEFAULT_MAX_HISTORY_MESSAGES = 8
DEFAULT_MAX_PREVIEW_CHARS = 700

SYSTEM_PROMPT = """\
You are a career assistant for the technology job market.

Answer in Spanish. Keep technical skill names in English when they are normally
used that way in job postings, such as Python, SQL, Airflow, AWS, React, Docker,
Kubernetes, or MLOps.

Use the retrieved job postings as your main evidence. Do not invent market facts,
salary ranges, hiring probabilities, or requirements that are not supported by
the provided context. If the retrieved context is weak, mixed, or insufficient,
say so briefly and explain what extra information would help.

Be precise about evidence strength. If a tool or requirement appears in only one
or a few retrieved postings, describe it as specific evidence, not as a broad
market trend. Use wording like "aparece en algunas ofertas" or "aparece en una
oferta recuperada" instead of absolute claims.

Be practical, specific, and complete. Prefer concrete skills and priorities over
generic market commentary. When the user asks what to learn, rank the skills by
priority and keep the answer focused on the retrieved evidence.

When useful, structure the answer with:
- resumen
- skills relevantes, ordenadas por prioridad
- gaps o riesgos
- recomendaciones priorizadas

Keep each bullet short and actionable. Limit the skills section to the 4 most
important items and the recommendations section to 3 items.

For senior or lead roles, mention leadership only when the retrieved context
actually supports it. For junior or upskilling questions, prioritize the core
technical stack first.

Do not promise employability or guarantee outcomes.
"""

NO_CONTEXT_NOTICE = """\
No retrieved job postings were provided. Answer only with general guidance and
state that the response is not grounded in the local job-posting corpus.
"""


def format_conversation_history(
    messages: Sequence[BaseMessage],
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
) -> str:
    """Format recent chat messages as compact conversational context."""
    if max_messages <= 0:
        raise ValueError("max_messages must be greater than 0.")
    if not messages:
        return ""

    role_labels = {
        "human": "usuario",
        "ai": "asistente",
        "system": "sistema",
    }
    formatted_messages = []
    for message in messages[-max_messages:]:
        content = message.content.strip() if isinstance(message.content, str) else str(
            message.content
        ).strip()
        if not content:
            continue
        role = role_labels.get(message.type, message.type)
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def format_retrieved_documents(
    documents: Sequence[RetrievedDocument],
    max_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
    max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
) -> str:
    """Format retrieved documents as compact context for the LLM."""
    if not documents:
        return NO_CONTEXT_NOTICE
    if max_documents <= 0:
        raise ValueError("max_documents must be greater than 0.")
    if max_preview_chars <= 0:
        raise ValueError("max_preview_chars must be greater than 0.")

    formatted_docs = []
    for index, document in enumerate(documents[:max_documents], start=1):
        description = document.description or document.description_preview
        description = description or "<missing description>"
        if len(description) > max_preview_chars:
            description = f"{description[:max_preview_chars].rstrip()}..."
        formatted_docs.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"title: {document.title or '<missing>'}",
                    f"category: {document.job_category or '<missing>'}",
                    f"job_id: {document.job_id or '<missing>'}",
                    "description:",
                    description,
                ]
            )
        )
    return "\n\n".join(formatted_docs)


def build_rag_prompt(
    query: str,
    documents: Sequence[RetrievedDocument],
    conversation_history: Sequence[BaseMessage] | None = None,
    max_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
) -> str:
    """Build the user prompt that combines retrieved context and the query."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    context = format_retrieved_documents(
        documents,
        max_documents=max_documents,
        max_preview_chars=max_preview_chars,
    )
    history = format_conversation_history(
        conversation_history or [],
        max_messages=max_history_messages,
    )
    history_section = (
        f"Historial de conversación previo:\n\n{history}\n\n"
        if history
        else ""
    )
    return f"""\
Contexto recuperado de ofertas de empleo:

{context}

{history_section}Pregunta del usuario:
{query}

Responde usando el contexto anterior como evidencia principal. Si la pregunta
pide qué aprender o cómo mejorar el perfil, prioriza las skills técnicas más
repetidas o más exigidas en el contexto, explica por qué importan y termina con
3 recomendaciones accionables, ordenadas por prioridad. Si el contexto no es
suficiente para responder con seguridad, indicalo claramente.

No conviertas una evidencia puntual en una tendencia general. Si una herramienta
solo aparece en una parte del contexto recuperado, dilo con cautela.

Mantén cada punto breve. Si necesitas recortar, recorta primero la explicación
de contexto general antes que las skills y las recomendaciones.
"""
