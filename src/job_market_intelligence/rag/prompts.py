"""Prompt templates for the job-market RAG assistant."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from job_market_intelligence.config.settings import settings
from job_market_intelligence.models.profile_classifier import CategoryPrediction
from job_market_intelligence.rag.retriever import RetrievedDocument

HIGH_CLASSIFIER_CONFIDENCE = 0.70
MEDIUM_CLASSIFIER_CONFIDENCE = 0.50

COMMON_GROUNDING_RULES = """\
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
generic market commentary.

For senior or lead roles, mention leadership only when the retrieved context
actually supports it. For junior or upskilling questions, prioritize the core
technical stack first.

Do not promise employability or guarantee outcomes.
"""

CHATBOT_SYSTEM_PROMPT = (
    """\
You are a conversational career assistant for the technology job market.

"""
    + COMMON_GROUNDING_RULES
    + """

Use a flexible conversational format. Match the structure to the user's
question instead of always producing a full report. For simple follow-up
questions, answer directly in one or two short paragraphs or a short bullet
list. Use headings only when they make the answer easier to scan.

When the user asks what to learn, rank the skills by priority and keep the
answer focused on the retrieved evidence. Keep the list short: usually 3 or 4
items are enough.

When referencing specific retrieved postings, cite them with bracketed source
numbers such as [1] or [2].

When the user asks for a plan, comparison, or roadmap, structure the answer
around that requested format. Avoid repeating a full skills/gaps/recommendations
template unless the user explicitly asks for a full assessment.

If the retrieved context is not enough to answer safely, say so clearly. Do not
turn one-off evidence into a broad trend. If a tool appears only in part of the
retrieved context, mention it cautiously.

Prefer concise complete answers over long reports. If the answer needs to be
shortened, keep the direct recommendation and evidence caveat, then omit
secondary detail.
"""
)

PROFILE_EVALUATION_SYSTEM_PROMPT = (
    """\
You generate structured profile evaluation reports for technology career
orientation. This is not an open-ended chatbot flow.

"""
    + COMMON_GROUNDING_RULES
    + """

When evaluating a user profile, compare the profile only against the retrieved
job postings. Be constructive but honest. Do not infer credentials, seniority,
years of experience, degrees, or certifications that are not present in the
profile text.

The main target user is a junior, self-taught, bootcamp, or career-transition
profile. Do not treat every retrieved requirement as mandatory for entry-level
readiness. If retrieved postings look advanced, use them as aspirational market
evidence for roadmap planning.

The classifier category belongs to a closed taxonomy from this project. Do not
treat it as an absolute truth or as a score for the user's profile.

Use classifier certainty as follows:
- alta: use the predicted category as the main evaluation frame.
- media: use it as guidance and contrast it with the user's explicit skills and
  retrieved postings.
- baja: present it as tentative and do not force the whole evaluation toward
  that category.

In "skills detectadas en el perfil", include only skills, tools, or concepts
that appear explicitly in the user's text. Do not add implicit or assumed
skills. If a skill is a reasonable inference, treat it as a gap, related area,
or recommendation.

Do not classify the seniority of the user or retrieved postings unless the title
or description states it explicitly. If you infer it from technical
requirements, use wording like "ofertas con requisitos más avanzados".
"""
)

NO_CONTEXT_NOTICE = """\
No retrieved job postings were provided. Answer only with general guidance and
state that the response is not grounded in the local job-posting corpus.
"""


def estimate_classifier_certainty(confidence: float | None) -> str:
    """Convert classifier confidence into a qualitative certainty band."""
    if confidence is None:
        return "no disponible"
    if confidence >= HIGH_CLASSIFIER_CONFIDENCE:
        return "alta"
    if confidence >= MEDIUM_CLASSIFIER_CONFIDENCE:
        return "media"
    return "baja"


def format_conversation_history(
    messages: Sequence[BaseMessage],
    max_messages: int = settings.max_history_messages,
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
        content = (
            message.content.strip()
            if isinstance(message.content, str)
            else str(message.content).strip()
        )
        if not content:
            continue
        role = role_labels.get(message.type, message.type)
        formatted_messages.append(f"{role}: {content}")
    return "\n".join(formatted_messages)


def format_retrieved_documents(
    documents: Sequence[RetrievedDocument],
    max_documents: int = settings.max_context_documents,
    max_preview_chars: int = settings.max_preview_chars,
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
                    "description:",
                    description,
                ]
            )
        )
    return "\n\n".join(formatted_docs)


def build_profile_evaluation_prompt(
    profile_text: str,
    documents: Sequence[RetrievedDocument],
    target_role: str | None = None,
    category_prediction: CategoryPrediction | None = None,
    classifier_certainty: str | None = None,
    max_documents: int = settings.max_context_documents,
    max_preview_chars: int = settings.max_preview_chars,
) -> str:
    """Build a grounded prompt for evaluating a user's profile against the market."""
    profile_text = profile_text.strip()
    if not profile_text:
        raise ValueError("Profile text cannot be empty.")

    target_section = (
        f"Rol objetivo declarado por el usuario:\n{target_role.strip()}\n\n"
        if target_role and target_role.strip()
        else ""
    )
    category_section = ""
    if category_prediction:
        certainty = classifier_certainty or estimate_classifier_certainty(
            category_prediction.confidence
        )
        category_section = (
            "Resultado del clasificador ML:\n"
            f"categoría probable: {category_prediction.label}\n"
            f"certeza estimada: {certainty}\n\n"
        )

    context = format_retrieved_documents(
        documents,
        max_documents=max_documents,
        max_preview_chars=max_preview_chars,
    )

    return f"""\
Perfil del usuario:

{profile_text}

{target_section}{category_section}Ofertas recuperadas como referencia de mercado:

{context}

Evalúa el perfil comparándolo con las ofertas recuperadas. Usa exactamente esta
estructura Markdown:

- **Resumen del encaje**:
  2-3 frases breves sobre el encaje general.

- **Skills detectadas en el perfil**:
  - Agrupa las skills explícitas por tipo cuando sea natural, por ejemplo
    "AWS Core", "Herramientas de IaC", "Lenguajes", "Frameworks" o
    "Visualización".

- **Fundamentos de entrada ya cubiertos**:
  - Resume fundamentos ya cubiertos; agrupa skills relacionadas si ayuda.

- **Siguientes pasos prácticos**:
  1. Propón un paso práctico concreto.
  2. Propón un paso práctico concreto.
  3. Propón un paso práctico concreto.

- **Gaps avanzados o aspiracionales**:
  - Agrupa gaps relacionados cuando sea natural.

- **Recomendaciones priorizadas**:
  1. Da una recomendación priorizada y accionable.
  2. Da una recomendación priorizada y accionable.
  3. Da una recomendación priorizada y accionable.
"""


def build_rag_prompt(
    query: str,
    documents: Sequence[RetrievedDocument],
    conversation_history: Sequence[BaseMessage] | None = None,
    max_documents: int = settings.max_context_documents,
    max_history_messages: int = settings.max_history_messages,
    max_preview_chars: int = settings.max_preview_chars,
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
        f"Historial de conversación previo:\n\n{history}\n\n" if history else ""
    )
    return f"""\
Contexto recuperado de ofertas de empleo:

{context}

{history_section}Pregunta del usuario:
{query}

Responde usando el contexto anterior como evidencia principal.
"""
