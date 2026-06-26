"""Prompt templates for the job-market RAG assistant."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from job_market_intelligence.config.settings import settings
from job_market_intelligence.models.profile_classifier import CategoryPrediction
from job_market_intelligence.rag.retriever import RetrievedDocument

HIGH_CLASSIFIER_CONFIDENCE = 0.70
MEDIUM_CLASSIFIER_CONFIDENCE = 0.50

COMMON_GROUNDING_RULES = """\
Answer in Spanish. Keep common technical terms in English when job postings use
them that way, such as Python, SQL, Airflow, AWS, React, Docker, Kubernetes, or
MLOps.

Use the retrieved job postings as the main evidence. Do not invent market facts,
salary ranges, hiring probabilities, or unsupported requirements. If the context
is weak, mixed, or insufficient, say so and explain what information is missing.

Be precise about evidence strength: one or a few postings are specific evidence,
not a broad trend. Be practical and prioritize concrete skills over generic
market commentary.

Do not promise employability or guarantee outcomes.
"""

CHATBOT_SYSTEM_PROMPT = (
    """\
You are a conversational career assistant for the technology job market.

"""
    + COMMON_GROUNDING_RULES
    + """

Match the structure to the user's question. Use short paragraphs or bullets for
simple follow-ups, and headings only when they help scanning.

When the user asks what to learn, rank 3 or 4 skills by priority and ground them
in the retrieved evidence. When referencing specific postings, cite source
numbers such as [1] or [2].

For plans, comparisons, or roadmaps, follow the requested format. Avoid a full
skills/gaps/recommendations report unless the user asks for one.
"""
)

QUERY_REWRITER_SYSTEM_PROMPT = """\
You rewrite conversational user questions into concise standalone search queries
for retrieving relevant job postings.

Rules:
- Do not answer the user.
- If the latest question is already standalone or changes topic, return it
  unchanged.
- If the latest question depends on previous context, add only the minimum
  missing context needed for retrieval.
- Preserve technical terms, role names, and seniority when they matter.
- Return only the rewritten search query, without explanations or formatting.
"""

PROFILE_EVALUATION_SYSTEM_PROMPT = (
    """\
You generate structured profile evaluation reports for technology career
orientation. This is not an open-ended chatbot flow.

"""
    + COMMON_GROUNDING_RULES
    + """

Compare the profile only against the retrieved postings. Be constructive and
honest. Do not infer the user's credentials, seniority, years of experience,
degrees, or certifications.

Assume the main target user may be junior, self-taught, bootcamp-trained, or in
career transition. Do not treat every retrieved requirement as mandatory for
entry-level readiness. If postings look advanced, use them as aspirational
roadmap evidence.

The classifier category belongs to this project's closed taxonomy. Treat it as
guidance, not as an absolute truth or profile score.

Use classifier certainty as follows:
- alta: use the predicted category as the main evaluation frame.
- media: use it as guidance and contrast it with the user's explicit skills and
  retrieved postings.
- baja: present it as tentative and do not force the whole evaluation toward
  that category.

In "skills detectadas en el perfil", include only skills, tools, or concepts
explicitly mentioned by the user. Treat reasonable inferences as gaps, related
areas, or recommendations.

Avoid labeling retrieved postings by seniority unless the title or description
states it explicitly. If requirements suggest a higher level, use wording like
"ofertas con requisitos más avanzados".
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
        f"Target role declared by the user:\n{target_role.strip()}\n\n"
        if target_role and target_role.strip()
        else ""
    )
    category_section = ""
    if category_prediction:
        certainty = classifier_certainty or estimate_classifier_certainty(
            category_prediction.confidence
        )
        category_section = (
            "ML classifier result:\n"
            f"likely category: {category_prediction.label}\n"
            f"estimated certainty: {certainty}\n\n"
        )

    context = format_retrieved_documents(
        documents,
        max_documents=max_documents,
        max_preview_chars=max_preview_chars,
    )

    return f"""\
User profile:

{profile_text}

{target_section}{category_section}Retrieved postings used as market evidence:

{context}

Evaluate the profile against the retrieved postings. Answer in Spanish and use
exactly this Markdown structure:

Start with 2-3 short sentences summarizing the overall fit, without a heading.

- **Skills detectadas en el perfil**:
  - Group explicit skills by type when useful.

- **Fundamentos de entrada ya cubiertos**:
  - Summarize entry-level foundations already covered.

- **Gaps avanzados o aspiracionales**:
  - Group related gaps when useful.

- **Siguientes pasos priorizados**:
  1. Give one prioritized, actionable recommendation.
  2. Give one prioritized, actionable recommendation.
  3. Give one prioritized, actionable recommendation.
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
        f"Previous conversation history:\n\n{history}\n\n" if history else ""
    )
    return f"""\
Retrieved job-posting context:

{context}

{history_section}User question:
{query}

Answer in Spanish using the retrieved context as the main evidence.
"""


def build_query_rewrite_prompt(
    query: str,
    conversation_history: Sequence[BaseMessage],
    max_history_messages: int = settings.max_history_messages,
) -> str:
    """Build the prompt used to rewrite a follow-up into a retrieval query."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    history = format_conversation_history(
        conversation_history,
        max_messages=max_history_messages,
    )
    history_section = history or "<no previous conversation>"
    return f"""\
Conversation history:

{history_section}

Latest user question:
{query}

Standalone retrieval query:
"""
