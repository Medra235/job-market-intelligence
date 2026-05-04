"""Profile evaluation flow backed by the job-market RAG corpus."""

from dataclasses import dataclass

from src.rag.generator import GenerationConfig, OllamaGenerator
from src.rag.prompts import (
    DEFAULT_MAX_CONTEXT_DOCUMENTS,
    DEFAULT_MAX_PREVIEW_CHARS,
    SYSTEM_PROMPT,
    format_retrieved_documents,
)
from src.rag.retriever import QdrantRetriever, RetrievedDocument

PROFILE_EVALUATION_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

When evaluating a user profile, compare the profile only against the retrieved
job postings. Be constructive but honest. Do not infer credentials, seniority,
years of experience, degrees, or certifications that are not present in the
profile text.

The main target user is a junior, self-taught, bootcamp, or career-transition
profile. Do not treat every retrieved requirement as mandatory for entry-level
readiness. If retrieved postings look advanced, use them as aspirational market
evidence for roadmap planning.
"""
)


@dataclass(frozen=True)
class ProfileEvaluationResponse:
    """Response returned by the profile evaluator."""

    evaluation: str
    profile_text: str
    target_role: str | None
    documents: list[RetrievedDocument]
    prompt: str


def build_profile_evaluation_prompt(
    profile_text: str,
    documents: list[RetrievedDocument],
    target_role: str | None = None,
    max_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
    max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
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
    context = format_retrieved_documents(
        documents,
        max_documents=max_documents,
        max_preview_chars=max_preview_chars,
    )

    return f"""\
Perfil del usuario:

{profile_text}

{target_section}Ofertas recuperadas como referencia de mercado:

{context}

Evalúa el perfil comparándolo con las ofertas recuperadas. Responde en español
y usa esta estructura:

- resumen del encaje
- skills detectadas en el perfil
- fundamentos de entrada ya cubiertos
- siguientes pasos prácticos
- gaps avanzados o aspiracionales
- recomendaciones priorizadas

No clasifiques el seniority del usuario ni de las ofertas salvo que el título o
la descripción lo indiquen explícitamente. Si lo infieres por requisitos
técnicos, usa "ofertas con requisitos más avanzados".
"""


class ProfileEvaluator:
    """Evaluate a user profile against retrieved job-posting evidence."""

    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        generator: OllamaGenerator | None = None,
        generation_config: GenerationConfig | None = None,
        system_prompt: str = PROFILE_EVALUATION_SYSTEM_PROMPT,
        max_context_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
        max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
    ) -> None:
        self.retriever = retriever or QdrantRetriever()
        self.generator = generator or OllamaGenerator()
        self.generation_config = generation_config or GenerationConfig(num_predict=1024)
        self.system_prompt = system_prompt
        self.max_context_documents = max_context_documents
        self.max_preview_chars = max_preview_chars

    def evaluate(
        self,
        profile_text: str,
        target_role: str | None = None,
        top_k: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
        score_threshold: float | None = None,
    ) -> ProfileEvaluationResponse:
        """Evaluate one profile using retrieved market context."""
        profile_text = profile_text.strip()
        if not profile_text:
            raise ValueError("Profile text cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        retrieval_query = (
            f"{target_role.strip()}\n\n{profile_text}"
            if target_role and target_role.strip()
            else profile_text
        )
        documents = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        prompt = build_profile_evaluation_prompt(
            profile_text=profile_text,
            target_role=target_role,
            documents=documents,
            max_documents=self.max_context_documents,
            max_preview_chars=self.max_preview_chars,
        )
        evaluation = self.generator.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=self.generation_config,
        )
        return ProfileEvaluationResponse(
            evaluation=evaluation,
            profile_text=profile_text,
            target_role=target_role.strip() if target_role and target_role.strip() else None,
            documents=documents,
            prompt=prompt,
        )
