"""Profile evaluation flow backed by the job-market RAG corpus."""

from dataclasses import dataclass

from src.models.profile_classifier import CategoryPrediction, JobCategoryClassifier
from src.rag.generator import GenerationConfig, OllamaGenerator
from src.rag.prompts import (
    DEFAULT_MAX_CONTEXT_DOCUMENTS,
    DEFAULT_MAX_PREVIEW_CHARS,
    PROFILE_EVALUATION_SYSTEM_PROMPT,
    build_profile_evaluation_prompt,
    estimate_classifier_certainty,
)
from src.rag.retriever import QdrantRetriever, RetrievedDocument


@dataclass(frozen=True)
class ProfileEvaluationResponse:
    """
    Response returned by the profile evaluator.

    category_prediction is None when no classifier signal was requested or
    configured. It is not a semantic "unclassifiable profile" flag.
    """

    evaluation: str
    profile_text: str
    target_role: str | None
    category_prediction: CategoryPrediction | None
    documents: list[RetrievedDocument]
    prompt: str


class ProfileEvaluator:
    """Evaluate a user profile against retrieved job-posting evidence."""

    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        generator: OllamaGenerator | None = None,
        category_classifier: JobCategoryClassifier | None = None,
        generation_config: GenerationConfig | None = None,
        system_prompt: str = PROFILE_EVALUATION_SYSTEM_PROMPT,
        max_context_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
        max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
    ) -> None:
        self.retriever = retriever or QdrantRetriever()
        self.generator = generator or OllamaGenerator()
        self.category_classifier = category_classifier
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
        include_category_prediction: bool = True,
    ) -> ProfileEvaluationResponse:
        """
        Evaluate one profile using retrieved market context.

        The category prediction is optional. When omitted, it means the
        classifier signal was disabled or not configured, not that the profile
        could not be assigned to a category.
        """
        profile_text = profile_text.strip()
        if not profile_text:
            raise ValueError("Profile text cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        category_prediction = (
            self.category_classifier.predict(profile_text=profile_text)
            if include_category_prediction and self.category_classifier
            else None
        )
        retrieval_signal = (
            target_role.strip()
            if target_role and target_role.strip()
            else category_prediction.label
            if category_prediction
            else None
        )
        retrieval_query = (
            f"{retrieval_signal}\n\n{profile_text}"
            if retrieval_signal
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
            category_prediction=category_prediction,
            classifier_certainty=estimate_classifier_certainty(
                category_prediction.confidence
            )
            if category_prediction
            else None,
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
            target_role=target_role.strip()
            if target_role and target_role.strip()
            else None,
            category_prediction=category_prediction,
            documents=documents,
            prompt=prompt,
        )
