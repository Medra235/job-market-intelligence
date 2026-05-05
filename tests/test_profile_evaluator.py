"""Tests for the RAG profile evaluator."""

import pytest

from src.models.profile_classifier import CategoryPrediction
from src.rag.profile_evaluator import (
    PROFILE_EVALUATION_SYSTEM_PROMPT,
    ProfileEvaluator,
    build_profile_evaluation_prompt,
    estimate_classifier_certainty,
)
from src.rag.retriever import RetrievedDocument


def sample_document() -> RetrievedDocument:
    """Return one retrieved document for evaluator tests."""
    return RetrievedDocument(
        job_id=123,
        title="Data Engineer",
        job_category="Data Engineering",
        description="Python, SQL, AWS and pipelines are required.",
        description_preview="Python, SQL and AWS role.",
        score=0.8,
        payload={"job_id": 123},
    )


class FakeRetriever:
    """Fake retriever returning deterministic documents."""

    def __init__(self) -> None:
        self.last_call = None

    def retrieve(self, query: str, top_k: int = 5, score_threshold=None):
        """Return one document and record retrieval arguments."""
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        return [sample_document()]


class FakeGenerator:
    """Fake generator returning a deterministic profile evaluation."""

    def __init__(self) -> None:
        self.last_call = None

    def generate(self, prompt: str, system_prompt=None, config=None) -> str:
        """Return an evaluation and record generation arguments."""
        self.last_call = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "config": config,
        }
        return "Refuerza AWS y pipelines."


class FakeCategoryClassifier:
    """Fake classifier returning a deterministic category prediction."""

    def __init__(self) -> None:
        self.last_call = None

    def predict(self, profile_text: str):
        """Return one category prediction and record arguments."""
        self.last_call = {
            "profile_text": profile_text,
        }
        return CategoryPrediction(
            label="Data Engineering",
            confidence=0.74,
            model_type="random_forest",
        )


def test_build_profile_evaluation_prompt_includes_profile_target_and_context() -> None:
    """Profile prompts should combine profile text, target role, and evidence."""
    prompt = build_profile_evaluation_prompt(
        profile_text="Python, SQL, pandas and basic AWS.",
        target_role="Data Engineer",
        documents=[sample_document()],
    )

    assert "Perfil del usuario" in prompt
    assert "Python, SQL, pandas and basic AWS." in prompt
    assert "Rol objetivo declarado por el usuario" in prompt
    assert "Data Engineer" in prompt
    assert "Ofertas recuperadas como referencia de mercado" in prompt
    assert "Usa exactamente esta" in prompt
    assert "- **resumen del encaje**:" in prompt
    assert "- **skills detectadas en el perfil**:" in prompt
    assert "fundamentos de entrada ya cubiertos" in prompt
    assert "siguientes pasos prácticos" in prompt
    assert "gaps avanzados o aspiracionales" in prompt
    assert "Agrupa las skills explícitas por tipo" in prompt
    assert '"AWS Core", "Herramientas de IaC"' in prompt
    assert "Agrupa gaps relacionados" in prompt
    assert "Da una recomendación priorizada y accionable" in prompt
    assert "ofertas con requisitos más avanzados" not in prompt
    assert "score orientativo de competitividad" not in prompt
    assert "score=0.8000" not in prompt


def test_build_profile_evaluation_prompt_includes_classifier_signal() -> None:
    """Profile prompts should include classifier certainty guidance."""
    prompt = build_profile_evaluation_prompt(
        profile_text="Python, SQL, pandas and basic AWS.",
        target_role="Data Engineer",
        category_prediction=CategoryPrediction(
            label="Data Engineering",
            confidence=0.74,
            model_type="random_forest",
        ),
        documents=[sample_document()],
    )

    assert "Resultado del clasificador ML" in prompt
    assert "categoría probable: Data Engineering" in prompt
    assert "confidence: 0.74" not in prompt
    assert "certeza estimada: alta" in prompt
    assert "modelo: random_forest" not in prompt


def test_profile_evaluation_system_prompt_includes_classifier_certainty_rules() -> None:
    """Classifier weighting rules should live in the evaluator system prompt."""
    system_prompt = PROFILE_EVALUATION_SYSTEM_PROMPT.replace("\n", " ")

    assert "Use classifier certainty as follows" in PROFILE_EVALUATION_SYSTEM_PROMPT
    assert "alta: use the predicted category as the main evaluation frame" in (
        system_prompt
    )
    assert "baja: present it as tentative" in system_prompt
    assert "Do not add implicit or assumed skills" in system_prompt


@pytest.mark.parametrize(
    ("confidence", "expected_certainty"),
    [
        (None, "no disponible"),
        (0.49, "baja"),
        (0.50, "media"),
        (0.69, "media"),
        (0.70, "alta"),
    ],
)
def test_estimate_classifier_certainty(
    confidence: float | None,
    expected_certainty: str,
) -> None:
    """Classifier confidence should be exposed as a qualitative certainty band."""
    assert estimate_classifier_certainty(confidence) == expected_certainty


def test_profile_evaluator_retrieves_builds_prompt_and_generates_evaluation() -> None:
    """The evaluator should connect retrieval, prompt building, and generation."""
    retriever = FakeRetriever()
    generator = FakeGenerator()
    evaluator = ProfileEvaluator(
        retriever=retriever,
        generator=generator,
        max_context_documents=1,
        max_preview_chars=100,
    )

    response = evaluator.evaluate(
        profile_text="Python, SQL and pandas.",
        target_role="Data Engineer",
        top_k=3,
        score_threshold=0.4,
    )

    assert retriever.last_call == {
        "query": "Data Engineer\n\nPython, SQL and pandas.",
        "top_k": 3,
        "score_threshold": 0.4,
    }
    assert "Python, SQL and pandas." in generator.last_call["prompt"]
    assert "Data Engineer" in generator.last_call["prompt"]
    assert response.evaluation == "Refuerza AWS y pipelines."
    assert response.target_role == "Data Engineer"
    assert response.documents[0].job_id == 123
    system_prompt = generator.last_call["system_prompt"].replace("\n", " ")
    assert "junior, self-taught, bootcamp" in system_prompt
    assert "aspirational market evidence" in system_prompt


def test_profile_evaluator_adds_category_prediction_when_classifier_is_present() -> None:
    """The evaluator should expose and pass the classical category signal."""
    retriever = FakeRetriever()
    generator = FakeGenerator()
    classifier = FakeCategoryClassifier()
    evaluator = ProfileEvaluator(
        retriever=retriever,
        generator=generator,
        category_classifier=classifier,
    )

    response = evaluator.evaluate(
        profile_text="Python, SQL and pandas.",
        target_role="Data Engineer",
    )

    assert classifier.last_call == {"profile_text": "Python, SQL and pandas."}
    assert retriever.last_call["query"] == "Data Engineer\n\nPython, SQL and pandas."
    assert response.category_prediction == CategoryPrediction(
        label="Data Engineering",
        confidence=0.74,
        model_type="random_forest",
    )
    assert "categoría probable: Data Engineering" in generator.last_call["prompt"]


def test_profile_evaluator_uses_predicted_category_for_retrieval_without_target() -> None:
    """Predicted category should guide retrieval when no target role is declared."""
    retriever = FakeRetriever()
    evaluator = ProfileEvaluator(
        retriever=retriever,
        generator=FakeGenerator(),
        category_classifier=FakeCategoryClassifier(),
    )

    evaluator.evaluate(profile_text="Python, SQL and pandas.")

    assert retriever.last_call["query"] == "Data Engineering\n\nPython, SQL and pandas."


def test_profile_evaluator_can_skip_category_prediction() -> None:
    """Classification should remain optional even when a classifier is injected."""
    classifier = FakeCategoryClassifier()
    evaluator = ProfileEvaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        category_classifier=classifier,
    )

    response = evaluator.evaluate(
        profile_text="Python, SQL and pandas.",
        include_category_prediction=False,
    )

    assert classifier.last_call is None
    assert response.category_prediction is None
    assert "Resultado del clasificador ML" not in response.prompt


@pytest.mark.parametrize("profile_text", ["", "   "])
def test_profile_evaluator_rejects_empty_profile(profile_text: str) -> None:
    """Empty profile text should fail before retrieval."""
    evaluator = ProfileEvaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    with pytest.raises(ValueError, match="Profile text cannot be empty"):
        evaluator.evaluate(profile_text=profile_text)


def test_profile_evaluator_rejects_invalid_top_k() -> None:
    """top_k must be positive."""
    evaluator = ProfileEvaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        evaluator.evaluate(profile_text="Python and SQL.", top_k=0)
