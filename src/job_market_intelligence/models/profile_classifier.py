"""Inference helper for predicting a likely job category from profile text."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from job_market_intelligence.config.settings import settings
from job_market_intelligence.features.build_features import prepare_features

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"


@dataclass(frozen=True)
class CategoryPrediction:
    """Predicted job category signal from the classical classifier."""

    label: str
    confidence: float | None
    model_type: str


def build_profile_feature_frame(
    profile_text: str,
) -> pd.DataFrame:
    """Adapt free-form profile text to the training pipeline feature schema."""
    profile_text = profile_text.strip()
    if not profile_text:
        raise ValueError("Profile text cannot be empty.")

    source = pd.DataFrame(
        [
            {
                "title": "User profile",
                "description": profile_text,
            }
        ]
    )
    return prepare_features(source)


class JobCategoryClassifier:
    """Load the persisted sklearn pipeline and LabelEncoder for inference."""

    def __init__(
        self,
        model_type: str = settings.classifier_model_type,
        model_path: Path | None = None,
        label_encoder_path: Path | None = None,
    ) -> None:
        self.model_type = model_type
        model_dir = MODELS_DIR / model_type
        self.model_path = model_path or model_dir / "model.joblib"
        self.label_encoder_path = label_encoder_path or model_dir / "label_encoder.pkl"
        self.model = self._load_artifact(self.model_path, "model")
        self.label_encoder = self._load_artifact(
            self.label_encoder_path,
            "LabelEncoder",
        )

    @staticmethod
    def _load_artifact(path: Path, artifact_name: str):
        """Load one joblib artifact with a clear error for missing files."""
        if not path.exists():
            raise FileNotFoundError(f"{artifact_name} not found at {path}.")
        return joblib.load(path)

    def predict(
        self,
        profile_text: str,
    ) -> CategoryPrediction:
        """Predict the most likely job category for one profile."""
        features = build_profile_feature_frame(profile_text=profile_text)
        encoded_prediction = self.model.predict(features)
        label = self.label_encoder.inverse_transform(encoded_prediction)[0]
        confidence = self._predict_confidence(features)
        return CategoryPrediction(
            label=str(label),
            confidence=confidence,
            model_type=self.model_type,
        )

    def _predict_confidence(self, features: pd.DataFrame) -> float | None:
        """Return max class probability when the estimator exposes it."""
        if not hasattr(self.model, "predict_proba"):
            return None
        probabilities = self.model.predict_proba(features)
        return float(probabilities[0].max())
