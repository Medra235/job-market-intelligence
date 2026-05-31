"""
Feature Engineering Pipeline

Defines the preprocessing pipeline for the job category classification task.
The pipeline applies TF-IDF vectorization to job title and description text.

Target:
    - job_category (12 classes) -> encoded with LabelEncoder in train_model.py
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path Definitions
PROJECT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

# Columns used as features
TEXT_COMBINED_COL = "text_combined"


def add_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add a combined title + description column to the dataframe."""
    df = df.copy()
    df[TEXT_COMBINED_COL] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the feature dataframe from raw processed data.

    Adds the combined text column and selects only the columns
    needed by the preprocessor. Does not fit any transformers.
    """
    df = add_combined_text(df)

    required_cols = [TEXT_COMBINED_COL]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    features = df[required_cols].copy()
    return features


def build_preprocessor() -> ColumnTransformer:
    """
    Build the feature preprocessing pipeline.

    Returns an unfitted ColumnTransformer that combines:
    - TF-IDF on the combined text field (title + description)

    The ColumnTransformer is intended to be used as the first step
    of a full sklearn Pipeline alongside a classifier.
    """
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams and bigrams
        stop_words="english",
        sublinear_tf=True,  # apply log(1 + TF) to reduce effect of very frequent terms
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, TEXT_COMBINED_COL),
        ],
        remainder="drop",
    )

    return preprocessor


def main() -> None:
    """Load a sample and inspect the preprocessor output shape."""
    data_path = PROCESSED_DATA_DIR / "train.csv"

    logger.info(f"Loading sample from {data_path}...")
    df = pd.read_csv(data_path, nrows=500)

    X = prepare_features(df)
    logger.info(f"Feature dataframe shape: {X.shape}")
    logger.info(f"Columns: {list(X.columns)}")
    logger.info(
        f"Sample combined text (first row):\n  {X[TEXT_COMBINED_COL].iloc[0][:120]}..."
    )

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    logger.info(f"Transformed matrix shape: {X_transformed.shape}")
    logger.info(f"Matrix type: {type(X_transformed)}")
    logger.info(
        f"Non-zero elements: {X_transformed.nnz:,} out of {X_transformed.shape[0] * X_transformed.shape[1]:,}"
    )
    logger.info(
        f"Sparsity: {1 - X_transformed.nnz / (X_transformed.shape[0] * X_transformed.shape[1]):.1%}"
    )

    logger.info("Sanity check passed")


if __name__ == "__main__":
    main()
