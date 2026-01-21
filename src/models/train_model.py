"""
Baseline Logistic Regression Model for Skills Classification

This script trains a simple logistic regression model to classify
job postings as 'Information Technology' vs 'Other' based on
text features from title and description.

Features:
- TF-IDF vectorization of job title and description
- Simple logistic regression classifier
- MLflow integration for experiment tracking
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_config(config_path: str = "src/config/mlflow_config.yml") -> dict:
    """Load MLflow configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(data_path: str) -> tuple:
    """
    Load and prepare data for training.

    Creates a binary target: 'Information Technology' vs 'Other'
    """
    df = pd.read_csv(data_path)

    # Create binary target: IT vs Other
    df["skills_clean"] = df["skills"].str.strip("[]").str.replace("'", "")
    df["is_it"] = (
        df["skills_clean"].str.contains("Information Technology", na=False).astype(int)
    )

    # Combine title and description for text features
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

    # Remove rows with missing text
    df = df[df["text"].str.len() > 0].copy()

    print(f"Dataset size: {len(df)}")
    print(f"IT jobs: {df['is_it'].sum()} ({df['is_it'].mean()*100:.1f}%)")
    print(f"Other jobs: {(1-df['is_it']).sum()} ({(1-df['is_it']).mean()*100:.1f}%)")

    return df[["text", "is_it"]]


def train_baseline_model(X_train, X_val, y_train, y_val, config: dict):
    """Train baseline logistic regression model with MLflow tracking."""

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name="baseline_logistic_regression"):
        # Log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_iter", config["model"]["max_iter"])
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("target", "is_it_binary")

        # Create pipeline
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000, ngram_range=(1, 2), stop_words="english"
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=config["model"]["max_iter"],
                        random_state=config["model"]["random_state"],
                    ),
                ),
            ]
        )

        # Train model
        print("\nTraining model...")
        pipeline.fit(X_train, y_train)

        # Predict on validation set
        y_pred = pipeline.predict(X_val)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print("\nValidation Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=["Other", "IT"]))

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        print("\n✅ Model logged to MLflow")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        return pipeline, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


def main():
    """Main training pipeline."""

    # Load configuration
    config = load_config()

    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data(config["data"]["train_path"])

    X = df["text"]
    y = df["is_it"]

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    print(f"\nTrain set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")

    # Train model
    model, metrics = train_baseline_model(X_train, X_val, y_train, y_val, config)

    print("\n🎉 Training complete!")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    if metrics["f1_score"] >= 0.60:
        print("✅ Baseline goal achieved (F1 > 0.60)")
    else:
        print("⚠️  Baseline goal not yet reached (F1 < 0.60)")


if __name__ == "__main__":
    main()
