"""
Evaluate trained model candidates on the held-out test split.

The script loads Logistic Regression and/or Random Forest from MLflow, computes
final test metrics, logs them back to the source run with a `test_` prefix, and
writes local evaluation artifacts under `models/<model>/evaluation/`.

Usage:
    uv run python -m job_market_intelligence.models.evaluate_model --model both
    uv run python -m job_market_intelligence.models.evaluate_model --model lr
    uv run python -m job_market_intelligence.models.evaluate_model --model rf --run-id <RUN_ID>
"""

import argparse
import logging
import re
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from job_market_intelligence.features.build_features import prepare_features

PROJECT_DIR = Path(__file__).resolve().parents[3]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_PATH = (
    PROJECT_DIR / "src" / "job_market_intelligence" / "config" / "mlflow_config.yml"
)
MODELS_DIR = PROJECT_DIR / "models"


MODEL_TYPES = {
    "lr": "logistic_regression",
    "rf": "random_forest",
}


def load_config() -> dict:
    """Load project MLflow/model configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Evaluate trained MLflow models.")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID.")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "both"],
        default="both",
        help="Model to evaluate.",
    )
    args = parser.parse_args()
    if args.model == "both" and args.run_id:
        raise ValueError("--run-id can only be used when evaluating a single model.")
    return args


def load_test_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split and prepare model features."""
    target_col = config["training"]["target_col"]
    path = PROJECT_DIR / config["data"]["test_path"]
    logger.info(f"Loading test data from {path}...")

    df = pd.read_csv(path)
    X_test = prepare_features(df)
    y_test = df[target_col]

    logger.info(f"Test set: {len(df)} rows | {y_test.nunique()} classes")
    return X_test, y_test


def load_logged_model(run_id: str | None, config: dict, model_type: str):
    """Load the latest logged MLflow model for a model type, or one explicit run."""
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    if experiment is None:
        raise ValueError(
            f"Experiment '{config['experiment_name']}' not found. "
            "Run train_model.py first."
        )

    filter_str = f"params.model_type = '{model_type}'"
    if run_id:
        filter_str = f"source_run_id = '{run_id}' AND {filter_str}"

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[{"field_name": "creation_time", "ascending": False}],
        max_results=1,
        output_format="list",
    )
    if not logged_models:
        raise ValueError(f"No logged model found for model_type='{model_type}'.")

    logged_model = logged_models[0]
    model_uri = f"models:/{logged_model.model_id}"
    logger.info(f"Loading model | URI: {model_uri} | Run: {logged_model.source_run_id}")
    return mlflow.sklearn.load_model(model_uri), logged_model.source_run_id, model_uri


def load_label_encoder_from_mlflow(run_id: str):
    """Load the LabelEncoder artifact from the source MLflow run."""
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="label_encoder.pkl",
    )
    logger.info(f"Loading LabelEncoder from MLflow artifact: {artifact_path}")
    return joblib.load(artifact_path)


def load_label_encoder_from_local(model_type: str):
    """Load the LabelEncoder persisted locally during training."""
    path = MODELS_DIR / model_type / "label_encoder.pkl"
    if not path.exists():
        raise FileNotFoundError(f"LabelEncoder not found at {path}.")
    logger.info(f"Loading LabelEncoder from local path: {path}")
    return joblib.load(path)


def load_label_encoder(model_type: str, run_id: str):
    """Load the LabelEncoder from MLflow, falling back to the local copy."""
    try:
        return load_label_encoder_from_mlflow(run_id)
    except MlflowException as exc:
        logger.warning(
            "Could not load LabelEncoder from MLflow artifacts; "
            f"falling back to local copy. Reason: {exc}"
        )
        return load_label_encoder_from_local(model_type)


def safe_metric_suffix(label: str) -> str:
    """Convert a class label into a safe MLflow metric suffix."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", label.lower()).strip("_")


def compute_metrics(y_true, y_pred, labels) -> dict[str, float]:
    """Compute final aggregate and per-class test metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    per_class = {
        "precision": precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
    }
    for index, label in enumerate(labels):
        suffix = safe_metric_suffix(label)
        metrics[f"precision_class_{suffix}"] = per_class["precision"][index]
        metrics[f"recall_class_{suffix}"] = per_class["recall"][index]
        metrics[f"f1_class_{suffix}"] = per_class["f1"][index]

    return metrics


def write_artifacts(model_type: str, y_true, y_pred, labels) -> dict[str, Path]:
    """Write local evaluation artifacts for one model candidate."""
    output_dir = MODELS_DIR / model_type / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=labels, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=labels,
        columns=labels,
    )

    paths = {
        "report_txt": output_dir / "classification_report.txt",
        "report_json": output_dir / "classification_report.json",
        "confusion_matrix": output_dir / "confusion_matrix.csv",
    }
    paths["report_txt"].write_text(report_text)
    pd.DataFrame(report).transpose().to_json(paths["report_json"], indent=2)
    confusion.to_csv(paths["confusion_matrix"])
    return paths


def log_evaluation(
    run_id: str, model_uri: str, config: dict, metrics, artifacts
) -> None:
    """Log final test metrics and artifacts to the source MLflow run."""
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(
            {
                "evaluation_status": "test_evaluated",
                "evaluation_dataset": config["data"]["test_path"],
                "evaluated_model_uri": model_uri,
            }
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        for path in artifacts.values():
            mlflow.log_artifact(str(path), artifact_path="evaluation")


def evaluate_candidate(
    model_key: str,
    run_id: str | None,
    config: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Evaluate one candidate model and log final metrics/artifacts."""
    model_type = MODEL_TYPES[model_key]
    title = model_type.replace("_", " ").title()

    model, source_run_id, model_uri = load_logged_model(run_id, config, model_type)
    label_encoder = load_label_encoder(model_type, source_run_id)

    evaluation_start = perf_counter()
    y_pred = label_encoder.inverse_transform(model.predict(X_test))
    y_true = y_test.values
    labels = label_encoder.classes_

    metrics = compute_metrics(y_true, y_pred, labels)
    artifacts = write_artifacts(model_type, y_true, y_pred, labels)
    metrics["evaluation_duration_seconds"] = perf_counter() - evaluation_start
    log_evaluation(source_run_id, model_uri, config, metrics, artifacts)

    print_summary(title, source_run_id, metrics, artifacts)


def print_summary(title: str, run_id: str, metrics: dict, artifacts: dict) -> None:
    """Print a concise human-readable evaluation summary."""
    print("\n" + "=" * 65)
    print(f"MODEL EVALUATION - {title.upper()}")
    print("=" * 65)
    print(f"Using MLflow training run: {run_id}")
    print("Logged final test metrics with prefix: test_*")

    print("\nCLASSIFICATION REPORT")
    print(artifacts["report_txt"].read_text())

    print("CONFUSION MATRIX")
    print(pd.read_csv(artifacts["confusion_matrix"], index_col=0).to_string())

    print("\nSUMMARY")
    print(f"  Test Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Test F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  Test F1 Weighted: {metrics['f1_weighted']:.4f}")


def main() -> None:
    """Load test data and evaluate selected model candidates."""
    args = parse_args()
    config = load_config()
    mlflow.set_tracking_uri(config["tracking_uri"])

    X_test, y_test = load_test_data(config)
    model_keys = ["lr", "rf"] if args.model == "both" else [args.model]
    for model_key in model_keys:
        evaluate_candidate(model_key, args.run_id, config, X_test, y_test)


if __name__ == "__main__":
    main()
