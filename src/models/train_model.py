"""
Multiclass Job Category Classifier - Training Script

Trains and registers two model candidates (Logistic Regression and
Random Forest) for multiclass classification of job postings into 12 categories.

Pipeline:
    - Feature preprocessing: TF-IDF on title + description text (build_features.py)
    - LabelEncoder for the target variable (job_category)
    - GridSearchCV: hyperparameter search using cross-validation on train data
    - MLflow tracking: parameters, CV metrics, and model artifacts
    - joblib serialization of the LabelEncoder for inference

Usage:
    python src/models/train_model.py --model both
    python src/models/train_model.py --model lr
    python src/models/train_model.py --model rf
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Path definitions
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.features.build_features import (  # noqa: E402
    build_preprocessor,
    prepare_features,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_DIR / "src" / "config" / "mlflow_config.yml"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_CHOICES = ("both", "lr", "rf")


def load_config() -> dict:
    """Load MLflow and model configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train job category classifier candidates."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="both",
        help="Model candidate to train: both, lr, or rf.",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load processed dataset from CSV."""
    path = PROJECT_DIR / data_path
    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    """Fit and return a LabelEncoder on the target column."""
    le = LabelEncoder()
    le.fit(y)
    logger.info(
        f"LabelEncoder fitted on {len(le.classes_)} classes: {list(le.classes_)}"
    )
    return le


def save_label_encoder(le: LabelEncoder, model_dir: Path) -> Path:
    """Persist the LabelEncoder to disk with joblib."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "label_encoder.pkl"
    joblib.dump(le, path)
    logger.info(f"LabelEncoder saved to {path}")
    return path


def build_pipeline(classifier) -> Pipeline:
    """
    Assemble the full Scikit-Learn pipeline.

    Combines the ColumnTransformer text preprocessor with the provided classifier
    into a single Pipeline object.
    """
    preprocessor = build_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def serializable_params(params: dict) -> dict:
    """Return MLflow-friendly params without nested search configuration."""
    serializable = {}
    for key, value in params.items():
        if key == "param_grid":
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                output_key = f"{key}_{nested_key}"
                serializable[output_key] = (
                    str(nested_value) if nested_value is None else nested_value
                )
        else:
            serializable[key] = str(value) if value is None else value
    return serializable


def build_search(pipeline: Pipeline, param_grid: dict, config: dict) -> GridSearchCV:
    """Create a GridSearchCV wrapper using the shared training configuration."""
    cv = StratifiedKFold(
        n_splits=config["training"]["cv_folds"],
        shuffle=True,
        random_state=config["training"]["random_state"],
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=config["training"]["scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )


def train_and_log(
    run_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train_encoded,
    params: dict,
    param_grid: dict,
    registered_model_name: str,
    config: dict,
    model_dir: Path,
    label_encoder_path: Path,
) -> dict:
    """
    Train a pipeline and log the candidate model to MLflow.

    Uses only the training split. The held-out test split is evaluated later in
    evaluate_model.py to keep final evaluation separate from model fitting.
    """
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters defined in config
        mlflow_params = serializable_params(params)
        mlflow.log_params(mlflow_params)
        mlflow.log_param("cv_folds", config["training"]["cv_folds"])
        mlflow.log_param("scoring", config["training"]["scoring"])

        # Train
        logger.info(f"Training '{run_name}' with GridSearchCV...")
        search = build_search(pipeline, param_grid, config)
        training_start = perf_counter()
        search.fit(X_train, y_train_encoded)
        training_duration_seconds = perf_counter() - training_start
        best_pipeline = search.best_estimator_
        best_params = {
            key: str(value) if value is None else value
            for key, value in search.best_params_.items()
        }

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("cv_best_f1_macro", search.best_score_)
        mlflow.log_metric("training_duration_seconds", training_duration_seconds)
        logger.info(f"Best params for '{run_name}': {search.best_params_}")

        model_path = model_dir / "model.joblib"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path)
        mlflow.log_artifact(str(label_encoder_path))

        # Log scikit-learn pipeline (preprocessor + model) as MLflow artifact
        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            registered_model_name=registered_model_name,
            input_example=X_train.head(5),
            params={**mlflow_params, **best_params},
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(
            f"Run '{run_name}' complete | CV F1 macro: {search.best_score_:.4f} | "
            f"Training duration: {training_duration_seconds:.1f}s | "
            f"Run ID: {run_id} | Model URI: {model_info.model_uri}"
        )
        return {
            "run_name": run_name,
            "run_id": run_id,
            "registered_model_name": registered_model_name,
            "model_uri": model_info.model_uri,
            "best_params": search.best_params_,
            "cv_best_f1_macro": search.best_score_,
            "training_duration_seconds": training_duration_seconds,
            "model_path": str(model_path),
        }


def should_train(selected_model: str, model_key: str) -> bool:
    """Return whether a candidate should be trained for this execution."""
    return selected_model == "both" or selected_model == model_key


def main() -> None:
    """Main training pipeline: loads data, fits both models, and logs to MLflow."""
    args = parse_args()
    config = load_config()

    # Configure MLflow
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    target_col = config["training"]["target_col"]

    # Load train data
    df_train = load_data(config["data"]["train_path"])

    # Prepare features (adds text_combined, selects relevant columns)
    X_train = prepare_features(df_train)
    y_train_raw = df_train[target_col]

    # Fit LabelEncoder on training labels
    label_encoder = fit_label_encoder(y_train_raw)
    y_train_encoded = label_encoder.transform(y_train_raw)

    results = []

    # -------------------------
    # Model 1: Logistic Regression
    # -------------------------
    if should_train(args.model, "lr"):
        lr_cfg = config["models"]["logistic_regression"]
        lr_classifier = LogisticRegression(**lr_cfg["fixed_params"])
        lr_dir = MODELS_DIR / "logistic_regression"
        lr_label_encoder_path = save_label_encoder(label_encoder, lr_dir)

        lr_pipeline = build_pipeline(lr_classifier)
        results.append(
            train_and_log(
                run_name="logistic_regression_baseline",
                pipeline=lr_pipeline,
                X_train=X_train,
                y_train_encoded=y_train_encoded,
                params={
                    **lr_cfg,
                    "model_type": "logistic_regression",
                    "target": target_col,
                },
                param_grid=lr_cfg["param_grid"],
                registered_model_name=config["registry"]["logistic_regression"],
                config=config,
                model_dir=lr_dir,
                label_encoder_path=lr_label_encoder_path,
            )
        )

    # -------------------------
    # Model 2: Random Forest
    # -------------------------
    if should_train(args.model, "rf"):
        rf_cfg = config["models"]["random_forest"]
        rf_classifier = RandomForestClassifier(**rf_cfg["fixed_params"])
        rf_dir = MODELS_DIR / "random_forest"
        rf_label_encoder_path = save_label_encoder(label_encoder, rf_dir)

        rf_pipeline = build_pipeline(rf_classifier)
        results.append(
            train_and_log(
                run_name="random_forest_baseline",
                pipeline=rf_pipeline,
                X_train=X_train,
                y_train_encoded=y_train_encoded,
                params={**rf_cfg, "model_type": "random_forest", "target": target_col},
                param_grid=rf_cfg["param_grid"],
                registered_model_name=config["registry"]["random_forest"],
                config=config,
                model_dir=rf_dir,
                label_encoder_path=rf_label_encoder_path,
            )
        )

    cv_summary = ", ".join(
        f"{result['run_name']}={result['cv_best_f1_macro']:.4f}" for result in results
    )
    logger.info(f"Training complete. CV summary: {cv_summary}")
    logger.info("Run evaluate_model.py to compute final test metrics.")


if __name__ == "__main__":
    main()
