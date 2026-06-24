"""Download and verify the pinned local inference models."""

import argparse
import hashlib
import tomllib
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_DIR / "config" / "runtime_models.toml"
HASH_CHUNK_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    """Parse model preparation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("all", "llamacpp", "embeddings", "classifier"),
        default="all",
        help="Model component to prepare.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only artifacts already available in the Hugging Face cache.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify project model files without downloading or replacing them.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the versioned runtime-model manifest."""
    with path.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


def project_path(relative_path: str) -> Path:
    """Resolve a manifest path relative to the project root."""
    return PROJECT_DIR / relative_path


def download_file(
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
    local_files_only: bool,
) -> None:
    """Download one pinned Hugging Face file to a stable project directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        local_dir=local_dir,
        local_files_only=local_files_only,
    )


def calculate_sha256(path: Path) -> str:
    """Calculate a file SHA256 without loading the full artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> None:
    """Raise a clear error when a model file is missing or has changed."""
    if not path.is_file():
        raise FileNotFoundError(f"Required model file not found: {path}")

    actual_sha256 = calculate_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def prepare_llamacpp_model(
    config: dict[str, Any],
    local_files_only: bool,
    verify_only: bool,
) -> None:
    """Prepare and verify the pinned GGUF model."""
    model_dir = project_path(config["local_dir"])
    destination = model_dir / config["filename"]
    if not verify_only:
        download_file(
            repo_id=config["repo_id"],
            revision=config["revision"],
            filename=config["filename"],
            local_dir=model_dir,
            local_files_only=local_files_only,
        )

    print(f"Verifying llama.cpp model: {destination}")
    verify_sha256(destination, config["sha256"])
    print("llama.cpp model checksum is valid.")


def prepare_embedding_model(
    config: dict[str, Any],
    local_files_only: bool,
    verify_only: bool,
) -> None:
    """Prepare the pinned Sentence Transformer snapshot and test offline loading."""
    model_dir = project_path(config["local_dir"])
    if not verify_only:
        for filename in config["files"]:
            download_file(
                repo_id=config["repo_id"],
                revision=config["revision"],
                filename=filename,
                local_dir=model_dir,
                local_files_only=local_files_only,
            )

    missing_files = [
        filename for filename in config["files"] if not (model_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Embedding model files are missing from {model_dir}: {missing_files}"
        )

    print(f"Testing offline embedding model load: {model_dir}")
    model = SentenceTransformer(str(model_dir), local_files_only=True)
    embedding_dimension = model.get_embedding_dimension()
    print(f"Embedding model is valid (dimension={embedding_dimension}).")


def verify_classifier_artifacts(config: dict[str, Any]) -> None:
    """Verify the DVC-managed classifier inference bundle."""
    model_dir = project_path(config["local_dir"])
    model_path = model_dir / config["model_file"]
    label_encoder_path = model_dir / config["label_encoder_file"]

    print(f"Verifying classifier model: {model_path}")
    verify_sha256(model_path, config["model_sha256"])
    print(f"Verifying classifier label encoder: {label_encoder_path}")
    verify_sha256(label_encoder_path, config["label_encoder_sha256"])
    print(
        "Classifier artifacts are valid "
        f"(MLflow run={config['mlflow_run_id']}, model={config['mlflow_model_id']})."
    )


def main() -> None:
    """Prepare the requested reproducible runtime-model artifacts."""
    args = parse_args()
    manifest = load_manifest(DEFAULT_MANIFEST)

    if args.component in {"all", "llamacpp"}:
        prepare_llamacpp_model(
            manifest["llamacpp"],
            local_files_only=args.local_files_only,
            verify_only=args.verify_only,
        )
    if args.component in {"all", "embeddings"}:
        prepare_embedding_model(
            manifest["embeddings"],
            local_files_only=args.local_files_only,
            verify_only=args.verify_only,
        )
    if args.component in {"all", "classifier"}:
        verify_classifier_artifacts(manifest["classifier"])


if __name__ == "__main__":
    main()
