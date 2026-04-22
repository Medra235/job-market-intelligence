"""Index processed job postings into Qdrant using sentence-transformers.

Usage examples:
    .venv/bin/python src/rag/index_corpus.py
    .venv/bin/python src/rag/index_corpus.py --limit 20 --recreate
    .venv/bin/python src/rag/index_corpus.py --batch-size 128 --collection job_postings
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_DIR / "data" / "processed" / "test.csv"
DEFAULT_COLLECTION = "job_postings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LIMIT = 20
DESCRIPTION_PREVIEW_CHARS = 300


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for indexing configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name to create or reuse.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Sentence-transformers embedding model name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant base URL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of documents to index per upload batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Optional maximum number of documents to index. "
            "Defaults to a small sample for the first iteration."
        ),
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Index the full processed dataset instead of a sample.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the target collection before indexing.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def load_corpus(limit: int | None) -> pd.DataFrame:
    """Load the processed train/test corpus and optionally keep only a sample."""
    logger.info("Loading processed corpus from train and test splits...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    corpus = pd.concat([train_df, test_df], ignore_index=True)

    required_cols = ["job_id", "title", "description", "job_category"]
    missing = [col for col in required_cols if col not in corpus.columns]
    if missing:
        raise ValueError(f"Missing required corpus columns: {missing}")

    corpus = corpus[required_cols].copy()
    corpus["title"] = corpus["title"].fillna("")
    corpus["description"] = corpus["description"].fillna("")
    corpus["job_category"] = corpus["job_category"].fillna("Unknown")

    if limit is not None:
        corpus = corpus.head(limit).copy()
        logger.info(f"Using a sample of {len(corpus)} documents for indexing.")
    else:
        logger.info(f"Using the full corpus: {len(corpus)} documents.")

    return corpus


def build_document_text(row: pd.Series) -> str:
    """Build the text that will be converted into an embedding."""
    title = row["title"].strip()
    description = row["description"].strip()
    return f"Title: {title}\n\nDescription: {description}".strip()


def build_payload(row: pd.Series) -> dict:
    """Build the payload stored in Qdrant alongside each vector."""
    description = row["description"].strip()
    return {
        "job_id": int(row["job_id"]),
        "title": row["title"].strip(),
        "job_category": row["job_category"],
        "description_preview": description[:DESCRIPTION_PREVIEW_CHARS],
    }


def get_embedding_model(
    model_name: str, local_files_only: bool = False
) -> SentenceTransformer:
    """Load the embedding model used for semantic retrieval."""
    logger.info(f"Loading embedding model: {model_name}")
    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the embedding model. "
            "sentence-transformers needs internet access to download the model "
            "from Hugging Face unless it is already cached locally."
        ) from exc


def get_qdrant_client(qdrant_url: str) -> QdrantClient:
    """Create a Qdrant client and verify connectivity."""
    client = QdrantClient(url=qdrant_url)
    client.get_collections()
    logger.info(f"Connected to Qdrant at {qdrant_url}")
    return client


def recreate_collection_if_requested(
    client: QdrantClient, collection_name: str, recreate: bool
) -> None:
    """Delete an existing collection when the CLI explicitly requests it."""
    if recreate and client.collection_exists(collection_name):
        logger.info(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dimension: int,
) -> None:
    """Create the collection if needed, using cosine distance."""
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists.")
        return

    logger.info(
        f"Creating collection '{collection_name}' with dimension {embedding_dimension}..."
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
    )


def index_documents(
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
    corpus: pd.DataFrame,
    batch_size: int,
) -> int:
    """Generate embeddings in batches and upload them to Qdrant."""
    texts = [build_document_text(row) for _, row in corpus.iterrows()]
    payloads = [build_payload(row) for _, row in corpus.iterrows()]
    ids = corpus["job_id"].astype(int).tolist()

    total_points = 0
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_payloads = payloads[start:end]
        batch_ids = ids[start:end]

        logger.info(f"Encoding and uploading batch {start}:{end}...")
        vectors = model.encode(batch_texts, show_progress_bar=False).tolist()
        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(batch_ids, vectors, batch_payloads)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        total_points += len(points)

    logger.info(f"Indexed {total_points} documents into '{collection_name}'.")
    return total_points


def main() -> None:
    """Run the first-pass indexing pipeline for the processed job corpus."""
    args = parse_args()
    limit = None if args.full_corpus else args.limit

    corpus = load_corpus(limit=limit)
    model = get_embedding_model(args.model_name, local_files_only=args.local_files_only)
    embedding_dimension = model.get_embedding_dimension()
    client = get_qdrant_client(args.qdrant_url)

    recreate_collection_if_requested(client, args.collection, args.recreate)
    ensure_collection(client, args.collection, embedding_dimension)
    index_documents(
        client=client,
        model=model,
        collection_name=args.collection,
        corpus=corpus,
        batch_size=args.batch_size,
    )

    logger.info(
        "Indexing complete. Next step: run a semantic query to validate retrieval."
    )


if __name__ == "__main__":
    main()
