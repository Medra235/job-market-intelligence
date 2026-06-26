# Local Demo Runbook

This runbook describes how to start the local demo once the processed data and
classifier artifacts are available. If they are missing, restore them from an
authorized artifact package, pull them from the private DVC remote as a
maintainer, or rebuild them with [`full_reproduction.md`](full_reproduction.md).

## Runtime Overview

Docker Compose starts four services:

- Qdrant vector database
- llama.cpp server with the `gemma-26b` model alias
- FastAPI backend at <http://localhost:8000>
- Streamlit frontend at <http://localhost:8501>

Large artifacts are mounted at runtime and are not baked into the API image.

## Requirements

- Python 3.12
- `uv`
- Docker and Docker Compose
- NVIDIA Container Toolkit for the CUDA llama.cpp image
- Network access to Hugging Face when preparing runtime models
- Prepared data/classifier artifacts, restored from a maintainer-provided
  artifact package or regenerated from the original dataset

## Prepare Artifacts

Install project dependencies:

```bash
uv sync
```

Restore processed data and classifier artifacts using one of these paths:

- extract a maintainer-provided artifact package into the expected project
  paths
- regenerate the artifacts from the original Kaggle dataset with
  [`full_reproduction.md`](full_reproduction.md)
- for maintainers only, restore the artifacts from the private DVC remote

Expected local paths:

```text
data/processed/train.csv
data/processed/test.csv
models/random_forest/model.joblib
models/random_forest/label_encoder.pkl
```

Maintainer-only DVC command:

```bash
uv run dvc pull data/processed.dvc models/random_forest.dvc
```

This command requires configured access to the private DVC remote. The project
does not publish credentials or provide public access to that remote.

Prepare and verify the pinned runtime models:

```bash
uv run python scripts/models/prepare_runtime_models.py
```

The runtime model configuration is `config/runtime_models.toml`. It pins:

- llama.cpp image and GGUF model revision/checksum
- `sentence-transformers/all-MiniLM-L6-v2` embedding snapshot
- Random Forest artifact checksums and MLflow run/model identifiers

By default, `prepare_runtime_models.py` prepares all runtime artifacts. To skip
Random Forest verification and prepare only Hugging Face artifacts:

```bash
uv run python scripts/models/prepare_runtime_models.py --component llamacpp
uv run python scripts/models/prepare_runtime_models.py --component embeddings
```

## Start the Demo

```bash
docker compose up -d
```

Check containers:

```bash
docker compose ps
```

## Regenerate Qdrant

Qdrant storage is kept in the local Docker volume, but it is not versioned. The
collection can be rebuilt from the processed dataset and the pinned embedding
model.

Start Qdrant if it is not running:

```bash
docker compose up -d qdrant
```

Regenerate the full corpus collection:

```bash
uv run python -m job_market_intelligence.rag.index_corpus \
  --model-name models/embeddings/all-MiniLM-L6-v2 \
  --full-corpus \
  --recreate
```

For a quick local check with the script default sample:

```bash
uv run python -m job_market_intelligence.rag.index_corpus \
  --model-name models/embeddings/all-MiniLM-L6-v2 \
  --recreate
```

Check retrieval:

```bash
uv run python scripts/rag/check_retrieval.py \
  --model-name models/embeddings/all-MiniLM-L6-v2 \
  --query "python sql airflow aws" \
  --top-k 3
```

## Smoke Checks

Qdrant:

```bash
curl http://localhost:6333/
curl http://localhost:6333/collections
```

llama.cpp:

```bash
curl http://localhost:8080/v1/models
```

FastAPI:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/market/summary
```

Application chat endpoint:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What should I learn for Data Engineering?","history":[],"top_k":3}'
```

Open the UI:

```text
http://localhost:8501
```

## Common Issues

Missing processed data or classifier artifacts:

```text
Restore a maintainer-provided artifact package, regenerate the artifacts from
the original Kaggle dataset and project pipelines, or use the private DVC remote
if you are the maintainer.
```

The DVC remote is declared in `.dvc/config` as `aws-s3`, but credentials are not
published with the repository. External users do not need DVC remote access when
using a maintainer-provided artifact package or regenerating the artifacts from
source.

Embedding model not prepared locally:

```bash
uv run python scripts/models/prepare_runtime_models.py --component embeddings
```

Qdrant collection missing:

```bash
uv run python -m job_market_intelligence.rag.index_corpus \
  --model-name models/embeddings/all-MiniLM-L6-v2 \
  --full-corpus \
  --recreate
```

llama.cpp service unhealthy:

```bash
docker compose logs llamacpp
curl http://localhost:8080/v1/models
```

API unhealthy:

```bash
docker compose logs api
curl http://localhost:8000/health
```

Stop the demo:

```bash
docker compose down
```
