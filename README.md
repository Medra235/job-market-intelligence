# Job Market Intelligence

Local MLOps and GenAI project for exploring the US technology job market. The
system combines a processed LinkedIn job-postings dataset, a Random Forest
classifier, a Qdrant-backed RAG pipeline, a local llama.cpp LLM backend, a
FastAPI API, and a Streamlit demo interface.

## Repository Structure

- `src/job_market_intelligence/`: application package, API, RAG, ML services,
  configuration, and Streamlit frontend.
- `scripts/`: operational scripts for model preparation, checks, and evaluation.
- `config/runtime_models.toml`: pinned runtime model configuration.
- `data/processed.dvc`: DVC pointer for the processed CSV datasets.
- `models/random_forest.dvc`: DVC pointer for the selected classifier bundle.
- `compose.yaml`: local runtime with Qdrant, llama.cpp, FastAPI, and Streamlit.
- `docs/local_demo.md`: local demo and reproducibility runbook.
- `docs/full_reproduction.md`: from-source data and classifier rebuild guide.

## Local Demo

Main requirements:

- Python 3.12 and `uv`
- Docker and Docker Compose
- NVIDIA Container Toolkit for the CUDA llama.cpp container
- Hugging Face access for the pinned GGUF and Sentence Transformer snapshots
- Prepared data/classifier artifacts, obtained through a maintainer-provided
  artifact package or regeneration from source data

Validated local environment:

- NVIDIA GeForce RTX 5070 Ti

Clone the repository:

```bash
git clone git@github.com:Medra235/job-market-intelligence.git
cd job-market-intelligence
```

Install dependencies:

```bash
uv sync
```

If `data/processed/` and `models/random_forest/` are already available, use
[`docs/local_demo.md`](docs/local_demo.md) to prepare runtime models, start
Docker Compose, regenerate Qdrant if needed, and run smoke checks.

If those artifacts are not available, first rebuild them from the original
Kaggle dataset with [`docs/full_reproduction.md`](docs/full_reproduction.md),
then return to [`docs/local_demo.md`](docs/local_demo.md) for the local startup
flow.

Minimal startup once the required artifacts are present:

```bash
uv run python scripts/models/prepare_runtime_models.py
docker compose up -d
```

In a clean environment, create the Qdrant `job_postings` collection before using
the RAG features. The indexing command is documented in
[`docs/local_demo.md`](docs/local_demo.md#regenerate-qdrant).

Service URLs:

- FastAPI: <http://localhost:8000>
- Streamlit: <http://localhost:8501>
- Qdrant: <http://localhost:6333>
- llama.cpp: <http://localhost:8080/v1/models>

For detailed startup checks and troubleshooting, see
[`docs/local_demo.md`](docs/local_demo.md).

## Reproducibility Notes

Large runtime artifacts are kept outside Docker images. The GGUF LLM and
embedding model are prepared from pinned Hugging Face revisions. For private
development, processed data and the selected Random Forest bundle are versioned
with DVC in a private S3 remote. Qdrant is regenerated from `data/processed`
instead of versioning its local storage.

The public repository does not publish credentials or open access to the private
DVC remote. External reproduction should use either a maintainer-provided
artifact package or full regeneration from the original Kaggle dataset and
project pipelines.

## Data and Model Licenses

- [`LinkedIn Job Postings Dataset`](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings):
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [`unsloth/gemma-4-26B-A4B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF):
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
