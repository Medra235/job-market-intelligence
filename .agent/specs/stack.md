# Technology Stack

## MLOps (40%)

| Tool             | Purpose                        |
| ---------------- | ------------------------------ |
| Python 3.12.x    | Main language                  |
| Pandas/Polars    | Data processing                |
| DVC + MinIO      | Data versioning (local S3)     |
| Ruff             | Linter & Formatter             |
| Pandera          | Schema validation              |
| DuckDB           | Analytical SQL over files      |
| MLflow           | Experiment tracking            |
| Optuna           | Hyperparameter optimization    |
| FastAPI          | REST API                       |
| Docker + Compose | Containerization               |
| GitHub Actions   | CI/CD                          |
| Pytest           | Testing (>70% coverage target) |

## Generative AI (30%)

| Tool                  | Purpose             |
| --------------------- | ------------------- |
| LangChain             | LLM orchestration   |
| Sentence Transformers | Embeddings          |
| Qdrant                | Vector DB (Docker)  |
| OpenAI API / Ollama   | LLM (Llama 3 local) |

## Traditional ML (25%)

| Tool                | Purpose                    |
| ------------------- | -------------------------- |
| Scikit-learn        | Basic models               |
| Logistic Regression | Skills classification      |
| Random Forest       | Alternative classification |
| XGBoost             | Simple boosting            |
| KMeans              | Clustering over embeddings |

> ⚠️ **NO**: Complex deep learning, exhaustive optimization. Accuracy >70% is sufficient.

## Frontend (5%)

| Tool      | Purpose                    |
| --------- | -------------------------- |
| Streamlit | Dashboard + Chat interface |
| Plotly    | Visualizations             |

## Deployment Architecture

**Strategy**: Hybrid "Cloud-Smart" (Local Compute + Cloud Storage)

- **Development**: All local (Docker) with RTX 5070 Ti GPU
- **Current storage**: AWS S3 (Free Tier)
- **Previous storage**: MinIO (deprecated due to version incompatibility)
- **Demo deployment**:
  - Option A: Railway (Free Tier)
  - Option B: Localhost + Ngrok

> ❌ **Do NOT use**: AWS EC2/SageMaker (cost risk, unnecessary complexity)
