# Full Reproduction From Source

This guide describes how to rebuild the local demo artifacts from the original
Kaggle dataset instead of using a maintainer-provided artifact package.

It covers:

- raw Kaggle dataset placement
- processed dataset generation
- Random Forest training

After these files exist, continue with [`local_demo.md`](local_demo.md) to
prepare runtime models, start Docker Compose, regenerate Qdrant, and run smoke
checks.

Install project dependencies:

```bash
uv sync
```

## 1. Prepare Raw Kaggle Data

The project uses the Kaggle LinkedIn Job Postings Dataset:

```text
https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
```

Download the dataset archive from Kaggle and place only the CSV files required
by the ETL under `data/raw/`:

```text
data/raw/postings.csv
data/raw/salaries.csv
```

Optional files used only for exploration/notebooks:

```text
data/raw/companies.csv
data/raw/job_skills.csv
data/raw/skills.csv
```

## 2. Generate Processed Data

Run the ETL pipeline:

```bash
uv run python -m job_market_intelligence.data.make_dataset
```

Expected outputs:

```text
data/processed/train.csv
data/processed/test.csv
```

## 3. Train the Random Forest Classifier

Train only the Random Forest model:

```bash
uv run python -m job_market_intelligence.models.train_model --model rf
```

Expected runtime artifacts:

```text
models/random_forest/model.joblib
models/random_forest/label_encoder.pkl
```

The training script also creates local MLflow runs using the tracking URI
configured in `src/job_market_intelligence/config/mlflow_config.yml`.

The checksums in `config/runtime_models.toml` refer to the selected maintained
Random Forest bundle. A newly trained model may have different checksums.

## 4. Continue With Local Startup

After the processed data and classifier artifacts exist, continue with
[`local_demo.md`](local_demo.md).
