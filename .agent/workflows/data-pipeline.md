---
description: Run the complete data pipeline from raw to processed
---

# Data Pipeline

## Prerequisites

1. MinIO running in Docker
2. DVC configured with MinIO remote

## Step 1: Verify infrastructure

```bash
docker-compose up -d minio
dvc remote list
```

## Step 2: Get raw data

```bash
# If already in DVC
dvc pull data/raw

# If initial download from Kaggle
kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw
```

## Step 3: Run processing

```bash
python src/data/make_dataset.py
```

## Step 4: Validate output

```bash
# Verify generated files
ls -la data/processed/

# Verify record count
python -c "import pandas as pd; print(len(pd.read_csv('data/processed/train.csv')))"
```

## Step 5: Version with DVC

```bash
dvc add data/processed
git add data/processed.dvc
git commit -m "data: update processed dataset"
dvc push
```
