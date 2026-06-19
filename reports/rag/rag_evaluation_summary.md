# RAG Evaluation Summary

This evaluation combines automatic retrieval metrics over a manual golden set
with a small manual review of generated answers.

- Golden set: `data/evaluation/rag_golden_set.yaml`
- Retrieval evaluation: 12 queries with `top_k=5`
- Manual review: 5 selected cases
- LLM model used for manual review: `qwen3.5:9b`

## Retrieval Metrics

- `Precision@5`: proportion of the top 5 retrieved documents marked as relevant.
- `Hit@5`: whether at least one relevant document appears in the top 5.
- `MRR@5`: reciprocal rank of the first relevant document in the top 5.

| Metric           |  Value |
| ---------------- | -----: |
| Mean Precision@5 | 0.9500 |
| Mean Hit@5       | 1.0000 |
| Mean MRR@5       | 0.9583 |

## Manual Generation Review

Scores:

- `0`: poor
- `1`: partial
- `2`: good

Criteria:

- Context relevance: retrieved sources contain useful evidence for the query.
- Faithfulness: the answer is supported by the retrieved sources.
- Answer relevance: the answer directly addresses the user's goal.

| Query ID                         | Context relevance | Faithfulness | Answer relevance | Notes                                                                                                              |
| -------------------------------- | ----------------: | -----------: | ---------------: | ------------------------------------------------------------------------------------------------------------------ |
| `data_engineering_learning_path` |               2/2 |          2/2 |              2/2 | Sources strongly support Python, SQL, AWS/Redshift and data pipeline recommendations.                              |
| `devops_platform_skills`         |               2/2 |          2/2 |              2/2 | Sources support Kubernetes, cloud platforms, CI/CD, IaC, monitoring and Azure DevOps.                              |
| `ai_ml_engineer_skills`          |               2/2 |          1/2 |              1/2 | Sources are relevant, but the answer overstates Docker/Kubernetes/MLOps compared with the retrieved evidence.      |
| `backend_api_development`        |               2/2 |          2/2 |              2/2 | Automatic retrieval was weak by category, but the retrieved descriptions are useful and the answer stays grounded. |
| `follow_up_mlops_deployment`     |               2/2 |          2/2 |              2/2 | Rewritten standalone query retrieves strong MLOps evidence and the answer is well supported.                       |

Average manual scores:

| Metric            | Average |
| ----------------- | ------: |
| Context relevance |    2.00 |
| Faithfulness      |    1.80 |
| Answer relevance  |    1.80 |
