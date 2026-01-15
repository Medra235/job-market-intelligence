# MVP - Features and Metrics

## Main Features

### 1. Market Intelligence Dashboard

- Top 20 most demanded skills (bar chart)
- Temporal demand evolution (line chart)
- Salary distribution by skill (box plots)
- Job postings clustering (UMAP scatter plot)

### 2. Conversational Chatbot (RAG)

- User profile analysis (input: CV or text)
- Market comparison (via RAG)
- Priority gap identification
- Personalized learning roadmap
- Conversational memory (multi-turn context)

### 3. Profile Evaluator

**Input**: CV or skills description

**Output**:

- Competitiveness score (0-100)
- Categorized detected skills
- Gap analysis with salary impact
- Profile-based salary estimation
- Prioritized recommendations

---

## Success Metrics

### ML

| Metric                     | Target |
| -------------------------- | ------ |
| Skills classification (F1) | >0.70  |
| Clustering (Silhouette)    | >0.40  |
| Salary prediction (MAE)    | <25%   |

### System

| Metric          | Target                   |
| --------------- | ------------------------ |
| API latency p95 | <3s (normal), <7s (chat) |
| Test coverage   | >50%                     |
| Demo uptime     | >95%                     |

### Functional

| Metric               | Target            |
| -------------------- | ----------------- |
| Analyzed postings    | ≥5,000            |
| Taxonomy skills      | ≥50               |
| Chatbot conversation | 5+ coherent turns |
| Reproducible setup   | <30 min           |

---

## Prioritization

### MUST HAVE (Core MVP)

- ✅ Functional data pipeline
- ✅ 1 trained + evaluated ML model
- ✅ Basic REST API (3-4 endpoints)
- ✅ Docker + local deployment
- ✅ Minimum documentation

### SHOULD HAVE (Actual target)

- ✅ 2 ML models
- ✅ Basic RAG system
- ✅ Functional chatbot
- ✅ MLflow tracking
- ✅ CI/CD

### NICE TO HAVE (If time permits)

- Elaborate frontend
- Advanced monitoring (Grafana)
- Public cloud deployment
- Airflow orchestration

> ⚠️ If time remains (Week 15+): Prioritize robust MLOps > Sophisticated GenAI > Frontend > ~~Scraping~~
