# Project Decisions Log

Key decisions made during TFG development, documenting rationale and context.

---

## January 2026: Initial Setup

### Decision: Public dataset over web scraping

**Context**: Needed job postings data source for market intelligence system

**Choice**: Use Kaggle [LinkedIn Job Postings Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (~33k postings, filter ~5-10k AI/ML)

**Rationale**:

- ❌ Web scraping too complex for beginner
- ❌ Not core to the TFG (core is MLOps + GenAI)
- ✅ Public dataset academically valid
- ✅ Allows immediate start on actual system development

---

### Decision: CRISP-ML(Q) over CRISP-DM

**Context**: Needed methodology framework for TFG

**Choice**: CRISP-ML(Q) instead of traditional CRISP-DM

**Rationale**: MLOps-focused methodology better aligned with project goals (production-ready system, not just model development)

---

### Decision: Scope adjustments from original proposal

**Context**: Initial proposal needed refinement based on time/complexity constraints

**Changes**:

| Aspect      | Original           | Current                 | Reason                         |
| ----------- | ------------------ | ----------------------- | ------------------------------ |
| Methodology | CRISP-DM           | CRISP-ML(Q)             | Better fit for MLOps focus     |
| Data        | Freelance scraping | Public tech dataset     | Reduce complexity, focus on ML |
| Scope       | Freelance market   | Tech/AI market          | Aligned with available data    |
| ML Phase    | 4 weeks            | 2 weeks (simple models) | Realistic for TFG timeline     |
| GenAI Phase | 4-5 weeks          | 5-6 weeks (more robust) | Core differentiator of project |
| MLOps Phase | 4 weeks            | 5 weeks (more emphasis) | Primary learning objective     |

---

## Active Risks

### RISK 1: Low quality data

**Mitigation**:

- LinkedIn dataset validated in Colab (Week 1-2)
- Plan B: 1-2 additional Kaggle/Hugging Face datasets
- Plan C: Simple scraping (50-100 postings) only if necessary

### RISK 2: Overly complex ML models

**Mitigation**:

- SIMPLE models (Logistic Regression, Random Forest)
- Realistic target metrics (70%, not 90%)
- Reduced ML phase (2 weeks vs 4)

### RISK 3: Scope creep

**Mitigation**:

- Clearly defined MVP
- "Nice to have" separated from "Must have"
- No feature additions after Week 15

### RISK 4: Unexpected Cloud costs

**Mitigation**:

- Local MinIO for development (€0 cost)
- AWS Budgets (alert at €0.01) if migrating
- Weekly Billing Dashboard review

---

## Future Decisions

Document new decisions here as they arise during development.
