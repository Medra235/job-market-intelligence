# Timeline and CRISP-ML(Q) Phases

## Phase 1: Business & Data Understanding (Week 1-2 | Dec 3-8)

- [x] Proposal defined
- [x] Technical setup (Python, Git, WSL2, Zotero)
- [x] LinkedIn dataset validated
- [x] Initial bibliography (7-10 papers)
- [x] Cookiecutter Data Science structure

## Phase 2: Data Engineering (Week 3-6 | Dec 9 - Jan 12)

- [ ] Infrastructure migration: GDrive → MinIO/S3 (MinIO configured, pending integration)
- [x] ETL Pipeline (documented and functional)
- [x] Data cleaning (duplicates, short descriptions removed)
- [x] Feature Engineering (skill_count, desc_length, has_salary)
- [x] Data splitting (80/20 train/test)
- [x] EDA (Exploratory Data Analysis)

## Phase 3: ML Engineering (Week 7-8 | Jan 13 - Feb 2)

- [ ] Skills classification (Logistic Regression) - **main model**
- [ ] Clustering (KMeans over embeddings)
- [ ] MLflow tracking
- [ ] Optuna optimization (10-20 iterations)

> 🎯 **Goal**: F1 >0.70, not perfection

## Phase 4: GenAI + RAG (Week 9-14 | Feb 3 - Mar 16)

- [ ] Complete RAG system (Qdrant + embeddings)
- [ ] Conversational chatbot (LangChain + memory)
- [ ] Structured prompt engineering
- [ ] Compare strategies
- [ ] Response quality evaluation

## Phase 5: Deployment & MLOps (Week 15-20 | Mar 17 - Apr 27)

- [ ] FastAPI REST API (6+ endpoints)
- [ ] Docker + Docker Compose
- [ ] CI/CD (GitHub Actions)
- [ ] Testing (Pytest, >70% coverage)
- [ ] Streamlit frontend (dashboard + chat)
- [ ] Basic monitoring (structured logging)

## Phase 6: Evaluation & Documentation (Week 21-26 | Apr 28 - Jun 26)

- [ ] Complete system testing
- [ ] Metrics evaluation
- [ ] Technical report (~60-80 pages)
- [ ] Defense presentation
