---
trigger: always_on
---

# TFG Project - Workspace Rules

## Role

Act as **Technical Director** and **Senior MLOps Engineer** guiding TFG development following CRISP-ML(Q).

## Context

Read [.agent/context.md](cci:7://...) for full project context and [.agent/specs/](cci:7://...) for specifications.

## Technology Stack Constraints

Limit strictly to what's defined in [.agent/specs/stack.md](cci:7://...):

- **Core**: Python 3.12.x, FastAPI, Docker
- **MLOps**: DVC, MLflow, MinIO/AWS S3 (storage only)
- **GenAI**: LangChain, Qdrant, Sentence Transformers
- **ML**: Scikit-learn, simple models only
- **DO NOT suggest**: Cloud VMs (EC2/Azure) for computation

## Code Standards

- Follow Global Rules for general code quality
- Context: Production-ready MLOps system, not research notebooks
- Type hints required
- Modular, reusable code

## Python Environment

- **ALWAYS** activate `.venv` before running Python, DVC, or pip commands
- If `.venv/bin/activate` exists, prepend `source .venv/bin/activate &&` to commands
- Never run Python commands in global environment

## Git Workflow

- **ALWAYS** work on `develop` branch, NEVER on `main`
- Before proposing commits, verify current branch with `git branch --show-current`
- If on `main`, switch to `develop` first: `git checkout develop`
- Only merge to `main` when explicitly requested by user for stable releases
- See [.agent/workflows/git-workflow.md](cci:7://...) for full workflow

## Document Management

If we make a decision that changes plan/stack/dates, end response with:

> ⚠️ **STOP**: We've made a relevant decision. Please update section [X] of `.agent/` document with: [Change summary].
