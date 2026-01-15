---
description: Git workflow for TFG development (main + develop strategy)
---

# Git Workflow

## Branch Strategy

```plaintext
main          # Stable, demo-ready code only
└── develop   # Active development (default working branch)
```

## Initial Setup

**Only run once at project start:**

```bash
# Create develop branch
git checkout -b develop
git push -u origin develop

# Set develop as default branch locally
git config branch.develop.remote origin
git config branch.develop.merge refs/heads/develop
```

## Daily Development Workflow

### Starting a work session

```bash
# Always work on develop
git checkout develop
git pull origin develop
```

### After making changes (with agent or manually)

```bash
# Stage relevant files
git add src/data/make_dataset.py
git add .agent/specs/decisions.md

# Commit with Conventional Commits format
git commit -m "feat: add data extraction pipeline"
# or
git commit -m "fix: resolve CSV encoding issue"
# or
git commit -m "docs: update timeline with completed tasks"

# Push to remote
git push origin develop
```

### End of work session

```bash
# Always commit and push before ending session
git add .
git commit -m "chore: end of session checkpoint"
git push origin develop
```

## Merging to Main (Stable Releases)

**Only when feature is 100% working and demo-ready:**

```bash
# Switch to main
git checkout main
git pull origin main

# Merge develop
git merge develop --no-ff -m "release: MVP data pipeline complete"

# Push to remote
git push origin main

# Return to develop for continued work
git checkout develop
```

## Commit Message Conventions

Follow **Conventional Commits**:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks
- `data:` DVC data updates

**Examples**:

```text
feat: implement RAG system with Qdrant
fix: resolve MinIO connection timeout
docs: add API endpoint documentation
refactor: extract transformation logic to utils
test: add unit tests for data cleaning
chore: update dependencies
data: update processed dataset v2
```

## Important Notes

- **Agent limitation**: Agent will PROPOSE git commands, you must APPROVE them
- **Commit frequency**: After each meaningful change (every 30-60 min of work)
- **main branch**: Reserve for stable, presentable versions only
- **develop branch**: Your daily workspace, can have temporary issues
