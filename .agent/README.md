# .agent Directory

Project documentation for AI assistants.

## Structure

```plaintext
.agent/
├── context.md        # Project overview, methodology, student profile
├── rules/            # Hard constraints and workspace rules
│   └── project-rules.md
├── specs/            # Project specifications
│   ├── mvp.md       # Features, success metrics, prioritization
│   ├── stack.md     # Technology stack
│   ├── decisions.md # Project decisions log and active risks
│   └── timeline.md  # CRISP-ML(Q) phases and learning roadmap
├── workflows/        # Operational workflows
│   ├── data-pipeline.md
│   └── git-workflow.md
└── docs/             # Technical documentation
    └── setup/        # Configuration guides
        └── aws-s3.md # AWS S3 + DVC setup
```

## Usage

Reference files with `@[.agent/filename.md]` or subdirectories when context is needed.

## Git Policy

- ✅ Commit all files except `private/` and `*.draft.md`
- ❌ Never commit secrets or credentials
