# CLAUDE.md

This file provides guidance to AI assistants (Claude, etc.) working in this repository.

## Repository Overview

**Name:** gpt
**Owner:** lucas2xx
**Purpose:** Learning project — currently in early/bootstrapping stage.

This is a minimal repository with no established language, framework, or build system yet. The codebase consists of:

| File | Description |
|------|-------------|
| `README.md` | Project title only (`# gpt`) |
| `Claude` | Empty file, created as a learning exercise |

## Repository State

This repository is essentially a blank slate. There are no:
- Source files or application code
- Package managers or dependency files (`package.json`, `requirements.txt`, `go.mod`, etc.)
- Build scripts or CI/CD configuration
- Tests or linting rules
- Established directory structure

## Git Conventions

### Branches
- `master` — default branch (contains the initial commit)
- `main` — remote default branch on origin
- `claude/*` — branches used by AI assistants for contributions

### Commit History
```
f9eddcb  Create Claude        (adds empty Claude file)
9d166d8  Initial commit       (adds README.md with project title)
```

### Commit Message Style
Based on existing commits, keep messages short and imperative:
- `Create <filename>`
- `Add <feature>`
- `Fix <issue>`

## Development Workflow

Since no build system exists, there are no build, test, or lint commands to run. When the project grows:

1. **Before adding code**, establish the language/framework and document it here.
2. **Update this file** whenever new tooling, conventions, or structure is introduced.
3. **Keep commits focused** — one logical change per commit.

## Guidance for AI Assistants

- **Do not assume a tech stack.** The project has none yet. Ask the user before introducing one.
- **Keep changes minimal.** This is a learning project; avoid over-engineering.
- **Document as you go.** Update this CLAUDE.md whenever new conventions, tools, or structure are added.
- **Branch discipline:** All AI-driven work should go to a `claude/` prefixed branch and be pushed via `git push -u origin <branch>`.
- **No tests exist** — do not reference or run test commands until a testing framework is added and documented here.

## Future Sections (fill in as the project evolves)

When the project grows, add the following sections to this file:

- **Tech Stack** — language, runtime version, frameworks
- **Directory Structure** — what lives where and why
- **Build & Run** — how to install dependencies and start the project
- **Testing** — how to run tests and what coverage is expected
- **Linting / Formatting** — tools used and how to run them
- **Environment Variables** — required `.env` keys and their purpose
- **Deployment** — how and where the project is deployed
