---
name: makefile
description: "Complete reference for all make targets in the project. Use when: looking up the right make command for any task — setup, testing, linting, formatting, or packaging."
---

# Makefile Reference

All developer tasks are exposed as `make` targets. Run from the project root. Run `make` or `make help` to list them.

---

| Target            | What it does                                                          |
| ----------------- | ---------------------------------------------------------------------- |
| `make help`       | List available commands (default target)                              |
| `make sync`       | Install/sync dependencies, create `.venv`                              |
| `make pre-commit` | Install pre-commit hooks                                               |
| `make test`       | Run pytest with coverage report                                        |
| `make lint`       | Check types (mypy) and lint/format (ruff); prints the fix command on failure |
| `make lint-fix`   | Auto-fix lint and formatting issues with ruff                          |
| `make build`      | Build the package (wheel + sdist)                                      |

**Typical workflow before committing:** `make lint-fix && make lint && make test`
