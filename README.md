# activation_viz

A lightweight LLM activation vizualizer

Simple GUI app which lets you visualize per-layer activations of a small
local LLM as it generates a response.

- GUI written with Python + Tkinter
- LLM: `HuggingFaceTB/SmolLM-135M-Instruct`
    - For now, the LLM used is not configurable

## Quickstart

Note: running the app for the first time may take a bit — the local LLM
model files are downloaded from Hugging Face and cached for subsequent runs.

```bash
make sync
uv run activation_viz
```

### Mock mode

Set `USE_MOCK_LLM=1` to skip downloading/running the local LLM and instead
use canned token/activation data from `activation_viz/fixtures.py`:

```bash
USE_MOCK_LLM=1 uv run activation_viz
```

## Development

```bash
make sync       # install dependencies, create .venv
make test       # run pytest
make lint       # check types (mypy) and lint/format (ruff)
make lint-fix   # auto-fix lint/formatting issues
```

Run `make help` to list all available commands. See `AGENTS.md` for coding conventions.
