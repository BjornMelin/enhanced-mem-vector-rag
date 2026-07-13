# Contributing

Use Python 3.11 or newer and `uv`.

```bash
uv sync --frozen
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
uv build
```

Changes must preserve the ownership boundary in [the architecture guide](docs/architecture.md). Prefer LangGraph, FastMCP, SQLite, or the Python standard library over custom storage or protocol code. Add or update tests for every contract change.
