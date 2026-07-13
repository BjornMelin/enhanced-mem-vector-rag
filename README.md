# Enhanced Memory

[![CI](https://github.com/BjornMelin/enhanced-mem-vector-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/BjornMelin/enhanced-mem-vector-rag/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/BjornMelin/enhanced-mem-vector-rag/blob/main/LICENSE)

Enhanced Memory gives agents local, durable memory. It composes LangGraph's native SQLite `BaseStore` with provenance, strict time-to-live (TTL) reads, deterministic export, and a standard-input/output Model Context Protocol (MCP) server.

It doesn't implement document ingestion, generic retrieval-augmented generation (RAG), crawling, agent orchestration, a hosted application programming interface (API), or another vector database. LangGraph owns storage semantics. FastMCP owns protocol behavior. SQLite owns persistence.

## Install

Add the Python library to a project:

```bash
uv add "enhanced-mem-vector-rag @ git+https://github.com/BjornMelin/enhanced-mem-vector-rag.git@v1.0.0"
```

Install the `emvr-mcp` command in an isolated environment:

```bash
uv tool install "enhanced-mem-vector-rag @ git+https://github.com/BjornMelin/enhanced-mem-vector-rag.git@v1.0.0"
```

PyPI trusted publishing is tracked in [issue #11](https://github.com/BjornMelin/enhanced-mem-vector-rag/issues/11). The Git tag remains the canonical install source until that publisher is verified.

For source development:

```bash
git clone https://github.com/BjornMelin/enhanced-mem-vector-rag.git
cd enhanced-mem-vector-rag
uv sync --frozen
uv run pytest
```

## Python API

Use `MemoryService` as the single owner of a local database connection:

```python
from pathlib import Path

from emvr import MemoryRecord, MemoryService

with MemoryService(Path("memory.sqlite3")) as memory:
    record = MemoryRecord.create(
        kind="preference",
        content={"theme": "dark", "font_size": 15},
        source="settings-import",
        evidence=("user-confirmed",),
    )
    memory.upsert(
        ("users", "bjorn", "preferences"),
        "editor",
        record,
    )

    saved = memory.get(("users", "bjorn", "preferences"), "editor")
    assert saved is not None
    print(saved.content)
```

`memory.native` is a lifecycle-guarded LangGraph `BaseStore`, so you can pass it directly to `graph.compile(store=...)`. Namespaces use non-empty tuples with non-empty labels that contain neither periods nor NUL bytes. The `langgraph` root is reserved. Namespace listing accepts `*` as a one-segment prefix or suffix wildcard.

The native surface supports filters, batch operations, TTL, and optional sqlite-vec indexing. Exact filters preserve JSON types, so booleans, numbers, arrays, and objects don't alias one another.

Semantic search is disabled by default. Construct the service with a LangGraph SQLite index configuration and an embedding function to enable cosine ranking. Other distance modes fail closed because the pinned SQLite adapter doesn't rank them consistently. The MCP server stays provider-free and supports exact lookup plus namespace and filter searches.

Semantic ranking supports the first 1,000 results per query. Narrow the namespace or filters instead of requesting a deeper semantic offset.

## MCP server

The server uses standard input and output only. This keeps the default deployment local and avoids a remote authentication boundary.

```json
{
  "mcpServers": {
    "enhanced-memory": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/BjornMelin/enhanced-mem-vector-rag.git@v1.0.0",
        "emvr-mcp"
      ],
      "env": {
        "EMVR_DB_PATH": "/absolute/path/to/memory.sqlite3"
      }
    }
  }
}
```

The server exposes nine tools:

- `memory_put`, `memory_get`, `memory_search`, and `memory_delete`
- `memory_purge_namespace` and `memory_list_namespaces`
- `memory_export`, `memory_sweep_expired`, and `memory_health`

`memory_put` is an upsert. Every record stores its source, observation time, optional evidence, and a SHA-256 content digest. `memory_export` returns stable JavaScript Object Notation Lines (JSON Lines) ordered by namespace and key.

## Persistence contract

Enhanced Memory preserves these storage guarantees:

- SQLite uses write-ahead logging (WAL) with `synchronous=NORMAL`, foreign keys enabled, and a five-second busy timeout.
- Expired records are swept before reads, searches, namespace listings, and exports, so callers don't receive stale records.
- Managed content must be a JSON object with string keys and finite numbers. Integers must fit LangGraph's native 64-bit storage range, from `-2^63` through `2^64 - 1`.
- TTL uses LangGraph's native minute-based sliding expiration semantics. Values must be finite, positive, and no more than 5,270,400 minutes (3,660 days).
- Vector indexing is opt-in. LangGraph's SQLite package supplies `sqlite-vec`, but Enhanced Memory selects no embedding provider or model.
- The on-disk schema is owned and migrated by `langgraph-checkpoint-sqlite`.

See [architecture](https://github.com/BjornMelin/enhanced-mem-vector-rag/blob/main/docs/architecture.md), [support](https://github.com/BjornMelin/enhanced-mem-vector-rag/blob/main/SUPPORT.md), and [security](https://github.com/BjornMelin/enhanced-mem-vector-rag/blob/main/SECURITY.md) for the operating boundary.

## Development

Run every local quality gate before opening a pull request:

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv build
```

## License

MIT. See the [license](https://github.com/BjornMelin/enhanced-mem-vector-rag/blob/main/LICENSE).
