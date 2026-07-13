# pyright: reportUnusedFunction=false
"""Local stdio MCP server for Enhanced Memory."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from langgraph.store.base import TTLConfig
from pydantic import Field, JsonValue

from emvr.models import HealthReport, MemoryEntry, MemoryRecord
from emvr.store import (
    MAX_TTL_MINUTES,
    MemoryService,
    SemanticIndexDisabledError,
    validate_ttl_minutes,
)

DEFAULT_DB_PATH = Path("~/.local/share/emvr/memory.sqlite3").expanduser()

NamespaceSegment = Annotated[
    str, Field(min_length=1, max_length=255, pattern=r"^[^.\s\x00]+$")
]
Namespace = Annotated[list[NamespaceSegment], Field(min_length=1, max_length=16)]
MemoryKey = Annotated[str, Field(min_length=1, max_length=512, pattern=r"\S")]
Source = Annotated[str, Field(min_length=1, max_length=2_048, pattern=r"\S")]
EvidenceItem = Annotated[str, Field(min_length=1, max_length=8_192, pattern=r"\S")]
Evidence = Annotated[list[EvidenceItem], Field(max_length=100)]
Query = Annotated[str, Field(min_length=1, max_length=8_192, pattern=r"\S")]
TTLMinutes = Annotated[float, Field(strict=True, gt=0, le=MAX_TTL_MINUTES)]
ResultLimit = Annotated[int, Field(ge=1, le=1_000)]
ResultOffset = Annotated[int, Field(ge=0, le=10_000_000)]


def _service(ctx: Context) -> MemoryService:
    service = ctx.lifespan_context.get("memory_service")
    if not isinstance(service, MemoryService):
        raise RuntimeError("Memory service is unavailable")
    return service


def _ttl_from_env() -> float | None:
    raw = os.environ.get("EMVR_DEFAULT_TTL_MINUTES")
    if raw is None or not raw.strip():
        return None
    try:
        ttl = float(raw)
    except ValueError as exc:
        raise ValueError("EMVR_DEFAULT_TTL_MINUTES must be a number") from exc
    return validate_ttl_minutes(ttl, name="EMVR_DEFAULT_TTL_MINUTES")


def create_server(
    *,
    db_path: str | Path | None = None,
    default_ttl_minutes: float | None = None,
) -> FastMCP:
    """Create a stdio-oriented server backed by one lifespan-owned service."""
    path = Path(db_path or os.environ.get("EMVR_DB_PATH", DEFAULT_DB_PATH)).expanduser()
    ttl = _ttl_from_env() if default_ttl_minutes is None else default_ttl_minutes
    if ttl is not None:
        ttl = validate_ttl_minutes(ttl, name="default_ttl_minutes")

    @lifespan
    async def memory_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, object]]:
        path.parent.mkdir(parents=True, exist_ok=True)
        ttl_config: TTLConfig | None = (
            None if ttl is None else {"default_ttl": ttl, "refresh_on_read": True}
        )
        with MemoryService(path, ttl=ttl_config) as service:
            yield {"memory_service": service}

    server = FastMCP(
        name="Enhanced Memory",
        instructions="Local, durable, provenance-aware memory for agents.",
        lifespan=memory_lifespan,
        mask_error_details=True,
        on_duplicate="error",
    )

    @server.tool
    def memory_put(  # noqa: PLR0913
        namespace: Namespace,
        key: MemoryKey,
        content: dict[str, JsonValue],
        source: Source,
        ctx: Context,
        evidence: Evidence | None = None,
        ttl_minutes: TTLMinutes | None = None,
    ) -> MemoryEntry:
        """Create or replace one memory with provenance and optional sliding TTL."""
        service = _service(ctx)
        record = MemoryRecord.create(
            kind="memory",
            content=content,
            source=source,
            evidence=tuple(evidence or ()),
        )
        if ttl_minutes is None:
            service.upsert(tuple(namespace), key, record)
        else:
            service.upsert(
                tuple(namespace),
                key,
                record,
                ttl_minutes=ttl_minutes,
            )
        stored = service.get_entry(tuple(namespace), key)
        if stored is None:
            raise RuntimeError("Memory write did not persist")
        return stored

    @server.tool
    def memory_get(
        namespace: Namespace,
        key: MemoryKey,
        ctx: Context,
    ) -> MemoryEntry | None:
        """Retrieve one unexpired memory by exact namespace and key."""
        return _service(ctx).get_entry(tuple(namespace), key)

    @server.tool
    def memory_search(  # noqa: PLR0913
        namespace_prefix: Namespace,
        ctx: Context,
        query: Query | None = None,
        filter: dict[str, JsonValue] | None = None,  # noqa: A002
        limit: ResultLimit = 20,
        offset: ResultOffset = 0,
    ) -> list[MemoryEntry]:
        """Search unexpired memories within a namespace prefix."""
        try:
            return _service(ctx).search(
                tuple(namespace_prefix),
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
            )
        except SemanticIndexDisabledError:
            raise ToolError(
                "Semantic search is disabled; omit query or configure an index."
            ) from None

    @server.tool
    def memory_delete(
        namespace: Namespace,
        key: MemoryKey,
        ctx: Context,
    ) -> bool:
        """Delete one memory by exact namespace and key."""
        return _service(ctx).delete(tuple(namespace), key)

    @server.tool
    def memory_purge_namespace(
        namespace_prefix: Namespace,
        ctx: Context,
    ) -> int:
        """Delete every memory below a namespace prefix."""
        return _service(ctx).purge_namespace(tuple(namespace_prefix))

    @server.tool
    def memory_list_namespaces(
        ctx: Context,
        prefix: Namespace | None = None,
        limit: ResultLimit = 100,
        offset: ResultOffset = 0,
    ) -> list[tuple[str, ...]]:
        """List unexpired memory namespaces with bounded pagination."""
        return _service(ctx).list_namespaces(
            None if prefix is None else tuple(prefix),
            limit=limit,
            offset=offset,
        )

    @server.tool
    def memory_export(
        ctx: Context,
        namespace_prefix: Namespace | None = None,
    ) -> str:
        """Export unexpired memories as deterministic JSON Lines."""
        return _service(ctx).export_jsonl(
            () if namespace_prefix is None else tuple(namespace_prefix)
        )

    @server.tool
    def memory_sweep_expired(ctx: Context) -> int:
        """Delete expired memories immediately."""
        return _service(ctx).sweep_expired()

    @server.tool
    def memory_health(ctx: Context) -> HealthReport:
        """Report local SQLite storage health."""
        return _service(ctx).health()

    return server


mcp = create_server()


def main() -> None:
    """Run the Enhanced Memory server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
