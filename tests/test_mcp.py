"""Protocol-level tests for the local Enhanced Memory MCP server."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from mcp.types import TextContent

from emvr.mcp import create_server

EXPECTED_TOOLS = {
    "memory_put",
    "memory_get",
    "memory_search",
    "memory_delete",
    "memory_purge_namespace",
    "memory_list_namespaces",
    "memory_export",
    "memory_sweep_expired",
    "memory_health",
}


def _server(path: Path) -> FastMCP:
    return create_server(db_path=path)


async def test_tool_catalog_and_schemas(tmp_path: Path) -> None:
    async with Client(_server(tmp_path / "memory.sqlite3")) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    by_name = {tool.name: tool for tool in tools}
    search_schema = by_name["memory_search"].inputSchema
    assert search_schema["properties"]["namespace_prefix"]["minItems"] == 1
    assert search_schema["properties"]["limit"]["minimum"] == 1
    assert search_schema["properties"]["limit"]["maximum"] == 1_000
    output_schema = by_name["memory_get"].outputSchema
    assert output_schema is not None
    assert output_schema["type"] == "object"


async def test_memory_tools_round_trip_over_protocol(tmp_path: Path) -> None:
    server = _server(tmp_path / "memory.sqlite3")
    namespace = ["users", "bjorn", "preferences"]

    async with Client(server) as client:
        put = await client.call_tool(
            "memory_put",
            {
                "namespace": namespace,
                "key": "editor",
                "content": {"theme": "dark", "font_size": 15},
                "source": "settings-import",
                "evidence": ["user-confirmed"],
                "ttl_minutes": 10,
            },
        )
        assert put.structured_content is not None
        assert put.structured_content["record"]["content"] == {
            "theme": "dark",
            "font_size": 15,
        }
        assert put.structured_content["record"]["provenance"]["source"] == (
            "settings-import"
        )

        get = await client.call_tool(
            "memory_get", {"namespace": namespace, "key": "editor"}
        )
        assert get.structured_content is not None
        assert get.structured_content["result"]["key"] == "editor"
        assert get.structured_content["result"]["created_at"].endswith("Z")
        assert get.structured_content["result"]["updated_at"].endswith("Z")

        search = await client.call_tool(
            "memory_search", {"namespace_prefix": namespace, "limit": 10}
        )
        assert search.structured_content is not None
        assert len(search.structured_content["result"]) == 1

        await client.call_tool(
            "memory_put",
            {
                "namespace": namespace,
                "key": "integer",
                "content": {"enabled": 1},
                "source": "test",
            },
        )
        typed = await client.call_tool(
            "memory_search",
            {
                "namespace_prefix": namespace,
                "filter": {"content.enabled": True},
            },
        )
        assert typed.structured_content == {"result": []}

        semantic = await client.call_tool(
            "memory_search",
            {"namespace_prefix": namespace, "query": "dark theme"},
            raise_on_error=False,
        )
        assert semantic.is_error is True
        assert isinstance(semantic.content[0], TextContent)
        assert "Semantic search is disabled" in semantic.content[0].text

        listed = await client.call_tool(
            "memory_list_namespaces", {"prefix": ["users"], "limit": 10}
        )
        assert listed.structured_content is not None
        assert namespace in listed.structured_content["result"]

        await client.call_tool(
            "memory_put",
            {
                "namespace": ["users-archive"],
                "key": "collision",
                "content": {"archived": True},
                "source": "test",
            },
        )
        listed_again = await client.call_tool(
            "memory_list_namespaces", {"prefix": ["users"], "limit": 10}
        )
        assert listed_again.structured_content is not None
        assert ["users-archive"] not in listed_again.structured_content["result"]

        exported = await client.call_tool(
            "memory_export", {"namespace_prefix": ["users"]}
        )
        assert exported.structured_content is not None
        assert '"key":"editor"' in exported.structured_content["result"]

        deleted = await client.call_tool(
            "memory_delete", {"namespace": namespace, "key": "editor"}
        )
        assert deleted.structured_content is not None
        assert deleted.structured_content["result"] is True

        missing = await client.call_tool(
            "memory_get", {"namespace": namespace, "key": "editor"}
        )
        assert missing.structured_content == {"result": None}

        deleted_again = await client.call_tool(
            "memory_delete", {"namespace": namespace, "key": "editor"}
        )
        assert deleted_again.structured_content is not None
        assert deleted_again.structured_content["result"] is False

        health = await client.call_tool("memory_health", {})
        assert health.structured_content is not None
        assert health.structured_content["status"] == "ok"


async def test_namespace_purge_and_sweep_are_structured(tmp_path: Path) -> None:
    server = _server(tmp_path / "memory.sqlite3")
    async with Client(server) as client:
        for key in ("one", "two"):
            await client.call_tool(
                "memory_put",
                {
                    "namespace": ["sessions", "alpha"],
                    "key": key,
                    "content": {"value": key},
                    "source": "test",
                },
            )

        purged = await client.call_tool(
            "memory_purge_namespace", {"namespace_prefix": ["sessions"]}
        )
        assert purged.structured_content is not None
        assert purged.structured_content["result"] == 2

        swept = await client.call_tool("memory_sweep_expired", {})
        assert swept.structured_content == {"result": 0}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("memory_get", {"namespace": [], "key": "x"}),
        ("memory_get", {"namespace": ["tenant.user"], "key": "x"}),
        ("memory_get", {"namespace": ["tenant\x00user"], "key": "x"}),
        ("memory_search", {"namespace_prefix": ["x"], "limit": 0}),
        (
            "memory_put",
            {
                "namespace": ["x"],
                "key": "x",
                "content": {},
                "source": "test",
                "ttl_minutes": -1,
            },
        ),
        (
            "memory_put",
            {
                "namespace": ["x"],
                "key": "x",
                "content": {},
                "source": "test",
                "ttl_minutes": True,
            },
        ),
    ],
)
async def test_malformed_inputs_return_protocol_errors(
    tmp_path: Path, tool: str, arguments: dict[str, object]
) -> None:
    async with Client(_server(tmp_path / "memory.sqlite3")) as client:
        result = await client.call_tool(
            tool,
            arguments,
            raise_on_error=False,
        )

    assert result.is_error is True


async def test_memory_persists_across_server_lifespans(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    server = _server(path)
    arguments = {
        "namespace": ["agents", "planner"],
        "key": "goal",
        "content": {"text": "Ship the release"},
        "source": "test",
    }

    async with Client(server) as client:
        await client.call_tool("memory_put", arguments)

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_get",
            {"namespace": arguments["namespace"], "key": arguments["key"]},
        )

    assert result.structured_content is not None
    assert result.structured_content["result"]["key"] == "goal"


def test_create_server_rejects_invalid_default_ttl(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="default_ttl_minutes"):
        create_server(
            db_path=tmp_path / "memory.sqlite3",
            default_ttl_minutes=0,
        )


@pytest.mark.parametrize("value", ["not-a-number", "0", "nan"])
def test_create_server_rejects_invalid_ttl_env(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EMVR_DEFAULT_TTL_MINUTES", value)
    with pytest.raises(ValueError, match="EMVR_DEFAULT_TTL_MINUTES"):
        create_server()


async def test_server_uses_environment_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "configured" / "memory.sqlite3"
    monkeypatch.setenv("EMVR_DB_PATH", str(path))
    monkeypatch.setenv("EMVR_DEFAULT_TTL_MINUTES", "5")

    async with Client(create_server()) as client:
        await client.call_tool(
            "memory_put",
            {
                "namespace": ["configured"],
                "key": "default-ttl",
                "content": {"enabled": True},
                "source": "test",
            },
        )
        exported = await client.call_tool("memory_export", {})
        listed = await client.call_tool("memory_list_namespaces", {})

    assert path.exists()
    with closing(sqlite3.connect(path)) as connection:
        ttl_minutes = connection.execute(
            "SELECT ttl_minutes FROM store WHERE key = ?", ("default-ttl",)
        ).fetchone()
    assert ttl_minutes == (5.0,)
    assert exported.structured_content is not None
    assert '"key":"default-ttl"' in exported.structured_content["result"]
    assert listed.structured_content == {"result": [["configured"]]}
