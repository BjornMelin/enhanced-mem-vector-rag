from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.store.sqlite.base import SqliteIndexConfig

from emvr import MemoryRecord, MemoryService, SemanticIndexDisabledError

OBSERVED_AT = datetime(2026, 7, 12, 18, 30, tzinfo=UTC)


def record(text: str) -> MemoryRecord:
    return MemoryRecord.create(
        kind="note",
        content={"text": text},
        source="unit-test",
        observed_at=OBSERVED_AT,
        evidence=("fixture",),
    )


def test_semantic_query_fails_clearly_when_index_is_disabled(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "memory.sqlite") as service:
        service.upsert(("notes",), "one", record("cats"))
        with pytest.raises(SemanticIndexDisabledError, match="requires an index"):
            service.search(("notes",), query="cats")
        with pytest.raises(SemanticIndexDisabledError, match="requires an index"):
            service.native.search(("notes",), query="cats")


def test_optional_native_vector_index_uses_local_embedding_function(
    tmp_path: Path,
) -> None:
    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0] for text in texts]

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    with MemoryService(
        tmp_path / "vector.sqlite",
        index=index,
    ) as service:
        service.upsert(("notes",), "cat", record("cat care"))
        service.upsert(("notes",), "dog", record("dog care"))

        results = service.search(("notes",), query="cat", limit=1)
        assert results[0].key == "cat"
        assert results[0].score is not None


def test_import_has_no_network_side_effect(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    script = """
import socket

def blocked(*args, **kwargs):
    raise AssertionError('network access attempted during import')

socket.create_connection = blocked
socket.socket.connect = blocked
import emvr
assert emvr.SCHEMA_VERSION == 1
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root)
    for name in tuple(environment):
        if name.startswith("COV_CORE_"):
            del environment[name]
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
