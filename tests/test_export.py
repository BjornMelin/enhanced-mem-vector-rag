from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from emvr import MemoryRecord, MemoryService

OBSERVED_AT = datetime(2026, 7, 12, 18, 30, tzinfo=UTC)


def record(value: int) -> MemoryRecord:
    return MemoryRecord.create(
        kind="fact",
        content={"value": value, "nested": {"z": 1, "a": 2}},
        source="unit-test",
        observed_at=OBSERVED_AT,
        evidence=(f"event:{value}",),
    )


def test_export_is_canonical_sorted_jsonl(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "memory.sqlite") as service:
        service.upsert(("z",), "b", record(2))
        service.upsert(("a", "child"), "c", record(3))
        service.upsert(("a",), "a", record(1))

        first = service.export_jsonl()
        second = service.export_jsonl()

    assert first == second
    assert first.endswith("\n")
    assert " " not in first
    rows = [json.loads(line) for line in first.splitlines()]
    assert [(row["namespace"], row["key"]) for row in rows] == [
        (["a"], "a"),
        (["a", "child"], "c"),
        (["z"], "b"),
    ]
    assert rows[0]["record"]["provenance"]["content_digest"].startswith("sha256:")


def test_export_prefix_and_empty_store(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "memory.sqlite") as service:
        assert service.export_jsonl() == ""
        service.upsert(("one",), "a", record(1))
        service.upsert(("two",), "b", record(2))

        exported = service.export_jsonl(("one",))

    rows = [json.loads(line) for line in exported.splitlines()]
    assert [(row["namespace"], row["key"]) for row in rows] == [(["one"], "a")]
