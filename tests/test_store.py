from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Event
from typing import cast

import pytest
from langgraph.store.base import BaseStore, InvalidNamespaceError, PutOp, TTLConfig
from langgraph.store.sqlite.base import SqliteIndexConfig

from emvr import MemoryRecord, MemoryService

OBSERVED_AT = datetime(2026, 7, 12, 18, 30, tzinfo=UTC)


def record(value: int) -> MemoryRecord:
    return MemoryRecord.create(
        kind="fact",
        content={"value": value},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )


def test_round_trip_upsert_hierarchy_filters_and_native_store(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "memory.sqlite") as service:
        service.upsert(("users", "one"), "preference", record(1))
        service.upsert(("users", "two"), "preference", record(2))
        service.upsert(("teams", "one"), "preference", record(3))
        service.upsert(("users", "one"), "preference", record(4))

        assert isinstance(service.native, BaseStore)
        assert service.get(("users", "one"), "preference") == record(4)
        entry = service.get_entry(("users", "one"), "preference")
        assert entry is not None
        assert entry.created_at.tzinfo is UTC
        assert entry.updated_at.tzinfo is UTC
        assert sorted(
            cast(int, entry.record.content["value"])
            for entry in service.search(("users",))
        ) == [2, 4]
        filtered = service.search(("users",), filter={"content.value": 2})
        assert [(entry.namespace, entry.key) for entry in filtered] == [
            (("users", "two"), "preference")
        ]

        native_record = record(9)
        service.native.put(
            ("native",), "compatible", native_record.to_dict(), index=False
        )
        assert service.get(("native",), "compatible") == native_record


def test_structured_search_preserves_recency_and_all_is_key_sorted(
    tmp_path: Path,
) -> None:
    database = tmp_path / "ordering.sqlite"
    with MemoryService(database) as service:
        service.upsert(("notes",), "a", record(1))
        service.upsert(("notes",), "z", record(2))
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """
                UPDATE store
                SET updated_at = CASE key
                    WHEN 'a' THEN '2025-01-01 00:00:00'
                    ELSE '2026-01-01 00:00:00'
                END
                """
            )
            connection.commit()

        assert [entry.key for entry in service.native.search(("notes",))] == ["z", "a"]
        assert [entry.key for entry in service.search(("notes",))] == ["z", "a"]
        assert [entry.key for entry in service.all(("notes",))] == ["a", "z"]


async def test_structured_filters_preserve_exact_json_types(tmp_path: Path) -> None:
    values: dict[str, object] = {
        "boolean": True,
        "integer": 1,
        "array": [1, 2],
        "large": (1 << 53) + 1,
        "adjacent": 1 << 53,
        "null": None,
        "object": {"b": True, "a": 1},
        "dollar-object": {"a": 2, "$meta": 1},
        "text-a": "a",
        "text-b": "b",
    }
    with MemoryService(tmp_path / "typed-filters.sqlite") as service:
        for key, value in values.items():
            service.upsert(
                ("filters",),
                key,
                MemoryRecord.create(
                    kind="fact",
                    content={"value": value},
                    source="unit-test",
                    observed_at=OBSERVED_AT,
                ),
            )

        assert [
            entry.key
            for entry in service.search(("filters",), filter={"content.value": True})
        ] == ["boolean"]
        assert [
            entry.key
            for entry in service.native.search(
                ("filters",), filter={"content.value": 1}
            )
        ] == ["integer"]
        assert [
            entry.key
            for entry in service.search(
                ("filters",), filter={"content.value": {"$eq": [1, 2]}}
            )
        ] == ["array"]
        assert [
            entry.key
            for entry in await service.native.asearch(
                ("filters",), filter={"content.value": (1 << 53) + 1}
            )
        ] == ["large"]
        assert [
            entry.key
            for entry in service.search(("filters",), filter={"content.value": None})
        ] == ["null"]
        assert [
            entry.key
            for entry in service.search(
                ("filters",), filter={"content.value": {"a": 1, "b": True}}
            )
        ] == ["object"]
        assert [
            entry.key
            for entry in service.search(
                ("filters",),
                filter={"content.value": {"$meta": 1, "a": 2}},
            )
        ] == ["dollar-object"]
        assert service.search(("filters",), filter={"content.missing": None}) == []

        numeric_matches = {
            "$gt": {"adjacent", "large"},
            "$gte": {"adjacent", "large"},
            "$lt": {"integer"},
            "$lte": {"integer"},
        }
        for operator, expected in numeric_matches.items():
            assert {
                entry.key
                for entry in service.search(
                    ("filters",), filter={"content.value": {operator: 2}}
                )
            } == expected

        text_matches = {
            "$gt": {"text-b"},
            "$gte": {"text-a", "text-b"},
            "$lt": {"text-a"},
            "$lte": {"text-a", "text-b"},
        }
        for operator, expected in text_matches.items():
            threshold = "a" if operator in {"$gt", "$gte"} else "b"
            assert {
                entry.key
                for entry in service.search(
                    ("filters",),
                    filter={"content.value": {operator: threshold}},
                    limit=20,
                )
            } == expected

        not_integer = service.search(
            ("filters",), filter={"content.value": {"$ne": 1}}, limit=20
        )
        assert "integer" not in {entry.key for entry in not_integer}
        assert "boolean" in {entry.key for entry in not_integer}
        with pytest.raises(ValueError, match="finite JSON"):
            service.search(("filters",), filter={"content.value": float("nan")})


def test_connection_pragmas_health_and_reopen_migrations(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    with MemoryService(database) as service:
        service.upsert(("test",), "one", record(1))
        health = service.health()
        assert health.status == "ok"
        assert health.journal_mode == "wal"
        assert health.synchronous == 1
        assert health.busy_timeout_ms == 5000
        assert health.foreign_keys is True
        assert health.records == 1

    with MemoryService(database) as reopened:
        assert reopened.get(("test",), "one") == record(1)
        assert reopened.health().records == 1


def test_concurrent_threads_can_initialize_one_new_database(tmp_path: Path) -> None:
    database = tmp_path / "threaded-setup.sqlite"
    barrier = Barrier(8)

    def open_service(_index: int) -> None:
        barrier.wait(timeout=5)
        with MemoryService(database) as service:
            assert service.health().status == "ok"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(open_service, range(8)))


def test_concurrent_processes_can_initialize_one_new_database(tmp_path: Path) -> None:
    database = tmp_path / "process-setup.sqlite"
    marker = tmp_path / "start"
    script = """
import sys
import time
from pathlib import Path
from emvr import MemoryService

database = Path(sys.argv[1])
marker = Path(sys.argv[2])
while not marker.exists():
    time.sleep(0.01)
with MemoryService(database) as service:
    assert service.health().status == "ok"
"""
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("COV_CORE_"):
            del environment[name]
    processes = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", script, str(database), str(marker)],
            cwd=Path(__file__).parents[1],
            env=environment,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _index in range(4)
    ]
    marker.touch()

    failures = [
        stderr
        for process in processes
        if (stderr := process.communicate(timeout=30)[1]) or process.returncode != 0
    ]
    assert failures == []


def test_delete_and_namespace_purge_are_hierarchical(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "memory.sqlite") as service:
        service.upsert(("tenant",), "root", record(1))
        service.upsert(("tenant", "user"), "child", record(2))
        service.upsert(("tenant-other",), "keep-collision", record(3))
        service.upsert(("other",), "keep", record(4))

        assert service.delete(("tenant",), "root") is True
        assert service.delete(("tenant",), "root") is False
        assert service.get(("tenant",), "root") is None
        assert service.purge_namespace(("tenant",)) == 1
        assert service.search(("tenant",)) == []
        assert service.get(("tenant-other",), "keep-collision") == record(3)
        assert service.get(("other",), "keep") == record(4)


def test_namespace_labels_cannot_alias_serialized_paths(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "namespace.sqlite") as service:
        service.upsert(("tenant", "user"), "preference", record(1))

        with pytest.raises(InvalidNamespaceError, match="periods"):
            service.get(("tenant.user",), "preference")
        with pytest.raises(InvalidNamespaceError, match="periods"):
            service.delete(("tenant.user",), "preference")
        with pytest.raises(InvalidNamespaceError, match="langgraph"):
            service.search(("langgraph",))
        with pytest.raises(InvalidNamespaceError, match="cannot be empty"):
            service.native.put((), "invalid", record(1).to_dict())
        with pytest.raises(InvalidNamespaceError, match="empty strings"):
            service.get(("",), "preference")
        with pytest.raises(InvalidNamespaceError, match="must be strings"):
            service.get(cast(tuple[str, ...], (42,)), "preference")
        with pytest.raises(InvalidNamespaceError, match="NUL"):
            service.upsert(("tenant\x00user",), "invalid", record(1))

        assert service.get(("tenant", "user"), "preference") == record(1)


def test_namespace_matching_is_literal_segmented_and_case_sensitive(
    tmp_path: Path,
) -> None:
    with MemoryService(tmp_path / "literal-namespaces.sqlite") as service:
        for index, namespace in enumerate(
            [
                ("tenant%",),
                ("tenantX",),
                ("tenant_",),
                ("tenantA",),
                ("Tenant",),
                ("tenant",),
                ("users",),
                ("users-archive",),
                ("a", "b"),
                ("xa", "b"),
                ("root", "a", "b"),
            ]
        ):
            service.upsert(namespace, "item", record(index))

        for namespace in (("tenant%",), ("tenant_",), ("Tenant",), ("tenant",)):
            assert [entry.namespace for entry in service.search(namespace)] == [
                namespace
            ]
        assert service.purge_namespace(("tenant%",)) == 1
        assert service.get(("tenantX",), "item") == record(1)
        assert service.native.list_namespaces(prefix=("users",)) == [("users",)]
        assert service.native.list_namespaces(suffix=("a", "b")) == [
            ("a", "b"),
            ("root", "a", "b"),
        ]


async def test_async_native_namespace_matching_is_segmented(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "async-namespaces.sqlite") as service:
        service.upsert(("users",), "one", record(1))
        service.upsert(("users-archive",), "two", record(2))

        assert await service.native.alist_namespaces(prefix=("users",)) == [("users",)]


async def test_native_namespace_wildcards_match_one_segment(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "wildcards.sqlite") as service:
        for index, namespace in enumerate(
            [
                ("documents", "a"),
                ("documents", "b"),
                ("documents", "c", "v1"),
                ("packages", "a", "v1"),
                ("packages", "a", "v2"),
            ]
        ):
            service.upsert(namespace, "item", record(index))

        assert service.native.list_namespaces(prefix=("documents", "*")) == [
            ("documents", "a"),
            ("documents", "b"),
            ("documents", "c", "v1"),
        ]
        assert service.native.list_namespaces(suffix=("*", "v1")) == [
            ("documents", "c", "v1"),
            ("packages", "a", "v1"),
        ]
        assert await service.native.alist_namespaces(
            prefix=("documents", "*"), max_depth=2
        ) == [
            ("documents", "a"),
            ("documents", "b"),
            ("documents", "c"),
        ]


def test_fixed_ttl_is_swept_before_native_read(tmp_path: Path) -> None:
    with MemoryService(
        tmp_path / "fixed.sqlite",
        ttl={"default_ttl": 0.001, "refresh_on_read": False},
    ) as service:
        service.upsert(("ttl",), "expired", record(1))
        time.sleep(1.1)

        assert service.native.get(("ttl",), "expired") is None
        assert service.health().records == 0


def test_sliding_ttl_refreshes_only_when_requested(tmp_path: Path) -> None:
    with MemoryService(
        tmp_path / "sliding.sqlite",
        ttl={"default_ttl": 0.05, "refresh_on_read": True},
    ) as service:
        service.upsert(("ttl",), "sliding", record(1))
        time.sleep(0.6)
        assert service.get(("ttl",), "sliding", refresh_ttl=True) == record(1)
        time.sleep(0.6)
        assert service.get(("ttl",), "sliding", refresh_ttl=False) == record(1)
        # Native SQLite timestamps have one-second precision.
        time.sleep(3.6)
        assert service.get(("ttl",), "sliding", refresh_ttl=False) is None


@pytest.mark.parametrize("refresh_ttl", [False, True])
def test_exact_expiry_is_never_returned_or_refreshed(
    tmp_path: Path, refresh_ttl: bool
) -> None:
    database = tmp_path / f"exact-{refresh_ttl}.sqlite"
    with MemoryService(
        database,
        ttl={"default_ttl": 10, "refresh_on_read": True},
    ) as service:
        service.upsert(("ttl",), "boundary", record(1))
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "UPDATE store SET expires_at = CURRENT_TIMESTAMP WHERE key = ?",
                ("boundary",),
            )
            connection.commit()

        assert service.get(("ttl",), "boundary", refresh_ttl=refresh_ttl) is None
        assert service.health().records == 0


def test_fractional_expiry_is_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "fractional.sqlite"
    with MemoryService(database) as service:
        service.upsert(("ttl",), "fractional", record(1), ttl_minutes=10)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """
                UPDATE store
                SET expires_at = STRFTIME(
                    '%Y-%m-%d %H:%M:%f', 'now', '-0.1 seconds'
                )
                WHERE key = 'fractional'
                """
            )
            connection.commit()

        assert service.get(("ttl",), "fractional") is None


def test_fractional_expiry_is_swept_before_namespace_listing(tmp_path: Path) -> None:
    database = tmp_path / "fractional-list.sqlite"
    with MemoryService(database) as service:
        service.upsert(("ttl",), "fractional", record(1), ttl_minutes=10)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """
                UPDATE store
                SET expires_at = STRFTIME(
                    '%Y-%m-%d %H:%M:%f', 'now', '-0.1 seconds'
                )
                """
            )
            connection.commit()

        assert service.list_namespaces() == []
        assert service.health().records == 0

    with MemoryService(database) as service:
        service.upsert(("ttl",), "fractional", record(1), ttl_minutes=10)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """
                UPDATE store
                SET expires_at = STRFTIME(
                    '%Y-%m-%d %H:%M:%f', 'now', '-0.1 seconds'
                )
                """
            )
            connection.commit()

        assert service.sweep_expired() == 1
        assert service.health().records == 0


def test_exhaustive_reads_do_not_skip_rows_when_expiry_changes(tmp_path: Path) -> None:
    database = tmp_path / "stable-all.sqlite"
    payload = "x" * 20_000
    with MemoryService(database) as service:
        for index in range(1_000):
            service.upsert(
                ("records",),
                f"{index:04}",
                MemoryRecord.create(
                    kind="fact",
                    content={"index": index, "payload": payload},
                    source="unit-test",
                    observed_at=OBSERVED_AT,
                ),
            )
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """
                UPDATE store
                SET ttl_minutes = 1,
                    expires_at = STRFTIME(
                        '%Y-%m-%d %H:%M:%f', 'now', '+0.05 seconds'
                    )
                WHERE CAST(key AS INTEGER) < 500
                """
            )
            connection.commit()

        entries = service.all(("records",))
        keys = {entry.key for entry in entries}
        assert {f"{index:04}" for index in range(500, 1_000)} <= keys
        assert '"key":"0999"' in service.export_jsonl(("records",))
        assert service.purge_namespace(("records",)) >= 500
        assert service.health().records == 0


def test_explicit_item_ttl_slides_without_a_default_ttl(tmp_path: Path) -> None:
    database = tmp_path / "explicit-ttl.sqlite"
    with MemoryService(database) as service:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="The default datetime adapter is deprecated.*",
                category=DeprecationWarning,
            )
            service.upsert(("ttl",), "explicit", record(1), ttl_minutes=10)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "UPDATE store SET expires_at = datetime('now', '+1 minute')"
            )
            connection.commit()

        assert service.get(("ttl",), "explicit") == record(1)
        with closing(sqlite3.connect(database)) as connection:
            remaining = connection.execute(
                "SELECT unixepoch(expires_at) - unixepoch() FROM store"
            ).fetchone()
        assert remaining is not None
        assert cast(int, remaining[0]) > 500


def test_explicit_none_default_ttl_disables_expiration(tmp_path: Path) -> None:
    database = tmp_path / "no-default-ttl.sqlite"
    with MemoryService(database, ttl={"default_ttl": None}) as service:
        service.upsert(("ttl",), "persistent", record(1))
        assert service.get(("ttl",), "persistent") == record(1)
        with closing(sqlite3.connect(database)) as connection:
            ttl = connection.execute(
                "SELECT ttl_minutes FROM store WHERE key = 'persistent'"
            ).fetchone()
        assert ttl == (None,)


def test_search_refreshes_only_the_returned_page(tmp_path: Path) -> None:
    database = tmp_path / "search-refresh.sqlite"
    with MemoryService(
        database,
        ttl={"default_ttl": 10, "refresh_on_read": True},
    ) as service:
        service.upsert(("users", "a"), "one", record(1))
        service.upsert(("users", "b"), "two", record(2))
        service.upsert(("users-archive",), "collision", record(3))
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "UPDATE store SET expires_at = datetime('now', '+1 minute')"
            )
            connection.commit()

        results = service.search(("users",), limit=1, refresh_ttl=True)

        assert [(entry.namespace, entry.key) for entry in results] == [
            (("users", "a"), "one")
        ]
        with closing(sqlite3.connect(database)) as connection:
            expirations = dict(
                connection.execute(
                    "SELECT key, unixepoch(expires_at) - unixepoch() FROM store"
                ).fetchall()
            )
        assert cast(int, expirations["one"]) > 500
        assert cast(int, expirations["two"]) < 120
        assert cast(int, expirations["collision"]) < 120


def test_slow_semantic_search_cannot_return_or_revive_expired_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "slow-search.sqlite"
    query_embedded = Event()

    def embed(texts: Sequence[str]) -> list[list[float]]:
        if any("block" in text for text in texts):
            query_embedded.set()
            time.sleep(1.1)
        return [[1.0, 0.0] for _text in texts]

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    with MemoryService(database, index=index) as service:
        service.upsert(("notes",), "expiring", record(1), ttl_minutes=10)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                """
                UPDATE store
                SET expires_at = STRFTIME(
                        '%Y-%m-%d %H:%M:%f', 'now', '+0.01 seconds'
                )
                """
            )
            connection.commit()

        assert service.search(("notes",), query="block") == []
        assert query_embedded.is_set()
        assert service.health().records == 0


def test_semantic_search_rejects_unsupported_deep_windows(tmp_path: Path) -> None:
    with (
        MemoryService(tmp_path / "semantic-window.sqlite") as service,
        pytest.raises(ValueError, match="first 1000"),
    ):
        service.search(("notes",), query="note", limit=1, offset=1_000)

    with (
        MemoryService(tmp_path / "native-window.sqlite") as service,
        pytest.raises(ValueError, match="first 1000"),
    ):
        service.native.search(("notes",), query="note", limit=1, offset=1_000)


async def test_semantic_pagination_matches_native_sync_and_async(
    tmp_path: Path,
) -> None:
    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text == "target" or '"value":1' in text:
                vectors.append([1.0, 0.0])
            elif '"value":2' in text:
                vectors.append([0.9, 0.1])
            else:
                vectors.append([0.8, 0.2])
        return vectors

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    with MemoryService(tmp_path / "pagination.sqlite", index=index) as service:
        for value in (1, 2, 3):
            service.upsert(("notes",), str(value), record(value))

        service_page = service.search(("notes",), query="target", limit=1, offset=2)
        native_page = service.native.search(
            ("notes",), query="target", limit=1, offset=2
        )
        async_page = await service.native.asearch(
            ("notes",), query="target", limit=1, offset=2
        )

        assert [entry.key for entry in service_page] == ["3"]
        assert [entry.key for entry in native_page] == ["3"]
        assert [entry.key for entry in async_page] == ["3"]
        assert len(service.search((), query="target", limit=1)) == 1
        assert [
            entry.key
            for entry in service.search(
                (), query="target", filter={"content.value": 1}, limit=1
            )
        ] == ["1"]


def test_semantic_pagination_refreshes_only_returned_items(tmp_path: Path) -> None:
    database = tmp_path / "semantic-refresh.sqlite"

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if text == "target" or '"value":1' in text
            else [0.9, 0.1]
            if '"value":2' in text
            else [0.8, 0.2]
            for text in texts
        ]

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    with MemoryService(database, index=index) as service:
        for value in (1, 2, 3):
            service.upsert(("notes",), str(value), record(value), ttl_minutes=10)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "UPDATE store SET expires_at = datetime('now', '+1 minute')"
            )
            connection.commit()

        page = service.search(
            ("notes",), query="target", limit=1, offset=1, refresh_ttl=True
        )

        assert [entry.key for entry in page] == ["2"]
        with closing(sqlite3.connect(database)) as connection:
            expirations = dict(
                connection.execute(
                    "SELECT key, unixepoch(expires_at) - unixepoch() FROM store"
                ).fetchall()
            )
        assert cast(int, expirations["1"]) < 120
        assert cast(int, expirations["2"]) > 500
        assert cast(int, expirations["3"]) < 120


def test_semantic_search_deduplicates_before_limiting(tmp_path: Path) -> None:
    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    with MemoryService(tmp_path / "multi-vector.sqlite", index=index) as service:
        for value in (1, 2, 3):
            service.upsert(
                ("notes",),
                str(value),
                MemoryRecord.create(
                    kind="fact",
                    content={"a": value, "b": value, "c": value},
                    source="unit-test",
                    observed_at=OBSERVED_AT,
                ),
                index=["content.a", "content.b", "content.c"],
            )

        assert len(service.search(("notes",), query="target", limit=3)) == 3


def test_global_semantic_fields_select_only_configured_content(tmp_path: Path) -> None:
    embedded: list[str] = []

    def embed(texts: Sequence[str]) -> list[list[float]]:
        embedded.extend(texts)
        return [[1.0, 0.0] for _text in texts]

    index = cast(
        SqliteIndexConfig,
        {"dims": 2, "embed": embed, "fields": ["content.title"]},
    )
    with MemoryService(tmp_path / "fields.sqlite", index=index) as service:
        service.upsert(
            ("notes",),
            "one",
            MemoryRecord.create(
                kind="note",
                content={"title": "indexed title", "body": "not indexed"},
                source="unit-test",
                observed_at=OBSERVED_AT,
            ),
        )

    assert embedded == ["indexed title"]


def test_upsert_preserves_creation_time_and_replaces_old_vectors(
    tmp_path: Path,
) -> None:
    database = tmp_path / "upsert.sqlite"

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    with MemoryService(database, index=index) as service:
        service.upsert(("notes",), "one", record(1))
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("UPDATE store SET created_at = '2000-01-01 00:00:00'")
            connection.commit()

    with MemoryService(database) as service:
        service.upsert(("notes",), "one", record(2))

        entry = service.get_entry(("notes",), "one")
        assert entry is not None
        assert entry.created_at == datetime(2000, 1, 1, tzinfo=UTC)

    with MemoryService(database, index=index) as service:
        with closing(sqlite3.connect(database)) as connection:
            vector_count = connection.execute(
                "SELECT COUNT(*) FROM store_vectors"
            ).fetchone()
        assert vector_count == (0,)
        assert service.search(("notes",), query="old") == []


async def test_empty_query_uses_structured_search_without_an_index(
    tmp_path: Path,
) -> None:
    with MemoryService(tmp_path / "empty-query.sqlite") as service:
        service.upsert(("notes",), "one", record(1))

        assert [entry.key for entry in service.search(("notes",), query="")] == ["one"]
        assert [entry.key for entry in service.native.search(("notes",), query="")] == [
            "one"
        ]
        assert [
            entry.key for entry in await service.native.asearch(("notes",), query="")
        ] == ["one"]


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(-1, 0), (0, 0), (True, 0), (1, -1), (1, True)],
)
async def test_native_pagination_rejects_invalid_windows(
    tmp_path: Path, limit: object, offset: object
) -> None:
    with MemoryService(tmp_path / f"pagination-{limit}-{offset}.sqlite") as service:
        with pytest.raises(ValueError):
            service.native.search(
                ("notes",),
                limit=cast(int, limit),
                offset=cast(int, offset),
            )
        with pytest.raises(ValueError):
            await service.native.asearch(
                ("notes",),
                limit=cast(int, limit),
                offset=cast(int, offset),
            )
        with pytest.raises(ValueError):
            service.native.list_namespaces(
                limit=cast(int, limit),
                offset=cast(int, offset),
            )


@pytest.mark.parametrize("distance_type", ["l2", "inner_product"])
def test_unsupported_semantic_distances_fail_closed(
    tmp_path: Path, distance_type: str
) -> None:
    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    index = cast(
        SqliteIndexConfig,
        {"dims": 2, "embed": embed, "distance_type": distance_type},
    )
    with pytest.raises(ValueError, match="cosine"):
        MemoryService(tmp_path / f"{distance_type}.sqlite", index=index)


@pytest.mark.parametrize("ttl_minutes", [-1, 0, float("nan"), float("inf"), True])
def test_invalid_item_ttl_is_rejected_before_write(
    tmp_path: Path, ttl_minutes: object
) -> None:
    with MemoryService(tmp_path / "invalid-ttl.sqlite") as service:
        with pytest.raises(ValueError, match="ttl_minutes"):
            service.upsert(
                ("ttl",),
                "invalid",
                record(1),
                ttl_minutes=cast(float, ttl_minutes),
            )
        assert service.health().records == 0


def test_maximum_item_ttl_is_accepted(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "maximum-ttl.sqlite") as service:
        service.upsert(
            ("ttl",),
            "maximum",
            record(1),
            ttl_minutes=5_270_400,
        )
        assert service.get(("ttl",), "maximum") == record(1)


def test_native_integer_boundaries_are_storable(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "integer-boundaries.sqlite") as service:
        for key, value in (("minimum", -(1 << 63)), ("maximum", (1 << 64) - 1)):
            boundary = record(value)
            service.upsert(("integers",), key, boundary)
            assert service.get(("integers",), key) == boundary


@pytest.mark.parametrize(
    "ttl_minutes",
    [-1, 0, float("nan"), float("inf"), True, 5_270_401],
)
async def test_invalid_native_ttl_is_rejected_before_write(
    tmp_path: Path, ttl_minutes: object
) -> None:
    with MemoryService(
        tmp_path / f"invalid-native-ttl-{ttl_minutes}.sqlite"
    ) as service:
        value = record(1).to_dict()
        with pytest.raises(ValueError, match="ttl_minutes"):
            service.native.put(("ttl",), "sync", value, ttl=cast(float, ttl_minutes))
        with pytest.raises(ValueError, match="ttl_minutes"):
            await service.native.aput(
                ("ttl",), "async", value, ttl=cast(float, ttl_minutes)
            )
        with pytest.raises(ValueError, match="ttl_minutes"):
            service.native.batch(
                [
                    PutOp(
                        ("ttl",),
                        "batch",
                        cast(dict[str, object], value),
                        ttl=cast(float, ttl_minutes),
                    )
                ]
            )
        assert service.health().records == 0


@pytest.mark.parametrize("default_ttl", [-1, 0, float("nan"), float("inf"), True])
def test_invalid_default_ttl_is_rejected(tmp_path: Path, default_ttl: object) -> None:
    config = cast(TTLConfig, {"default_ttl": default_ttl})
    with pytest.raises(ValueError, match=r"ttl\.default_ttl"):
        MemoryService(tmp_path / "invalid-default-ttl.sqlite", ttl=config)


def test_invalid_pydantic_copy_cannot_reach_sqlite(tmp_path: Path) -> None:
    invalid = record(1).model_copy(update={"schema_version": 2})
    with MemoryService(tmp_path / "invalid-copy.sqlite") as service:
        with pytest.raises(ValueError):
            service.upsert(("records",), "invalid", invalid)
        assert service.health().records == 0


def test_concurrent_writes_share_the_native_serialization_lock(tmp_path: Path) -> None:
    with MemoryService(tmp_path / "concurrent.sqlite") as service:

        def write(value: int) -> None:
            service.upsert(("concurrent",), str(value), record(value))
            assert service.get(("concurrent",), str(value)) == record(value)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(32)))

        entries = service.search(("concurrent",), limit=100)
        assert {cast(int, entry.record.content["value"]) for entry in entries} == set(
            range(32)
        )


def test_service_lifecycle_and_pagination_validation(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "lifecycle.sqlite")
    assert service.__enter__() is service
    assert service.purge_namespace(("empty",)) == 0
    with pytest.raises(ValueError, match="limit"):
        service.search(limit=0)
    with pytest.raises(ValueError, match="offset"):
        service.search(offset=-1)
    with pytest.raises(ValueError, match="limit"):
        service.list_namespaces(limit=0)
    with pytest.raises(ValueError, match="offset"):
        service.list_namespaces(offset=-1)
    service.__exit__(None, None, None)
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        service.get(("test",), "key")


def test_service_closes_connection_when_native_configuration_fails(
    tmp_path: Path,
) -> None:
    incomplete_index = cast(SqliteIndexConfig, {"dims": 2})
    with pytest.raises(ValueError, match="embed"):
        MemoryService(tmp_path / "invalid.sqlite", index=incomplete_index)


def test_close_waits_for_an_in_flight_search(tmp_path: Path) -> None:
    search_started = Event()
    release_search = Event()

    def embed(texts: Sequence[str]) -> list[list[float]]:
        if list(texts) == ["block"]:
            search_started.set()
            assert release_search.wait(timeout=5)
        return [[1.0, 0.0] for _text in texts]

    index: SqliteIndexConfig = {"dims": 2, "embed": embed}
    service = MemoryService(
        tmp_path / "close.sqlite",
        index=index,
    )
    service.upsert(("notes",), "one", record(1))
    with ThreadPoolExecutor(max_workers=2) as executor:
        native = service.native
        search = executor.submit(native.search, ("notes",), query="block")
        assert search_started.wait(timeout=5)
        close = executor.submit(service.close)
        assert not close.done()
        release_search.set()
        assert search.result(timeout=5)[0].key == "one"
        close.result(timeout=5)
