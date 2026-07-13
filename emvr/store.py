"""Native LangGraph SQLite storage for enhanced memory records."""

# ruff: noqa: S608 - variable-length IN lists use generated placeholders only

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, Literal, cast

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.base import PreparedGetQuery, SqliteIndexConfig

from emvr.models import (
    SCHEMA_VERSION,
    HealthReport,
    MemoryEntry,
    MemoryRecord,
    Namespace,
    _canonical_json,  # pyright: ignore[reportPrivateUsage]
)


class SemanticIndexDisabledError(RuntimeError):
    """Raised when semantic search is requested without an index."""


MAX_SEMANTIC_RESULTS = 1_000
MAX_TTL_MINUTES = 60.0 * 24 * 366 * 10
_LIVE_SQL = "(expires_at IS NULL OR julianday(expires_at) > julianday('now'))"
_LIVE_SQL_ALIAS = "(s.expires_at IS NULL OR julianday(s.expires_at) > julianday('now'))"
_PREFIX_SQL = "(prefix = ? OR substr(prefix, 1, length(?) + 1) = ? || '.')"
_PREFIX_SQL_ALIAS = "(s.prefix = ? OR substr(s.prefix, 1, length(?) + 1) = ? || '.')"
_SETUP_ATTEMPTS = 8
_ALL_RESULTS_LIMIT = 2_147_483_647
_FILTER_OPERATORS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"})


def validate_ttl_minutes(value: object, *, name: str = "ttl_minutes") -> float:
    """Return one finite, positive TTL within the supported ten-year bound."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= MAX_TTL_MINUTES
    ):
        raise ValueError(
            f"{name} must be greater than 0 and at most {MAX_TTL_MINUTES:g}"
        )
    return float(value)


def _validate_namespace(namespace: Namespace, *, allow_empty: bool = False) -> None:
    if not namespace:
        if allow_empty:
            return
        raise InvalidNamespaceError("Namespace cannot be empty.")
    for label in cast(tuple[object, ...], namespace):
        if not isinstance(label, str):
            raise InvalidNamespaceError("Namespace labels must be strings.")
        if not label:
            raise InvalidNamespaceError("Namespace labels cannot be empty strings.")
        if "." in label:
            raise InvalidNamespaceError(
                "Namespace labels cannot contain periods ('.')."
            )
        if "\x00" in label:
            raise InvalidNamespaceError("Namespace labels cannot contain NUL bytes.")
    if namespace[0] == "langgraph":
        raise InvalidNamespaceError('Root namespace label cannot be "langgraph".')


def _namespace_text(namespace: Namespace) -> str:
    return ".".join(namespace)


def _validate_pagination(limit: object, offset: object) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")


def _json_filter(
    document: str | bytes,
    path: str,
    operator: str,
    expected_json: str,
) -> int:
    matched = False
    try:
        actual = cast(object, json.loads(document))
        for segment in path.split("."):
            if not isinstance(actual, dict) or segment not in actual:
                break
            actual = cast(dict[str, object], actual)[segment]
        else:
            expected = cast(object, json.loads(expected_json))
            if operator in {"$eq", "$ne"}:
                equal = _canonical_json(actual).decode() == expected_json
                matched = equal if operator == "$eq" else not equal
            elif type(actual) in (int, float) and type(expected) in (int, float):
                left_number = cast(int | float, actual)
                right_number = cast(int | float, expected)
                matched = {
                    "$gt": left_number > right_number,
                    "$gte": left_number >= right_number,
                    "$lt": left_number < right_number,
                    "$lte": left_number <= right_number,
                }.get(operator, False)
            elif isinstance(actual, str) and isinstance(expected, str):
                matched = {
                    "$gt": actual > expected,
                    "$gte": actual >= expected,
                    "$lt": actual < expected,
                    "$lte": actual <= expected,
                }.get(operator, False)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError, ValueError):
        pass
    return int(matched)


def _namespace_match(namespace: str, pattern: str, mode: str) -> int:
    namespace_parts = namespace.split(".")
    pattern_parts = pattern.split(".")
    if mode == "prefix":
        selected = namespace_parts[: len(pattern_parts)]
    else:
        selected = namespace_parts[-len(pattern_parts) :]
    return int(
        len(selected) == len(pattern_parts)
        and all(
            expected in ("*", actual)
            for actual, expected in zip(selected, pattern_parts, strict=True)
        )
    )


def _normalize_index(index: SqliteIndexConfig | None) -> SqliteIndexConfig | None:
    if index is None:
        return None
    normalized = cast(dict[str, Any], dict(index))
    fields = normalized.pop("fields", None)
    if fields is not None:
        normalized["text_fields"] = fields
    if normalized.get("distance_type", "cosine") != "cosine":
        raise ValueError("Enhanced Memory supports cosine semantic distance only")
    return cast(SqliteIndexConfig, normalized)


def _retryable_setup_error(error: sqlite3.DatabaseError) -> bool:
    message = str(error).lower()
    return any(
        fragment in message
        for fragment in (
            "database is locked",
            "database schema is locked",
            "duplicate column name",
            "unique constraint failed: store_migrations.v",
            "unique constraint failed: vector_migrations.v",
        )
    )


class _SweepingSqliteStore(SqliteStore):
    """SqliteStore with precise fail-closed reads and true tuple namespaces."""

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        materialized = list(ops)
        self._validate_ops(materialized)
        if self.index_config is None and any(
            isinstance(op, SearchOp) and bool(op.query) for op in materialized
        ):
            raise SemanticIndexDisabledError(
                "semantic search requires an index configuration"
            )
        if any(
            isinstance(op, (GetOp, SearchOp, ListNamespacesOp)) for op in materialized
        ):
            self.sweep_ttl()
        safe_ops = [
            op._replace(refresh_ttl=False)
            if isinstance(op, GetOp)
            else op._replace(
                limit=op.offset + op.limit if op.query else op.limit,
                offset=0 if op.query else op.offset,
                refresh_ttl=False,
            )
            if isinstance(op, SearchOp)
            else op
            for op in materialized
        ]
        results = super().batch(safe_ops)
        for index, op in enumerate(materialized):
            if isinstance(op, SearchOp) and op.query:
                items = cast(list[SearchItem], results[index])
                results[index] = items[op.offset : op.offset + op.limit]
        return self._finalize_reads(materialized, results)

    def sweep_ttl(self) -> int:
        """Delete rows that have reached, not only passed, their expiry second."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL
                  AND julianday(expires_at) <= julianday('now')
                """
            )
            return cursor.rowcount

    def _get_batch_GET_ops_queries(
        self, get_ops: Sequence[tuple[int, GetOp]]
    ) -> list[PreparedGetQuery]:
        queries = super()._get_batch_GET_ops_queries(get_ops)
        return [
            query._replace(
                query=query.query.replace(
                    "WHERE prefix = ?",
                    f"WHERE {_LIVE_SQL} AND prefix = ?",
                    1,
                )
            )
            if query.kind == "get"
            else query
            for query in queries
        ]

    def _prepare_batch_PUT_queries(
        self, put_ops: Sequence[tuple[int, PutOp]]
    ) -> tuple[
        list[tuple[str, Sequence[object]]],
        tuple[str, Sequence[tuple[str, str, str, str]]] | None,
    ]:
        inherited = cast(
            tuple[
                list[tuple[str, Sequence[object]]],
                tuple[str, Sequence[tuple[str, str, str, str]]] | None,
            ],
            super()._prepare_batch_PUT_queries(  # pyright: ignore[reportUnknownMemberType]
                put_ops
            ),
        )
        queries, embedding_request = inherited
        prepared: list[tuple[str, Sequence[object]]] = []
        marker = "INSERT OR REPLACE INTO store ("
        for query, inherited_params in queries:
            params = [
                value.isoformat(sep=" ") if isinstance(value, datetime) else value
                for value in inherited_params
            ]
            if marker not in query:
                prepared.append((query, params))
                continue
            vector_table_exists = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'store_vectors'"
            ).fetchone()
            if vector_table_exists is not None:
                pairs = [
                    (params[offset], params[offset + 1])
                    for offset in range(0, len(params), 5)
                ]
                prepared.append(
                    (
                        "DELETE FROM store_vectors WHERE (prefix, key) IN ("
                        + ",".join("(?, ?)" for _pair in pairs)
                        + ")",
                        [part for pair in pairs for part in pair],
                    )
                )
            upsert_query = query.replace(
                "INSERT OR REPLACE INTO store", "INSERT INTO store", 1
            )
            upsert_query += """
                ON CONFLICT(prefix, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at,
                    ttl_minutes = excluded.ttl_minutes
            """
            prepared.append((upsert_query, params))
        return prepared, embedding_request

    def _get_filter_condition(
        self, key: str, op: str, value: object
    ) -> tuple[str, list[object]]:
        if not key or op not in _FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter path or operator: {key!r} {op!r}")
        try:
            expected = _canonical_json(value).decode()
        except (TypeError, ValueError) as exc:
            raise ValueError("filter values must be finite JSON values") from exc
        return "emvr_json_filter(value, ?, ?, ?) = 1", [key, op, expected]

    def _prepare_batch_search_queries(
        self, search_ops: Sequence[tuple[int, SearchOp]]
    ) -> tuple[
        list[tuple[str, list[None | str | list[float]], bool]],
        list[tuple[int, str]],
    ]:
        normalized_ops: list[tuple[int, SearchOp]] = []
        for index, op in search_ops:
            if not op.filter:
                normalized_ops.append((index, op))
                continue
            normalized_filter: dict[str, object] = {}
            for key, value in cast(dict[str, object], op.filter).items():
                is_operator = (
                    isinstance(value, dict)
                    and bool(cast(dict[object, object], value))
                    and all(
                        candidate in _FILTER_OPERATORS
                        for candidate in cast(dict[object, object], value)
                    )
                )
                normalized_filter[key] = value if is_operator else {"$eq": value}
            normalized_ops.append((index, op._replace(filter=normalized_filter)))
        queries, embeddings = super()._prepare_batch_search_queries(normalized_ops)
        prepared: list[tuple[str, list[None | str | list[float]], bool]] = []
        vector_join = "JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key"
        for (_, op), (raw_query, raw_params, _needs_refresh) in zip(
            search_ops, queries, strict=True
        ):
            query = raw_query
            params = raw_params.copy()
            if op.query and self.index_config:
                before, marker, after = query.partition(vector_join)
                if op.namespace_prefix:
                    query = query.replace(
                        "WHERE s.prefix LIKE ?",
                        f"WHERE {_LIVE_SQL_ALIAS} AND {_PREFIX_SQL_ALIAS}",
                        1,
                    )
                    prefix = _namespace_text(op.namespace_prefix)
                    params[1:2] = [prefix, prefix, prefix]
                elif "WHERE " in after.partition("ORDER BY score")[0]:
                    after = after.replace("WHERE ", f"WHERE {_LIVE_SQL_ALIAS} AND ", 1)
                    query = before + marker + after
                else:
                    query = (
                        before
                        + marker
                        + after.replace(
                            "ORDER BY score",
                            f"WHERE {_LIVE_SQL_ALIAS}\nORDER BY score",
                            1,
                        )
                    )
            elif op.namespace_prefix:
                query = query.replace(
                    "WHERE prefix LIKE ?",
                    f"WHERE {_LIVE_SQL} AND {_PREFIX_SQL}",
                    1,
                )
                prefix = _namespace_text(op.namespace_prefix)
                params[0:1] = [prefix, prefix, prefix]
            else:
                query = query.replace(
                    "WHERE prefix LIKE ?",
                    f"WHERE {_LIVE_SQL} AND prefix LIKE ?",
                    1,
                )
            if op.query and self.index_config:
                vector_source = query[query.index("FROM store s") :]
                vector_source = vector_source[
                    : vector_source.index("ORDER BY score DESC")
                ]
                query = f"""
                    WITH scored AS (
                        SELECT s.prefix, s.key, s.value, s.created_at,
                            s.updated_at, s.expires_at, s.ttl_minutes,
                            MAX(1.0 - vec_distance_cosine(sv.embedding, ?)) AS score
                        {vector_source}
                        GROUP BY s.prefix, s.key
                    )
                    SELECT prefix, key, value, created_at, updated_at,
                        expires_at, ttl_minutes, score
                    FROM scored
                    ORDER BY score DESC
                    LIMIT ? OFFSET ?
                """
                del params[-3]
            prepared.append((query, params, False))
        return prepared, embeddings

    def _get_batch_list_namespaces_queries(
        self, list_ops: Sequence[tuple[int, ListNamespacesOp]]
    ) -> list[tuple[str, Sequence[object]]]:
        inherited = cast(
            list[tuple[str, Sequence[object]]],
            super()._get_batch_list_namespaces_queries(  # pyright: ignore[reportUnknownMemberType]
                list_ops
            ),
        )
        prepared: list[tuple[str, Sequence[object]]] = []
        for (_, op), (raw_query, inherited_params) in zip(
            list_ops, inherited, strict=True
        ):
            query = raw_query
            clauses = [_LIVE_SQL]
            params: list[object] = []
            conditions = op.match_conditions or ()
            for condition in conditions:
                path = _namespace_text(condition.path)
                if condition.match_type == "prefix":
                    clauses.append("emvr_namespace_match(prefix, ?, 'prefix') = 1")
                else:
                    clauses.append("emvr_namespace_match(prefix, ?, 'suffix') = 1")
                params.append(path)
            inherited_where = "WHERE " + " AND ".join(
                "prefix LIKE ?" for _condition in conditions
            )
            replacement = "WHERE " + " AND ".join(clauses)
            if conditions:
                query = query.replace(inherited_where, replacement, 1)
            else:
                query = query.replace("FROM store", f"FROM store WHERE {_LIVE_SQL}", 1)
            tail_size = 4 if op.max_depth is not None else 2
            params.extend(inherited_params[-tail_size:])
            prepared.append((query, params))
        return prepared

    def _validate_ops(self, ops: Sequence[Op]) -> None:
        for op in ops:
            if isinstance(op, PutOp):
                _validate_namespace(op.namespace)
                if op.ttl is not None:
                    validate_ttl_minutes(op.ttl)
            elif isinstance(op, GetOp):
                _validate_namespace(op.namespace)
            elif isinstance(op, SearchOp):
                _validate_namespace(op.namespace_prefix, allow_empty=True)
                _validate_pagination(op.limit, op.offset)
                if op.query and op.offset + op.limit > MAX_SEMANTIC_RESULTS:
                    raise ValueError(
                        f"semantic search supports the first "
                        f"{MAX_SEMANTIC_RESULTS} results"
                    )
            else:
                _validate_pagination(op.limit, op.offset)
                if op.match_conditions:
                    for condition in op.match_conditions:
                        _validate_namespace(condition.path, allow_empty=True)

    def _finalize_reads(self, ops: Sequence[Op], results: list[Result]) -> list[Result]:
        protected = {(op.namespace, op.key) for op in ops if isinstance(op, PutOp)}
        candidates = {
            (item.namespace, item.key)
            for result in results
            for item in (
                [result]
                if isinstance(result, Item)
                else result
                if isinstance(result, list)
                and all(isinstance(value, Item) for value in result)
                else []
            )
            if isinstance(item, Item) and (item.namespace, item.key) not in protected
        }
        if not candidates:
            return results
        with self._cursor() as cursor:
            live_ttls = self._live_ttls(cursor, candidates)
            refresh_pairs = {
                (item.namespace, item.key)
                for op, result in zip(ops, results, strict=True)
                if isinstance(op, (GetOp, SearchOp)) and op.refresh_ttl
                for item in (
                    [result]
                    if isinstance(result, Item)
                    else result
                    if isinstance(result, list)
                    and all(isinstance(value, Item) for value in result)
                    else []
                )
                if isinstance(item, Item)
                and live_ttls.get((item.namespace, item.key)) is not None
            }
            refreshed = self._refresh_pairs(cursor, refresh_pairs)
            for index, (op, result) in enumerate(zip(ops, results, strict=True)):
                if isinstance(op, GetOp) and isinstance(result, Item):
                    pair = (result.namespace, result.key)
                    ttl = live_ttls.get(pair)
                    if pair not in protected and (
                        pair not in live_ttls
                        or (
                            op.refresh_ttl and ttl is not None and pair not in refreshed
                        )
                    ):
                        results[index] = None
                elif isinstance(op, SearchOp) and isinstance(result, list):
                    items = cast(list[SearchItem], result)
                    results[index] = [
                        item
                        for item in items
                        if (item.namespace, item.key) in protected
                        or (
                            (item.namespace, item.key) in live_ttls
                            and (
                                not op.refresh_ttl
                                or live_ttls[(item.namespace, item.key)] is None
                                or (item.namespace, item.key) in refreshed
                            )
                        )
                    ]
        return results

    def _live_ttls(
        self, cursor: sqlite3.Cursor, pairs: set[tuple[Namespace, str]]
    ) -> dict[tuple[Namespace, str], float | None]:
        serialized = [(_namespace_text(namespace), key) for namespace, key in pairs]
        live: dict[tuple[Namespace, str], float | None] = {}
        for start in range(0, len(serialized), 400):
            chunk = serialized[start : start + 400]
            placeholders = ",".join("(?, ?)" for _pair in chunk)
            query = f"""
                SELECT prefix, key, ttl_minutes
                FROM store
                WHERE (prefix, key) IN ({placeholders}) AND {_LIVE_SQL}
                """
            rows = cursor.execute(
                query,
                [part for pair in chunk for part in pair],
            ).fetchall()
            live.update(
                {
                    (tuple(prefix.split(".")), key): cast(float | None, ttl)
                    for prefix, key, ttl in rows
                }
            )
        return live

    def _refresh_pairs(
        self, cursor: sqlite3.Cursor, pairs: set[tuple[Namespace, str]]
    ) -> set[tuple[Namespace, str]]:
        serialized = [(_namespace_text(namespace), key) for namespace, key in pairs]
        refreshed: set[tuple[Namespace, str]] = set()
        for start in range(0, len(serialized), 400):
            chunk = serialized[start : start + 400]
            placeholders = ",".join("(?, ?)" for _pair in chunk)
            query = f"""
                UPDATE store
                SET expires_at = STRFTIME(
                    '%Y-%m-%d %H:%M:%f', 'now', '+' || ttl_minutes || ' minutes'
                )
                WHERE ttl_minutes IS NOT NULL
                  AND {_LIVE_SQL}
                  AND (prefix, key) IN ({placeholders})
                RETURNING prefix, key
                """
            rows = cursor.execute(
                query,
                [part for pair in chunk for part in pair],
            ).fetchall()
            refreshed.update((tuple(prefix.split(".")), key) for prefix, key in rows)
        return refreshed


class _GuardedStore(BaseStore):
    """Expose native LangGraph operations through the service lifecycle gate."""

    supports_ttl = True

    def __init__(self, owner: MemoryService, ttl: TTLConfig) -> None:
        self._owner = owner
        self.ttl_config = ttl

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._owner._batch_native(  # pyright: ignore[reportPrivateUsage]
            ops
        )

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(
            self._owner._batch_native,  # pyright: ignore[reportPrivateUsage]
            list(ops),
        )


def _entry_from_item(item: Item | SearchItem) -> MemoryEntry:
    def as_utc(value: datetime) -> datetime:
        return (
            value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        )

    return MemoryEntry(
        namespace=item.namespace,
        key=item.key,
        record=MemoryRecord.from_dict(item.value),
        created_at=as_utc(item.created_at),
        updated_at=as_utc(item.updated_at),
        score=cast(float | None, getattr(item, "score", None)),
    )


class MemoryService:
    """Typed owner of one native ``SqliteStore`` and SQLite connection."""

    def __init__(
        self,
        database: str | Path,
        *,
        ttl: TTLConfig | None = None,
        index: SqliteIndexConfig | None = None,
    ) -> None:
        """Open, configure, and migrate one SQLite database."""
        self._closed = False
        self._lifecycle_lock = RLock()
        effective_ttl: TTLConfig = {"refresh_on_read": True, **(ttl or {})}
        default_ttl = effective_ttl.get("default_ttl")
        if default_ttl is not None:
            effective_ttl["default_ttl"] = validate_ttl_minutes(
                default_ttl, name="ttl.default_ttl"
            )
        self._connection = sqlite3.connect(
            str(database),
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        try:
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.create_function(
                "emvr_namespace_match",
                3,
                _namespace_match,
                deterministic=True,
            )
            self._connection.create_function(
                "emvr_json_filter",
                4,
                _json_filter,
                deterministic=True,
            )
            self._native = _SweepingSqliteStore(
                self._connection,
                ttl=effective_ttl,
                index=_normalize_index(index),
            )
            for attempt in range(_SETUP_ATTEMPTS):
                try:
                    self._connection.execute("PRAGMA journal_mode=WAL")
                    self._connection.execute("PRAGMA synchronous=NORMAL")
                    self._connection.execute("PRAGMA foreign_keys=ON")
                    self._native.setup()
                    break
                except sqlite3.DatabaseError as error:
                    if (
                        not _retryable_setup_error(error)
                        or attempt == _SETUP_ATTEMPTS - 1
                    ):
                        raise
                    if self._connection.in_transaction:
                        self._connection.rollback()
                    time.sleep(0.05 * (attempt + 1))
            self._guarded = _GuardedStore(self, effective_ttl)
        except BaseException:
            self._connection.close()
            raise

    @property
    def native(self) -> BaseStore:
        """Return the native LangGraph store for graph/runtime integration."""
        with self._lifecycle_lock:
            self._require_open()
            return self._guarded

    def __enter__(self) -> MemoryService:
        """Return this open service."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store when leaving a context."""
        self.close()

    def close(self) -> None:
        """Stop native background work and close the connection."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def upsert(
        self,
        namespace: Namespace,
        key: str,
        record: MemoryRecord,
        *,
        ttl_minutes: float | None | NotProvided = NOT_PROVIDED,
        index: Literal[False] | list[str] | None = None,
    ) -> None:
        """Insert or replace one typed memory record."""
        with self._operation():
            self._native.put(
                namespace,
                key,
                cast(dict[str, Any], record.to_dict()),
                index=index,
                ttl=ttl_minutes,
            )

    def get(
        self,
        namespace: Namespace,
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> MemoryRecord | None:
        """Return one unexpired record by exact namespace and key."""
        entry = self.get_entry(namespace, key, refresh_ttl=refresh_ttl)
        return None if entry is None else entry.record

    def get_entry(
        self,
        namespace: Namespace,
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> MemoryEntry | None:
        """Return one unexpired record with its native identity and timestamps."""
        with self._operation():
            item = self._native.get(namespace, key, refresh_ttl=refresh_ttl)
            return None if item is None else _entry_from_item(item)

    def search(  # noqa: PLR0913
        self,
        namespace_prefix: Namespace = (),
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[MemoryEntry]:
        """Search records under a true hierarchical namespace prefix."""
        with self._operation():
            return [
                _entry_from_item(item)
                for item in self._native.search(
                    namespace_prefix,
                    query=query,
                    filter=filter,
                    limit=limit,
                    offset=offset,
                    refresh_ttl=refresh_ttl,
                )
            ]

    def delete(self, namespace: Namespace, key: str) -> bool:
        """Atomically delete one exact native key and report whether it existed."""
        with self._operation():
            result = self._native.batch(
                [
                    GetOp(namespace, key, False),
                    PutOp(namespace, key, None, ttl=None),
                ]
            )
            return result[0] is not None

    def purge_namespace(self, namespace_prefix: Namespace) -> int:
        """Delete every record at or below a hierarchical prefix."""
        with self._operation():
            entries = self.all(namespace_prefix)
            if entries:
                self._native.batch(
                    PutOp(entry.namespace, entry.key, None, ttl=None)
                    for entry in entries
                )
            return len(entries)

    def all(self, namespace_prefix: Namespace = ()) -> list[MemoryEntry]:
        """Return all records under a prefix in deterministic key order."""
        with self._operation():
            entries = [
                _entry_from_item(item)
                for item in self._native.search(
                    namespace_prefix,
                    limit=_ALL_RESULTS_LIMIT,
                    refresh_ttl=False,
                )
            ]
            return sorted(entries, key=lambda entry: (entry.namespace, entry.key))

    def list_namespaces(
        self,
        prefix: Namespace | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Namespace]:
        """List true hierarchical namespaces with bounded pagination."""
        with self._operation():
            return self._native.list_namespaces(
                prefix=prefix,
                limit=limit,
                offset=offset,
            )

    def export_jsonl(self, namespace_prefix: Namespace = ()) -> str:
        """Return canonical JSON Lines sorted by namespace and key."""
        with self._operation():
            lines = [
                json.dumps(
                    {
                        "key": entry.key,
                        "namespace": list(entry.namespace),
                        "record": entry.record.to_dict(),
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for entry in self.all(namespace_prefix)
            ]
            return "" if not lines else "\n".join(lines) + "\n"

    def sweep_expired(self) -> int:
        """Delete every record that has reached its expiry timestamp."""
        with self._operation():
            return self._native.sweep_ttl()

    def health(self) -> HealthReport:
        """Verify the connection and return its required runtime settings."""
        with self._operation():
            self._native.sweep_ttl()
            with self._native.lock:
                records = cast(
                    tuple[int],
                    self._connection.execute("SELECT COUNT(*) FROM store").fetchone(),
                )[0]
                journal_mode = cast(
                    tuple[str],
                    self._connection.execute("PRAGMA journal_mode").fetchone(),
                )[0]
                synchronous = cast(
                    tuple[int],
                    self._connection.execute("PRAGMA synchronous").fetchone(),
                )[0]
                busy_timeout = cast(
                    tuple[int],
                    self._connection.execute("PRAGMA busy_timeout").fetchone(),
                )[0]
                foreign_keys = cast(
                    tuple[int],
                    self._connection.execute("PRAGMA foreign_keys").fetchone(),
                )[0]
            return HealthReport(
                status="ok",
                schema_version=SCHEMA_VERSION,
                journal_mode=journal_mode.lower(),
                synchronous=synchronous,
                busy_timeout_ms=busy_timeout,
                foreign_keys=bool(foreign_keys),
                records=records,
            )

    def _batch_native(self, ops: Iterable[Op]) -> list[Result]:
        with self._operation():
            return self._native.batch(ops)

    @contextmanager
    def _operation(self) -> Generator[None]:
        with self._lifecycle_lock:
            self._require_open()
            yield

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("memory store is closed")
