"""Typed, versioned memory records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

JsonObject: TypeAlias = dict[str, JsonValue]
Namespace: TypeAlias = tuple[str, ...]

SCHEMA_VERSION = 1
_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = 64
_MIN_SERIALIZABLE_INTEGER = -(1 << 63)
_MAX_SERIALIZABLE_INTEGER = (1 << 64) - 1

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
ContentDigest = Annotated[
    str,
    Field(pattern=rf"^{_DIGEST_PREFIX}[0-9a-f]{{{_DIGEST_LENGTH}}}$"),
]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("content must be a JSON object with finite values") from exc


def _validate_json_tree(value: object, *, name: str) -> None:
    if type(value) is int and not (
        _MIN_SERIALIZABLE_INTEGER <= value <= _MAX_SERIALIZABLE_INTEGER
    ):
        raise ValueError("content integers must fit the native 64-bit storage range")
    if isinstance(value, dict):
        raw = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in raw):
            raise ValueError(f"{name} must use string keys at every depth")
        for item in raw.values():
            _validate_json_tree(item, name=name)
    elif isinstance(value, (list, tuple)):
        for item in cast(Sequence[object], value):
            _validate_json_tree(item, name=name)


def _normalize_object(value: object, *, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object with string keys")
    raw = cast(dict[object, object], value)
    _validate_json_tree(raw, name=name)
    normalized = cast(JsonObject, json.loads(_canonical_json(raw)))
    return normalized


def content_digest(content: object) -> str:
    """Return the stable SHA-256 digest of canonical JSON content."""
    normalized = _normalize_object(content, name="content")
    return _DIGEST_PREFIX + hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_StrictModel):
    """Evidence identifying where and when memory content was observed."""

    source: NonEmptyString
    observed_at: datetime
    evidence: tuple[NonEmptyString, ...]
    content_digest: ContentDigest

    @field_validator("observed_at", mode="before")
    @classmethod
    def _validate_observed_at(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            _format_datetime(value)
            return value.astimezone(UTC)
        if not isinstance(value, str):
            raise ValueError("provenance.observed_at must be an ISO-8601 string")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("provenance.observed_at must be valid ISO-8601") from exc
        _format_datetime(parsed)
        return parsed.astimezone(UTC)


class MemoryRecord(_StrictModel):
    """Versioned memory envelope stored as one native LangGraph value."""

    kind: NonEmptyString
    content: JsonObject
    provenance: Provenance
    schema_version: Literal[1] = SCHEMA_VERSION

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: object) -> JsonObject:
        return _normalize_object(value, name="content")

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> int:
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema version: {value}")
        return value

    @model_validator(mode="after")
    def _validate_digest(self) -> MemoryRecord:
        self._require_matching_digest()
        return self

    def _require_matching_digest(self) -> None:
        if self.provenance.content_digest != content_digest(self.content):
            raise ValueError("provenance content digest does not match content")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        content: Mapping[str, object],
        source: str,
        observed_at: datetime | None = None,
        evidence: tuple[str, ...] = (),
    ) -> MemoryRecord:
        """Create a validated record and compute its provenance digest."""
        normalized = _normalize_object(dict(content), name="content")
        return cls(
            kind=kind,
            content=normalized,
            provenance=Provenance(
                source=source,
                observed_at=observed_at or datetime.now(UTC),
                evidence=evidence,
                content_digest=content_digest(normalized),
            ),
        )

    def to_dict(self) -> JsonObject:
        """Return a revalidated JSON-native envelope."""
        validated = type(self).model_validate(self.model_dump(mode="python"))
        return cast(JsonObject, validated.model_dump(mode="json"))

    @classmethod
    def from_dict(cls, value: object) -> MemoryRecord:
        """Parse and validate a JSON-native envelope."""
        return cls.model_validate(value)


class MemoryEntry(_StrictModel):
    """A stored memory plus native key, namespace, and timestamps."""

    namespace: Namespace
    key: str
    record: MemoryRecord
    created_at: datetime
    updated_at: datetime
    score: float | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def _require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory entry timestamps must be timezone-aware")
        return value.astimezone(UTC)


class HealthReport(_StrictModel):
    """Small typed snapshot of the live SQLite store."""

    status: Literal["ok"]
    schema_version: int
    journal_mode: str
    synchronous: int
    busy_timeout_ms: int
    foreign_keys: bool
    records: int
