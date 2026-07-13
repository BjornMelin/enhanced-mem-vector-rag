from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import cast

import pytest

from emvr import MemoryEntry, MemoryRecord, Provenance, content_digest

OBSERVED_AT = datetime(2026, 7, 12, 18, 30, tzinfo=UTC)


def test_record_round_trip_and_digest_are_canonical() -> None:
    left = {"nested": {"b": 2, "a": 1}, "tags": ["one", "two"]}
    right = {"tags": ["one", "two"], "nested": {"a": 1, "b": 2}}

    record = MemoryRecord.create(
        kind="fact",
        content=left,
        source="unit-test",
        observed_at=OBSERVED_AT,
        evidence=("event:42",),
    )

    assert record.provenance.content_digest == content_digest(right)
    assert MemoryRecord.from_dict(record.to_dict()) == record
    assert record.to_dict()["schema_version"] == 1


def test_record_copies_content_and_rejects_non_json_values() -> None:
    content = {"answer": 42}
    record = MemoryRecord.create(
        kind="fact", content=content, source="unit-test", observed_at=OBSERVED_AT
    )
    content["answer"] = 7

    assert record.content == {"answer": 42}
    with pytest.raises(ValueError, match="JSON object"):
        MemoryRecord.create(
            kind="fact",
            content={"bad": object()},  # type: ignore[dict-item]
            source="unit-test",
            observed_at=OBSERVED_AT,
        )


@pytest.mark.parametrize("value", [1 << 64, -(1 << 63) - 1])
def test_record_rejects_integers_outside_native_storage_range(value: int) -> None:
    with pytest.raises(ValueError, match="64-bit storage range"):
        MemoryRecord.create(
            kind="fact",
            content={"nested": [value]},
            source="unit-test",
            observed_at=OBSERVED_AT,
        )


@pytest.mark.parametrize("key", [1, None])
def test_record_rejects_non_string_keys_at_every_depth(key: object) -> None:
    with pytest.raises(ValueError, match="string keys at every depth"):
        MemoryRecord.create(
            kind="fact",
            content={"nested": {key: "value"}},  # type: ignore[dict-item]
            source="unit-test",
            observed_at=OBSERVED_AT,
        )


def test_record_rejects_tampering_and_unknown_schema() -> None:
    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )
    tampered = record.to_dict()
    tampered["content"] = {"answer": 7}

    with pytest.raises(ValueError, match="digest does not match"):
        MemoryRecord.from_dict(tampered)

    unsupported = record.to_dict()
    unsupported["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported memory schema"):
        MemoryRecord.from_dict(unsupported)


def test_provenance_requires_aware_time_and_exact_shape() -> None:
    digest = content_digest({"answer": 42})
    with pytest.raises(ValueError, match="timezone-aware"):
        Provenance(
            source="unit-test",
            observed_at=datetime(2026, 7, 12),
            evidence=(),
            content_digest=digest,
        )

    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )
    malformed = record.to_dict()
    provenance = malformed["provenance"]
    assert isinstance(provenance, dict)
    provenance["extra"] = True
    with pytest.raises(ValueError, match="Extra inputs"):
        MemoryRecord.from_dict(malformed)


def test_json_object_and_timestamp_boundaries_are_strict() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        content_digest([])
    with pytest.raises(ValueError, match="string keys"):
        content_digest({1: "value"})

    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )
    for observed_at, message in ((42, "ISO-8601 string"), ("not-a-time", "valid")):
        malformed = record.to_dict()
        provenance = cast(dict[str, object], malformed["provenance"])
        provenance["observed_at"] = observed_at
        with pytest.raises(ValueError, match=message):
            MemoryRecord.from_dict(malformed)


@pytest.mark.parametrize(
    "digest",
    ["bad", "sha256:" + "a" * 63, "sha256:" + "g" * 64],
)
def test_provenance_rejects_invalid_identity_fields(digest: str) -> None:
    with pytest.raises(ValueError, match="content_digest"):
        Provenance(
            source="unit-test",
            observed_at=OBSERVED_AT,
            evidence=(),
            content_digest=digest,
        )

    valid_digest = content_digest({"answer": 42})
    with pytest.raises(ValueError, match="source"):
        Provenance(
            source="",
            observed_at=OBSERVED_AT,
            evidence=(),
            content_digest=valid_digest,
        )
    with pytest.raises(ValueError, match="evidence"):
        Provenance(
            source="unit-test",
            observed_at=OBSERVED_AT,
            evidence=("",),
            content_digest=valid_digest,
        )


def test_record_parser_rejects_invalid_field_types() -> None:
    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )

    invalid_provenance = record.to_dict()
    provenance = cast(dict[str, object], invalid_provenance["provenance"])
    provenance["source"] = 42
    with pytest.raises(ValueError, match="source"):
        MemoryRecord.from_dict(invalid_provenance)

    invalid_evidence = record.to_dict()
    provenance = cast(dict[str, object], invalid_evidence["provenance"])
    provenance["evidence"] = [42]
    with pytest.raises(ValueError, match="evidence"):
        MemoryRecord.from_dict(invalid_evidence)

    invalid_record = record.to_dict()
    invalid_record["kind"] = 42
    with pytest.raises(ValueError, match="kind"):
        MemoryRecord.from_dict(invalid_record)


def test_record_revalidates_kind_and_mutable_content_before_write() -> None:
    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )
    with pytest.raises(ValueError, match="kind"):
        MemoryRecord(kind="", content=record.content, provenance=record.provenance)

    mutated = deepcopy(record)
    mutated.content["answer"] = 7
    with pytest.raises(ValueError, match="digest does not match"):
        mutated.to_dict()


def test_record_revalidates_unchecked_pydantic_copies() -> None:
    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )
    invalid_records = [
        record.model_copy(update={"kind": ""}),
        record.model_copy(update={"schema_version": 2}),
        record.model_copy(
            update={"provenance": record.provenance.model_copy(update={"source": ""})}
        ),
        MemoryRecord.model_construct(
            kind="",
            content=record.content,
            provenance=record.provenance,
            schema_version=1,
        ),
    ]

    for invalid in invalid_records:
        with pytest.raises(ValueError):
            invalid.to_dict()


def test_memory_entry_requires_aware_timestamps() -> None:
    record = MemoryRecord.create(
        kind="fact",
        content={"answer": 42},
        source="unit-test",
        observed_at=OBSERVED_AT,
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        MemoryEntry(
            namespace=("notes",),
            key="one",
            record=record,
            created_at=datetime(2026, 7, 12),
            updated_at=OBSERVED_AT,
        )
