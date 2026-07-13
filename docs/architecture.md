# Architecture

## Why native LangGraph SQLite

Enhanced Memory is a thin domain layer over LangGraph's native `langgraph-checkpoint-sqlite` store.

The decision framework weights solution leverage at 35%, application value at 30%, maintenance and cognitive load at 25%, and adaptability at 10%.

| Option | Leverage | Value | Maintenance | Adaptability | Weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compose native LangGraph SQLite storage | 9.9 | 9.7 | 9.8 | 9.6 | **9.8** |
| Fork LangGraph's implementation | 7.0 | 9.3 | 6.3 | 9.0 | 7.7 |
| Build a custom SQLite `BaseStore` | 6.0 | 9.2 | 6.4 | 8.8 | 7.3 |
| Retain Qdrant and Neo4j | 4.0 | 6.0 | 2.5 | 5.0 | 4.3 |

The native composition wins because it already owns namespaces, filters, batch operations, time to live (TTL), deletion, migrations, synchronous and asynchronous application programming interfaces (APIs), and sqlite-vec integration. Reimplementing those behaviors would create a second contract with no product advantage.

## Component ownership

```text
FastMCP standard-input/output tools
        |
MemoryService: provenance, strict expiry, purge, export
        |
LangGraph SqliteStore / BaseStore
        |
SQLite write-ahead logging and optional sqlite-vec index
```

Enhanced Memory owns only:

- The versioned memory envelope
- Source and evidence provenance with content digests
- Fail-closed expiry before reads
- Deterministic JavaScript Object Notation Lines (JSON Lines) export
- The local Model Context Protocol (MCP) tool surface

LangGraph owns storage schema, migrations, namespace and filter contracts, batch behavior, TTL representation, and vector indexing. FastMCP owns MCP schemas, transport, and protocol compatibility. SQLite owns durability and concurrency.

Enhanced Memory pins `langgraph-checkpoint-sqlite` 3.1.0 because its guarded adapter corrects query behavior at private extension points. The adapter owns these corrections:

- Literal tuple-segment namespaces with one-segment listing wildcards
- Precise expiry and bounded TTL validation across Python and protocol calls
- Type-exact filter comparisons and stable creation timestamps on upsert
- Cosine aggregation before semantic pagination
- Concurrent migration retries
- UTC timestamps
- Lifecycle coordination for synchronous and asynchronous native calls

Upgrade pull requests must rerun the full native sync and async contract suite before changing that pin.

## Managed memory envelope

Every managed value is a JavaScript Object Notation (JSON) object with:

- `schema_version`: currently `1`
- `kind`: a non-empty caller-defined string
- `content`: the caller's JSON object, with string keys, finite numbers, and integers from `-2^63` through `2^64 - 1`
- `provenance`: source, observation timestamp, optional evidence, and SHA-256 digest of canonical JSON content

LangGraph callers can use the exposed native store directly. Managed records are the supported contract for the MCP and high-level Python APIs.

## Excluded v1 capabilities

The v1 contract excludes Qdrant, Neo4j, Supabase, Chainlit, crawlers, document ingestion, large language model providers, graph extraction, agent supervisors, remote Hypertext Transfer Protocol, authentication, Kubernetes, monitoring stacks, and compatibility adapters for the unreleased prototype. Those concerns already have canonical owners elsewhere in the portfolio.

## Predecessor acceptance evidence

The preserved Qdrant, Neo4j, and Crawl4AI predecessor supplied acceptance invariants, not implementation code: stable identity, JSON round trips, temporal metadata, source and evidence provenance, concurrent operations, and malformed MCP input. Cypher, graph traversal, crawler behavior, provider calls, and caching tiers aren't part of this successor.
