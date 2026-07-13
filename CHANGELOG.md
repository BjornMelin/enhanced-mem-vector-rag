# Changelog

All notable changes are documented here.

## 1.0.0 - 2026-07-12

### Added

- Native LangGraph SQLite `BaseStore` composition with write-ahead logging configuration.
- Versioned memory records with source/evidence provenance and content digests.
- Literal, case-sensitive namespaced create, read, update, delete, purge, and listing operations with precise sliding expiry.
- Semantic de-duplication before bounded pagination, configured-field indexing, deterministic JavaScript Object Notation Lines export, and health.
- Concurrent first-run migrations, UTC entry timestamps, bounded TTL validation, and segment-aware namespace wildcards.
- Local standard-input/output Model Context Protocol tools with in-process protocol tests.
- Strict typing, multi-version continuous integration, wheel verification, and release automation.

### Removed

- The unreleased Qdrant, Neo4j, Supabase, Chainlit, crawler, retrieval-augmented generation, agent, user-interface, deployment, and monitoring prototypes.
- Legacy dependency mirrors and compatibility paths.

This is a breaking, forward-only rewrite. The earlier prototype didn't publish a release or stable persisted contract, so this release provides no compatibility adapter.
