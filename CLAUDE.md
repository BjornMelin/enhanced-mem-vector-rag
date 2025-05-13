> **Inheritance**  
> This repo automatically loads the global Simplicity Charter (`~/.config/claude/CLAUDE.md`).  
> **Read / update these two docs before writing code:**  
> • `docs/implementation/todo_list.md` – open tasks & design notes  
> • `docs/status/implementation_status.md` – current build status & blockers

## Project Snapshot

Enhanced Mem‑Vector RAG builds a persistent memory fabric for personal AI agents by combining:

| Layer                 | Tech                            | Purpose                                |
| --------------------- | ------------------------------- | -------------------------------------- |
| **Vector memory**     | Qdrant + Mem0                   | Hybrid & filtered semantic search      |
| **Graph memory**      | Neo4j + Graphiti                | Temporal / procedural knowledge graphs |
| **RAG orchestration** | LlamaIndex                      | Multi‑store retrieval & indexing       |
| **Agent workflow**    | LangGraph (Supervisor → Worker) | Multi‑agent coordination               |
| **Agent logic/tools** | LangChain                       | Planning & tool routing                |
| **Custom API**        | FastMCP 2.0                     | Low‑latency MCP endpoints              |
| **Relational store**  | Supabase (PostgreSQL + RBAC)    | Structured metadata & auth             |
| **Packaging / lint**  | uv + ruff                       | Fast installs; one‑pass fix + format   |

## Tool‑Integration Order

1. `memory`   (custom FastMCP server) – central hub
2. `context7` – fetch latest library docs before coding
3. `sequential‑thinking` – plan complex logic first
4. **Search tools:**
   - `exa`    – fast web search API for coding & research
   - `linkup` – deep multi‑page crawl / extraction
5. `git` / `github` – VCS actions
6. `playwright` – only when ingestion needs browser automation

_(Removed `tavily`; replaced with `exa` and kept `linkup` to match TripSage tool palette.)_

## Memory Protocol

_Session start_ ➜ `memory.read_graph` ➜ `memory.search_nodes("Implementation Progress")`.  
Log each decision / implementation with `memory.add_observations`, `create_entities`, and `create_relations`.  
If context resets, repeat the session‑start steps immediately.

## Git Workflow

Branches: `main` (protected) • `dev` (protected) • `feat/*` • `fix/*`  
Commits: Conventional format – e.g. `feat(memory): implement entity observation history`.  
Pull requests target **main** and must pass CI.

## Agent Pattern Guide

| Pattern          | When to use                          |
| ---------------- | ------------------------------------ |
| Single LLM Call  | Deterministic, one‑shot tasks        |
| Workflow         | Predictable multi‑step jobs          |
| Autonomous Agent | Exploratory, dynamic planning (last) |

**Design mantra:** start simple; add complexity only when needed (KISS / YAGNI).

## Security Reminder

Store real secrets in `.env`; commit only `.env.example` placeholders.  
Supabase RBAC handles API and database access.

_End of project instructions._
