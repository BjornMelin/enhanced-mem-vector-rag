# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.x | Yes |
| Unreleased prototype commits | No |

## Reporting

GitHub private vulnerability reporting isn't enabled yet. [Issue #12](https://github.com/BjornMelin/enhanced-mem-vector-rag/issues/12) tracks that repository setting.

Until the private flow is live, open a public issue that asks the maintainer to arrange a private channel. Don't include exploit details, credentials, memory database contents, or personal data.

## Boundary

The supported Model Context Protocol (MCP) server uses local standard input and output only. The project doesn't claim a remote authentication or multi-tenant isolation boundary. Database files inherit the permissions and backup policy of their host account. Choose an appropriate path and protect exported JavaScript Object Notation Lines (JSON Lines).

The core package requires no credentials. Semantic indexing can receive an embedding function programmatically, but Enhanced Memory doesn't configure or transmit data to an embedding provider by default.
