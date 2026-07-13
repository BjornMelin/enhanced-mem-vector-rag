# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.x | Yes |
| Unreleased prototype commits | No |

## Reporting

Do not report vulnerabilities in public issues, discussions, or pull requests.

GitHub private vulnerability reporting isn't enabled yet. [Issue #12](https://github.com/BjornMelin/enhanced-mem-vector-rag/issues/12) is a sanitized tracking issue for enabling it; don't add vulnerability details there. Until the repository's **Security** tab offers **Report a vulnerability**, don't submit exploit details, credentials, memory database contents, or personal data through GitHub.

## Boundary

The supported Model Context Protocol (MCP) server uses local standard input and output only. The project doesn't claim a remote authentication or multi-tenant isolation boundary. Database files inherit the permissions and backup policy of their host account. Choose an appropriate path and protect exported JavaScript Object Notation Lines (JSON Lines).

The core package requires no credentials. Semantic indexing can receive an embedding function programmatically, but Enhanced Memory doesn't configure or transmit data to an embedding provider by default.
