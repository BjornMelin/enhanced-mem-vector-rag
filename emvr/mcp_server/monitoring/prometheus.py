import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.openmetrics.exposition import generate_latest

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# Metrics
REQUESTS = Counter(
    "emvr_mcp_server_requests_total",
    "Total number of requests received",
    ["method", "endpoint", "status"],
)

REQUESTS_IN_PROGRESS = Gauge(
    "emvr_mcp_server_requests_in_progress",
    "Number of requests currently being processed",
    ["method", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "emvr_mcp_server_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=(
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        25.0,
        50.0,
        75.0,
        100.0,
        float("inf"),
    ),
)

VECTOR_COUNT = Gauge(
    "emvr_qdrant_vector_count",
    "Number of vectors stored in Qdrant",
    ["collection"],
)

GRAPH_NODE_COUNT = Gauge(
    "emvr_neo4j_node_count",
    "Number of nodes in Neo4j graph",
    ["label"],
)

GRAPH_RELATION_COUNT = Gauge(
    "emvr_neo4j_relation_count",
    "Number of relations in Neo4j graph",
    ["type"],
)

AGENT_OPERATIONS = Counter(
    "emvr_agent_operations_total",
    "Total number of agent operations",
    ["agent_type", "operation", "status"],
)

MEM_OPERATIONS = Counter(
    "emvr_memory_operations_total",
    "Total number of memory operations",
    ["operation", "status"],
)

ACTIVE_SESSIONS = Gauge(
    "emvr_active_sessions",
    "Number of active sessions",
    ["session_type"],
)


def setup_metrics(app: FastAPI) -> None:
    """Setup metrics endpoint and middleware for the FastAPI app."""

    @app.get("/metrics")
    async def metrics():
        return generate_latest()

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        method = request.method
        path = request.url.path

        # Skip metrics endpoint itself
        if path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).inc()

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()
            REQUESTS.labels(method=method, endpoint=path, status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(
                time.time() - start_time,
            )


def track_agent_operation(agent_type: str, operation: str) -> Callable[[F], F]:
    """Decorator to track agent operations."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = await func(*args, **kwargs)
                AGENT_OPERATIONS.labels(
                    agent_type=agent_type,
                    operation=operation,
                    status="success",
                ).inc()
                return result
            except Exception:
                AGENT_OPERATIONS.labels(
                    agent_type=agent_type,
                    operation=operation,
                    status="failure",
                ).inc()
                raise

        return wrapper

    return decorator


def track_memory_operation(operation: str) -> Callable[[F], F]:
    """Decorator to track memory operations."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = await func(*args, **kwargs)
                MEM_OPERATIONS.labels(operation=operation, status="success").inc()
                return result
            except Exception:
                MEM_OPERATIONS.labels(operation=operation, status="failure").inc()
                raise

        return wrapper

    return decorator


def update_vector_count(collection: str, count: int) -> None:
    """Update the vector count metric for a collection."""
    VECTOR_COUNT.labels(collection=collection).set(count)


def update_graph_counts(
    node_counts: dict[str, int],
    relation_counts: dict[str, int],
) -> None:
    """Update Neo4j graph count metrics."""
    for label, count in node_counts.items():
        GRAPH_NODE_COUNT.labels(label=label).set(count)

    for rel_type, count in relation_counts.items():
        GRAPH_RELATION_COUNT.labels(type=rel_type).set(count)


def update_active_sessions(session_type: str, count: int) -> None:
    """Update active sessions metric."""
    ACTIVE_SESSIONS.labels(session_type=session_type).set(count)
