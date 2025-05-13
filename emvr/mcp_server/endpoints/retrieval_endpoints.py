"""Retrieval endpoints for the MCP server."""

import logging
from typing import Any

from fastapi import HTTPException
from fastmcp import MCPServer, ToolConfig

from emvr.retrieval.pipeline import RetrievalPipeline

# Configure logging
logger = logging.getLogger(__name__)

# Initialize retrieval pipeline (will be replaced on registration)
retrieval_pipeline = None


async def register_retrieval_endpoints(mcp_server: MCPServer) -> None:
    """
    Register retrieval endpoints with the MCP server.

    Args:
    ----
        mcp_server: MCP server instance

    """
    global retrieval_pipeline

    try:
        # Initialize retrieval pipeline if not already initialized
        if retrieval_pipeline is None:
            from emvr.memory.memory_manager import MemoryManager

            # Get memory manager from MCP server
            memory_manager = mcp_server.state.get("memory_manager")
            if memory_manager is None:
                # Create new memory manager if not in server state
                memory_manager = MemoryManager()
                mcp_server.state["memory_manager"] = memory_manager

            # Create retrieval pipeline
            retrieval_pipeline = RetrievalPipeline(
                memory_manager=memory_manager,
                retrieval_mode="fusion",
            )
            mcp_server.state["retrieval_pipeline"] = retrieval_pipeline

        # Register search endpoints
        search_tools = [
            ToolConfig(
                name="search__hybrid",
                function=hybrid_search,
                description="Perform hybrid search across vector and graph stores",
            ),
            ToolConfig(
                name="search__vector",
                function=vector_search,
                description="Perform vector search against the vector store",
            ),
            ToolConfig(
                name="search__graph",
                function=graph_search,
                description="Perform graph search against the knowledge graph",
            ),
            ToolConfig(
                name="search__enrich_context",
                function=enrich_context,
                description="Enrich context with retrieved information",
            ),
        ]

        # Register all search tools
        for tool in search_tools:
            mcp_server.register_tool(tool)

        logger.info("Retrieval endpoints registered successfully")

    except Exception as e:
        logger.exception(f"Failed to register retrieval endpoints: {e!s}")
        raise


# Define MCP tool functions


async def hybrid_search(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform hybrid search across vector and graph stores.

    Args:
    ----
        query: Search query string
        top_k: Number of results to return
        filters: Optional filters to apply to the search

    Returns:
    -------
        Dictionary with search results

    """
    try:
        return await retrieval_pipeline.search_hybrid(
            query=query,
            top_k=top_k,
            filters=filters,
        )
    except Exception as e:
        logger.exception(f"Error performing hybrid search: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


async def vector_search(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform vector search against the vector store.

    Args:
    ----
        query: Search query string
        top_k: Number of results to return
        filters: Optional filters to apply to the search

    Returns:
    -------
        Dictionary with search results

    """
    try:
        return await retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            mode="vector",
        )
    except Exception as e:
        logger.exception(f"Error performing vector search: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


async def graph_search(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Perform graph search against the knowledge graph.

    Args:
    ----
        query: Search query string
        top_k: Number of results to return
        filters: Optional filters to apply to the search

    Returns:
    -------
        Dictionary with search results

    """
    try:
        return await retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            mode="graph",
        )
    except Exception as e:
        logger.exception(f"Error performing graph search: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


async def enrich_context(
    query: str,
    context: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Enrich context with retrieved information.

    Args:
    ----
        query: Query string
        context: Optional existing context to enrich
        top_k: Number of results to include

    Returns:
    -------
        Dictionary with enriched context

    """
    try:
        return await retrieval_pipeline.enrich_context(
            query=query,
            context=context,
            top_k=top_k,
        )
    except Exception as e:
        logger.exception(f"Error enriching context: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))
