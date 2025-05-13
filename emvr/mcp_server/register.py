"""MCP server registration module."""

import logging

from fastmcp import MCPServer

from emvr.mcp_server.endpoints import (
    register_agent_endpoints,
    register_agent_resources,
    register_ingestion_endpoints,
    register_retrieval_endpoints,
)
from emvr.memory.memory_manager import MemoryManager

# Configure logging
logger = logging.getLogger(__name__)


async def register_all_endpoints(mcp_server: MCPServer) -> None:
    """
    Register all endpoints with the MCP server.

    Args:
    ----
        mcp_server: MCP server instance

    """
    try:
        logger.info("Registering all MCP endpoints")

        # Initialize memory manager if not already in server state
        if "memory_manager" not in mcp_server.state:
            memory_manager = MemoryManager()
            mcp_server.state["memory_manager"] = memory_manager

        # Register all endpoints
        await register_memory_endpoints(mcp_server)
        await register_retrieval_endpoints(mcp_server)
        await register_ingestion_endpoints(mcp_server)
        await register_agent_endpoints(mcp_server)
        await register_agent_resources(mcp_server)

        logger.info("All MCP endpoints registered successfully")

    except Exception as e:
        logger.exception(f"Failed to register all MCP endpoints: {e!s}")
        raise


async def register_memory_endpoints(mcp_server: MCPServer) -> None:
    """
    Register memory endpoints with the MCP server.

    Args:
    ----
        mcp_server: MCP server instance

    """
    # Memory endpoints are registered directly in server.py
