"""
MCP server implementation for the EMVR system.

This module implements the main FastMCP server for the EMVR system.
"""

import asyncio
import logging
import os
import signal
import sys

from fastmcp.server import MCPServer
from langchain_community.chat_models import ChatOpenAI

from emvr.agents.orchestration import get_orchestrator, initialize_orchestration
from emvr.config import get_settings
from emvr.core.db_connections import close_connections, initialize_connections
from emvr.ingestion.pipeline import ingestion_pipeline
from emvr.mcp_server.endpoints import register_endpoints, register_resources
from emvr.mcp_server.endpoints.agent_endpoints import (
    register_agent_endpoints,
    register_agent_resources,
)
from emvr.memory.memory_manager import memory_manager
from emvr.retrieval.pipeline import retrieval_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MemoryMCPServer:
    """
    MCP server for the EMVR system.

    Provides memory operations, search, ingestion capabilities, and agent orchestration.
    """

    def __init__(self) -> None:
        """Initialize the MCP server."""
        self._settings = get_settings()
        self._mcp_server = MCPServer(
            name="memory",
            description="EMVR Memory MCP Server",
            version="0.1.0",
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the server and all dependencies."""
        if self._initialized:
            return

        try:
            logger.info("Initializing MCP server")

            # Initialize database connections
            initialize_connections()

            # Initialize memory manager
            await memory_manager.initialize()

            # Initialize ingestion pipeline
            await ingestion_pipeline.initialize()

            # Initialize retrieval pipeline
            await retrieval_pipeline.initialize()

            # Initialize language model for agents
            llm = ChatOpenAI(
                temperature=0.0,
                model=self._settings.openai_model,
            )

            # Initialize agent orchestration
            await initialize_orchestration(llm=llm)

            # Register endpoints and resources
            await register_endpoints(self._mcp_server)
            await register_resources(self._mcp_server)

            # Register agent endpoints and resources
            await register_agent_endpoints(self._mcp_server)
            await register_agent_resources(self._mcp_server)

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            self._initialized = True
            logger.info("MCP server initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize MCP server: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def handle_exit_signal(sig, frame) -> None:
            logger.info(f"Received signal {sig}, shutting down...")
            self.cleanup()
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)

    async def run_stdio(self) -> None:
        """Run the server in stdio mode."""
        try:
            await self.initialize()
            logger.info("Starting MCP server in stdio mode")
            await self._mcp_server.start_stdio()
        except Exception as e:
            logger.exception(f"Error running MCP server in stdio mode: {e}")
            self.cleanup()
            raise

    async def run_http(self, host: str | None = None, port: int | None = None) -> None:
        """
        Run the server in HTTP mode.

        Args:
        ----
            host: Host to bind to (defaults to settings)
            port: Port to bind to (defaults to settings)

        """
        try:
            await self.initialize()

            host = host or self._settings.mcp_host
            port = port or self._settings.mcp_port

            logger.info(f"Starting MCP server on {host}:{port}")
            await self._mcp_server.start_http(host=host, port=port)
        except Exception as e:
            logger.exception(f"Error running MCP server in HTTP mode: {e}")
            self.cleanup()
            raise

    def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        logger.info("Cleaning up resources...")

        # Shutdown agent orchestration if initialized
        orchestrator = get_orchestrator()
        if orchestrator:
            asyncio.run(orchestrator.shutdown())

        # Close memory manager connections
        memory_manager.close()

        # Close database connections
        close_connections()

        logger.info("Cleanup complete")


# Main entry point
async def main() -> None:
    """Main entry point for the MCP server."""
    server = MemoryMCPServer()

    # Determine server mode based on environment variable
    server_mode = os.environ.get("MCP_SERVER_MODE", "stdio")

    if server_mode.lower() == "http":
        await server.run_http()
    else:
        await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
