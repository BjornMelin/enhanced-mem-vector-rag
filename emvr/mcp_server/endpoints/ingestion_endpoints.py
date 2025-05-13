"""Ingestion endpoints for the MCP server."""

import logging
import os
from typing import Any

from fastapi import HTTPException
from fastmcp import MCPServer, ToolConfig

from emvr.ingestion.base import Document

# Configure logging
logger = logging.getLogger(__name__)

# Initialize ingestion pipeline (will be replaced on registration)
ingestion_pipeline = None


async def register_ingestion_endpoints(mcp_server: MCPServer) -> None:
    """
    Register ingestion endpoints with the MCP server.

    Args:
    ----
        mcp_server: MCP server instance

    """
    global ingestion_pipeline

    try:
        # Initialize ingestion pipeline if not already initialized
        if ingestion_pipeline is None:
            from emvr.ingestion.pipeline import IngestionPipeline
            from emvr.memory.memory_manager import MemoryManager

            # Get memory manager from MCP server
            memory_manager = mcp_server.state.get("memory_manager")
            if memory_manager is None:
                # Create new memory manager if not in server state
                memory_manager = MemoryManager()
                mcp_server.state["memory_manager"] = memory_manager

            # Create ingestion pipeline
            ingestion_pipeline = IngestionPipeline(
                vector_store=memory_manager.vector_store,
            )
            mcp_server.state["ingestion_pipeline"] = ingestion_pipeline

        # Register ingestion endpoints
        ingestion_tools = [
            ToolConfig(
                name="ingest__text",
                function=ingest_text,
                description="Ingest text content into the memory system",
            ),
            ToolConfig(
                name="ingest__url",
                function=ingest_url,
                description="Ingest content from a URL into the memory system",
            ),
            ToolConfig(
                name="ingest__file",
                function=ingest_file,
                description="Ingest a file into the memory system",
            ),
            ToolConfig(
                name="ingest__delete",
                function=delete_document,
                description="Delete a document from the memory system",
            ),
        ]

        # Register all ingestion tools
        for tool in ingestion_tools:
            mcp_server.register_tool(tool)

        logger.info("Ingestion endpoints registered successfully")

    except Exception as e:
        logger.exception(f"Failed to register ingestion endpoints: {e!s}")
        raise


# Define MCP tool functions


async def ingest_text(
    content: str,
    metadata: dict[str, Any] | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    """
    Ingest text content into the memory system.

    Args:
    ----
        content: Text content to ingest
        metadata: Optional metadata for the content
        document_id: Optional document ID

    Returns:
    -------
        Dictionary with ingestion result

    """
    try:
        # Create document
        document = Document(
            id=document_id,
            content=content,
            metadata=metadata or {},
        )

        # Ingest document
        results = await ingestion_pipeline.ingest([document])

        return {
            "success": all(result.success for result in results),
            "results": [result.dict() for result in results],
        }
    except Exception as e:
        logger.exception(f"Error ingesting text: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


async def ingest_url(
    url: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ingest content from a URL into the memory system.

    Args:
    ----
        url: URL to ingest
        metadata: Optional metadata for the content

    Returns:
    -------
        Dictionary with ingestion result

    """
    try:
        # For now, we'll stub this function
        # In a real implementation, this would:
        # 1. Fetch content from the URL
        # 2. Process and chunk the content
        # 3. Call the ingestion pipeline

        logger.warning("URL ingestion not fully implemented yet")

        # Return a placeholder result
        return {
            "success": False,
            "error": "URL ingestion not fully implemented yet",
            "url": url,
        }
    except Exception as e:
        logger.exception(f"Error ingesting URL: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


async def ingest_file(
    file_path: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ingest a file into the memory system.

    Args:
    ----
        file_path: Path to the file
        metadata: Optional metadata for the file

    Returns:
    -------
        Dictionary with ingestion result

    """
    try:
        # For now, we'll stub this function
        # In a real implementation, this would:
        # 1. Read the file
        # 2. Process and chunk the content
        # 3. Call the ingestion pipeline

        logger.warning("File ingestion not fully implemented yet")

        # Verify file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Return a placeholder result
        return {
            "success": False,
            "error": "File ingestion not fully implemented yet",
            "file_path": file_path,
        }
    except Exception as e:
        logger.exception(f"Error ingesting file: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_document(
    document_ids: list[str],
) -> dict[str, Any]:
    """
    Delete documents from the memory system.

    Args:
    ----
        document_ids: List of document IDs to delete

    Returns:
    -------
        Dictionary with deletion result

    """
    try:
        # Delete documents
        results = await ingestion_pipeline.delete(document_ids)

        return {
            "success": all(result.success for result in results),
            "results": [result.dict() for result in results],
        }
    except Exception as e:
        logger.exception(f"Error deleting documents: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))
