"""
Ingestion tools for EMVR agents.

This module implements tools for ingesting data into the system.
"""

import logging
from typing import Any

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from emvr.ingestion.pipeline import ingestion_pipeline

# Configure logging
logger = logging.getLogger(__name__)


# ----- Tool Input/Output Schemas -----


class IngestTextInput(BaseModel):
    """Input schema for ingest_text tool."""

    content: str = Field(..., description="Text content to ingest")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata for the text",
    )
    source_name: str | None = Field(
        None,
        description="Optional source name for the text",
    )


class IngestFileInput(BaseModel):
    """Input schema for ingest_file tool."""

    file_path: str = Field(..., description="Path to the file to ingest")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata for the file",
    )


class IngestUrlInput(BaseModel):
    """Input schema for ingest_url tool."""

    url: str = Field(..., description="URL to ingest")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata for the URL content",
    )


class IngestDirectoryInput(BaseModel):
    """Input schema for ingest_directory tool."""

    directory_path: str = Field(..., description="Path to the directory to ingest")
    recursive: bool = Field(True, description="Whether to search subdirectories")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata for all documents",
    )
    exclude_hidden: bool = Field(
        True,
        description="Whether to exclude hidden files/dirs",
    )
    file_extensions: list[str] | None = Field(
        None,
        description="List of file extensions to include",
    )


# ----- Ingestion Tools -----


@tool
async def ingest_text(
    content: str,
    metadata: dict[str, Any] | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    """
    Ingest raw text into the memory system.

    Args:
    ----
        content: Text content to ingest
        metadata: Optional metadata for the text
        source_name: Optional source name for the text

    Returns:
    -------
        Dict containing the result of the ingestion

    """
    try:
        # Initialize ingestion pipeline if needed
        await ingestion_pipeline.initialize()

        # Process the text
        result = await ingestion_pipeline.ingest_text(
            content=content,
            metadata=metadata,
            source_name=source_name,
        )

        return {
            "id": result.get("id"),
            "chunks": result.get("chunks", 0),
            "status": "success",
            "message": f"Successfully ingested text ({len(content)} chars)",
        }
    except Exception as e:
        logger.exception(f"Text ingestion failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def ingest_file(
    file_path: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ingest a file into the memory system.

    Args:
    ----
        file_path: Path to the file to ingest
        metadata: Optional metadata for the file

    Returns:
    -------
        Dict containing the result of the ingestion

    """
    try:
        # Initialize ingestion pipeline if needed
        await ingestion_pipeline.initialize()

        # Process the file
        result = await ingestion_pipeline.ingest_file(
            file_path=file_path,
            metadata=metadata,
        )

        return {
            "id": result.get("id"),
            "chunks": result.get("chunks", 0),
            "status": "success",
            "message": f"Successfully ingested file: {file_path}",
        }
    except Exception as e:
        logger.exception(f"File ingestion failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def ingest_url(
    url: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ingest content from a URL into the memory system.

    Args:
    ----
        url: URL to ingest
        metadata: Optional metadata for the URL content

    Returns:
    -------
        Dict containing the result of the ingestion

    """
    try:
        # Initialize ingestion pipeline if needed
        await ingestion_pipeline.initialize()

        # Process the URL
        result = await ingestion_pipeline.ingest_url(
            url=url,
            metadata=metadata,
        )

        return {
            "id": result.get("id"),
            "chunks": result.get("chunks", 0),
            "status": "success",
            "message": f"Successfully ingested URL: {url}",
        }
    except Exception as e:
        logger.exception(f"URL ingestion failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def ingest_directory(
    directory_path: str,
    recursive: bool = True,
    metadata: dict[str, Any] | None = None,
    exclude_hidden: bool = True,
    file_extensions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Ingest all files from a directory into the memory system.

    Args:
    ----
        directory_path: Path to the directory to ingest
        recursive: Whether to search subdirectories
        metadata: Optional metadata for all documents
        exclude_hidden: Whether to exclude hidden files/dirs
        file_extensions: List of file extensions to include

    Returns:
    -------
        Dict containing the result of the ingestion

    """
    try:
        # Initialize ingestion pipeline if needed
        await ingestion_pipeline.initialize()

        # Process the directory
        result = await ingestion_pipeline.ingest_directory(
            directory_path=directory_path,
            recursive=recursive,
            metadata=metadata,
            exclude_hidden=exclude_hidden,
            file_extensions=file_extensions,
        )

        return {
            "files_processed": result.get("files_processed", 0),
            "files_failed": result.get("files_failed", 0),
            "total_chunks": result.get("total_chunks", 0),
            "status": "success",
            "message": f"Successfully ingested directory: {directory_path}",
        }
    except Exception as e:
        logger.exception(f"Directory ingestion failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


# ----- Tool Collection -----


def get_ingestion_tools() -> list[BaseTool]:
    """
    Get all ingestion tools.

    Returns
    -------
        List of ingestion tools

    """
    return [
        ingest_text,
        ingest_file,
        ingest_url,
        ingest_directory,
    ]
