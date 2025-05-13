"""
File upload component for EMVR UI.

This module provides functionality for file uploads and document ingestion.
"""

import logging
import os
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.types import FileDict

from emvr.ingestion.pipeline import ingestion_pipeline

# Configure logging
logger = logging.getLogger(__name__)


# ----- Supported File Types -----

SUPPORTED_TEXT_TYPES: set[str] = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "application/json",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
}

SUPPORTED_IMAGE_TYPES: set[str] = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

SUPPORTED_FILE_EXTENSIONS: set[str] = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
}


# ----- File Upload Functions -----


async def process_file_upload(file: FileDict) -> tuple[bool, str, dict[str, Any]]:
    """
    Process an uploaded file.

    Args:
    ----
        file: The uploaded file

    Returns:
    -------
        Tuple of (success, message, result)

    """
    try:
        # Check file type
        file_path = file["path"]
        file_name = file["name"]
        file_type = file.get("type", "")
        file_size = os.path.getsize(file_path)
        file_ext = Path(file_name).suffix.lower()

        # Check if file is supported
        if (
            file_type
            and file_type not in SUPPORTED_TEXT_TYPES
            and file_type not in SUPPORTED_IMAGE_TYPES
        ):
            if file_ext not in SUPPORTED_FILE_EXTENSIONS:
                return (
                    False,
                    f"Unsupported file type: {file_type or file_ext}",
                    {"status": "error", "error": "Unsupported file type"},
                )

        # Check file size (limit to 20MB)
        if file_size > 20 * 1024 * 1024:
            return (
                False,
                f"File too large: {file_size / (1024 * 1024):.2f} MB (max 20MB)",
                {"status": "error", "error": "File too large"},
            )

        # Get pipeline from session or initialize
        pipeline = cl.user_session.get("ingestion_pipeline")
        if pipeline is None:
            pipeline = ingestion_pipeline
            await pipeline.initialize()
            cl.user_session.set("ingestion_pipeline", pipeline)

        # Process the file with metadata
        result = await pipeline.ingest_file(
            file_path=file_path,
            metadata={
                "source": file_name,
                "file_type": file_type,
                "file_size": file_size,
            },
        )

        if result.get("status", "") == "success" or "id" in result:
            success_message = (
                f"Successfully processed file: {file_name} "
                f"({result.get('chunks', 0)} chunks created)"
            )
            return (True, success_message, result)
        error_message = (
            f"Failed to process file: {result.get('error', 'Unknown error')}"
        )
        return (False, error_message, result)

    except Exception as e:
        logger.exception(f"Error processing file: {e}")
        return (
            False,
            f"Error processing file: {e!s}",
            {"status": "error", "error": str(e)},
        )


# ----- UI Components -----


async def show_file_upload_ui() -> None:
    """Show the file upload UI."""
    # Create file upload elements
    elements = [
        cl.File(
            name="upload",
            accept=list(SUPPORTED_FILE_EXTENSIONS),
            max_files=5,
            max_size_mb=20,
        ),
        cl.Text(
            name="description",
            label="Description (Optional)",
            placeholder="Add a description for the uploaded files...",
        ),
    ]

    instructions = (
        "Upload files to add to the knowledge base. "
        "Supported formats: PDF, Word, Excel, PowerPoint, text, markdown, CSV, "
        "JSON, HTML, and images."
    )

    await cl.Message(
        content=instructions,
        elements=elements,
        author="EMVR",
    ).send()


async def show_url_ingestion_ui() -> None:
    """Show the URL ingestion UI."""
    # Create URL input elements
    elements = [
        cl.Text(
            name="url",
            label="URL to ingest",
            placeholder="https://example.com/article",
        ),
        cl.Toggle(
            name="recursive",
            label="Recursively process links on page",
            initial=False,
        ),
    ]

    instructions = (
        "Enter a URL to add to the knowledge base. "
        "The system will download and process the content."
    )

    await cl.Message(
        content=instructions,
        elements=elements,
        author="EMVR",
    ).send()
