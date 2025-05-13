"""
File-based document loaders for the EMVR system.

This module provides loaders for various file types using LlamaIndex.
"""

import logging
import os
from typing import Any

# Will use LlamaIndex loaders when integrated
# from llama_index.core.readers import SimpleDirectoryReader
# from llama_index.core.schema import Document as LlamaDocument
from emvr.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class FileLoader:
    """
    Base loader for file-based documents using LlamaIndex.

    Provides methods for loading documents from files and directories.
    """

    def __init__(self) -> None:
        """Initialize the file loader."""
        self._settings = get_settings()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the loader."""
        if self._initialized:
            return

        try:
            logger.info("Initializing file loader")

            # Any initialization logic would go here
            # For example, checking that required directories exist

            self._initialized = True
            logger.info("File loader initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize file loader: {e}")
            raise

    def ensure_initialized(self) -> None:
        """Ensure the loader is initialized."""
        if not self._initialized:
            self.initialize()

    def load_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load a single file.

        Args:
        ----
            file_path: Path to the file
            metadata: Optional metadata for the document

        Returns:
        -------
            List[Dict]: List of document dictionaries with "text" and "metadata"

        """
        self.ensure_initialized()

        try:
            logger.info(f"Loading file: {file_path}")

            # Placeholder implementation
            # This will be replaced with actual LlamaIndex usage
            file_path = os.path.abspath(file_path)

            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []

            # Basic file read as a placeholder
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Create a document dict
            doc_metadata = metadata or {}
            doc_metadata.update(
                {
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": os.path.splitext(file_path)[1][1:],
                    "file_size": os.path.getsize(file_path),
                },
            )

            # Return as a list of documents (single document in this case)
            return [
                {
                    "text": content,
                    "metadata": doc_metadata,
                },
            ]

            # TODO: Implement with LlamaIndex SimpleDirectoryReader
            # documents = SimpleDirectoryReader(
            #     input_files=[file_path],
            #     recursive=False,
            #     filename_as_id=True
            # ).load_data()
            #
            # # Convert LlamaIndex documents to our format
            # return [
            #     {
            #         "text": doc.text,
            #         "metadata": {**doc.metadata, **(metadata or {})}
            #     }
            #     for doc in documents
            # ]

        except Exception as e:
            logger.exception(f"Failed to load file {file_path}: {e}")
            return []

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
        exclude_hidden: bool = True,
        file_extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load all files from a directory.

        Args:
        ----
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            metadata: Optional metadata for all documents
            exclude_hidden: Whether to exclude hidden files/dirs
            file_extensions: List of file extensions to include

        Returns:
        -------
            List[Dict]: List of document dictionaries with "text" and "metadata"

        """
        self.ensure_initialized()

        try:
            logger.info(f"Loading directory: {directory_path} (recursive={recursive})")

            # Placeholder implementation
            # This will be replaced with actual LlamaIndex usage
            directory_path = os.path.abspath(directory_path)

            if not os.path.exists(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return []

            documents = []

            # Define a helper function to get files
            def get_files(path):
                files = []
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)

                    # Skip hidden files/dirs if requested
                    if exclude_hidden and item.startswith("."):
                        continue

                    if os.path.isfile(item_path):
                        # Check file extension if specified
                        if file_extensions:
                            ext = os.path.splitext(item)[1][1:].lower()
                            if ext not in [
                                e.lower().lstrip(".") for e in file_extensions
                            ]:
                                continue
                        files.append(item_path)
                    elif os.path.isdir(item_path) and recursive:
                        files.extend(get_files(item_path))
                return files

            # Get all matching files
            file_paths = get_files(directory_path)

            # Load each file
            for file_path in file_paths:
                try:
                    file_docs = self.load_file(file_path, metadata)
                    documents.extend(file_docs)
                except Exception as e:
                    logger.exception(f"Error loading file {file_path}: {e}")

            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents

            # TODO: Implement with LlamaIndex SimpleDirectoryReader
            # documents = SimpleDirectoryReader(
            #     input_dir=directory_path,
            #     recursive=recursive,
            #     filename_as_id=True,
            #     exclude_hidden=exclude_hidden,
            #     required_exts=file_extensions
            # ).load_data()
            #
            # # Convert LlamaIndex documents to our format
            # return [
            #     {
            #         "text": doc.text,
            #         "metadata": {**doc.metadata, **(metadata or {})}
            #     }
            #     for doc in documents
            # ]

        except Exception as e:
            logger.exception(f"Failed to load directory {directory_path}: {e}")
            return []


# Create a singleton instance for import
file_loader = FileLoader()
