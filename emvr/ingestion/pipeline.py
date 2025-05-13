"""
Ingestion pipeline for the EMVR system.

This module provides the main ingestion pipeline that processes documents
and stores them in the memory system.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

# Will use LlamaIndex for text splitting
# from llama_index.core.node_parser import SentenceSplitter
from emvr.config import get_settings
from emvr.core.embedding import embedding_manager
from emvr.ingestion.loaders.file_loaders import file_loader
from emvr.ingestion.loaders.web_loaders import web_loader
from emvr.memory.memory_manager import memory_manager

# Configure logging
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Ingestion pipeline for processing and storing documents.

    Handles:
    - Document loading from various sources
    - Text splitting/chunking
    - Embedding generation
    - Storage in memory system (vector + graph)
    """

    def __init__(self) -> None:
        """Initialize the ingestion pipeline."""
        self._settings = get_settings()
        self._embedding_manager = embedding_manager
        self._memory_manager = memory_manager
        self._file_loader = file_loader
        self._web_loader = web_loader
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the ingestion pipeline."""
        if self._initialized:
            return

        try:
            logger.info("Initializing ingestion pipeline")

            # Initialize components
            self._embedding_manager.initialize()
            await self._memory_manager.initialize()
            self._file_loader.initialize()
            self._web_loader.initialize()

            # Initialize text splitter with default settings
            # self._text_splitter = SentenceSplitter(
            #     chunk_size=self._settings.default_chunk_size,
            #     chunk_overlap=self._settings.default_chunk_overlap
            # )

            self._initialized = True
            logger.info("Ingestion pipeline initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize ingestion pipeline: {e}")
            raise

    async def ensure_initialized(self) -> None:
        """Ensure the pipeline is initialized."""
        if not self._initialized:
            await self.initialize()

    def _split_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Split text into chunks.

        Args:
        ----
            text: Text to split
            metadata: Optional metadata for the chunks

        Returns:
        -------
            List[Dict]: List of chunk dictionaries with "text" and "metadata"

        """
        # Placeholder implementation
        # This will be replaced with actual LlamaIndex usage

        # For now, simply treat the entire text as a single chunk
        chunk = {
            "text": text,
            "metadata": metadata or {},
        }

        return [chunk]

        # TODO: Implement with LlamaIndex SentenceSplitter
        # nodes = self._text_splitter.get_nodes_from_documents([LlamaDocument(text=text, metadata=metadata or {})])
        #
        # return [
        #     {
        #         "text": node.text,
        #         "metadata": node.metadata
        #     }
        #     for node in nodes
        # ]

    async def ingest_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest raw text into the memory system.

        Args:
        ----
            text: Text to ingest
            metadata: Optional metadata for the text
            source_name: Optional source name for the text

        Returns:
        -------
            Dict: Ingestion result

        """
        await self.ensure_initialized()

        try:
            logger.info(f"Ingesting text (length: {len(text)})")

            # Generate a unique ID if source name not provided
            source_id = source_name or f"text_{uuid.uuid4().hex[:8]}"

            # Add timestamp metadata
            full_metadata = metadata or {}
            full_metadata.update(
                {
                    "source": source_id,
                    "source_type": "text",
                    "ingestion_time": datetime.now(UTC).isoformat(),
                },
            )

            # Split text into chunks
            chunks = self._split_text(text, full_metadata)
            logger.info(f"Text split into {len(chunks)} chunks")

            # Process each chunk
            stored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk["metadata"].copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                )

                # Generate embedding
                embedding = self._embedding_manager.generate_embedding(chunk["text"])

                # Store in memory
                # For now, we'll add it to vector memory via Mem0
                # TODO: Also add to Supabase for original content storage
                mem0_id = self._memory_manager._mem0.add(
                    content=chunk["text"],
                    user_id=self._settings.mem0_memory_id,
                    metadata=chunk_metadata,
                    embedding=embedding,
                )

                stored_chunks.append(
                    {
                        "mem0_id": mem0_id,
                        "metadata": chunk_metadata,
                    },
                )

            # Create an entity in the graph for this document
            entity_name = source_name or f"Document: {source_id}"

            await self._memory_manager.create_entities(
                [
                    {
                        "name": entity_name,
                        "entityType": "Document",
                        "observations": [
                            f"Text content with {len(chunks)} chunks. First 100 chars: {text[:100]}...",
                        ],
                    },
                ],
            )

            logger.info(f"Successfully ingested text as '{entity_name}'")

            return {
                "success": True,
                "source_id": source_id,
                "entity_name": entity_name,
                "chunk_count": len(chunks),
                "stored_chunks": stored_chunks,
            }

        except Exception as e:
            logger.exception(f"Failed to ingest text: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def ingest_file(
        self,
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
            Dict: Ingestion result

        """
        await self.ensure_initialized()

        try:
            logger.info(f"Ingesting file: {file_path}")

            # Load the file
            documents = self._file_loader.load_file(file_path, metadata)

            if not documents:
                return {
                    "success": False,
                    "error": f"Failed to load file: {file_path}",
                }

            # Process each document
            results = []
            for doc in documents:
                result = await self.ingest_text(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    source_name=f"File: {doc['metadata'].get('file_name', file_path)}",
                )
                results.append(result)

            return {
                "success": True,
                "file_path": file_path,
                "document_count": len(documents),
                "results": results,
            }

        except Exception as e:
            logger.exception(f"Failed to ingest file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
        exclude_hidden: bool = True,
        file_extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest all files from a directory.

        Args:
        ----
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            metadata: Optional metadata for all documents
            exclude_hidden: Whether to exclude hidden files/dirs
            file_extensions: List of file extensions to include

        Returns:
        -------
            Dict: Ingestion result

        """
        await self.ensure_initialized()

        try:
            logger.info(f"Ingesting directory: {directory_path}")

            # Load the directory
            documents = self._file_loader.load_directory(
                directory_path,
                recursive,
                metadata,
                exclude_hidden,
                file_extensions,
            )

            if not documents:
                return {
                    "success": False,
                    "error": f"No documents found in directory: {directory_path}",
                }

            # Process each document
            results = []
            for doc in documents:
                result = await self.ingest_text(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    source_name=f"File: {doc['metadata'].get('file_name', 'unknown')}",
                )
                results.append(result)

            return {
                "success": True,
                "directory_path": directory_path,
                "document_count": len(documents),
                "results": results,
            }

        except Exception as e:
            logger.exception(f"Failed to ingest directory {directory_path}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def ingest_url(
        self,
        url: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest content from a URL.

        Args:
        ----
            url: The URL to ingest
            metadata: Optional metadata for the URL

        Returns:
        -------
            Dict: Ingestion result

        """
        await self.ensure_initialized()

        try:
            logger.info(f"Ingesting URL: {url}")

            # Load the URL
            documents = self._web_loader.load_url(url, metadata)

            if not documents:
                return {
                    "success": False,
                    "error": f"Failed to load URL: {url}",
                }

            # Process each document
            results = []
            for doc in documents:
                result = await self.ingest_text(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    source_name=f"URL: {url}",
                )
                results.append(result)

            return {
                "success": True,
                "url": url,
                "document_count": len(documents),
                "results": results,
            }

        except Exception as e:
            logger.exception(f"Failed to ingest URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Create a singleton instance for import
ingestion_pipeline = IngestionPipeline()
