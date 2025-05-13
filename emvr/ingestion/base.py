"""Base classes for the ingestion pipeline."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Document(BaseModel):
    """Document representation for ingestion."""

    content: str
    metadata: dict[str, Any]
    id: str | None = None


class IngestResult(BaseModel):
    """Result of an ingestion operation."""

    document_id: str
    success: bool
    message: str | None = None
    metadata: dict[str, Any] | None = None


class BaseIngestionConnector(ABC):
    """Base class for ingestion connectors."""

    @abstractmethod
    async def ingest(self, documents: list[Document]) -> list[IngestResult]:
        """
        Ingest documents into the system.

        Args:
        ----
            documents: List of documents to ingest

        Returns:
        -------
            List of ingestion results

        """

    @abstractmethod
    async def delete(self, document_ids: list[str]) -> list[IngestResult]:
        """
        Delete documents from the system.

        Args:
        ----
            document_ids: List of document IDs to delete

        Returns:
        -------
            List of deletion results

        """
