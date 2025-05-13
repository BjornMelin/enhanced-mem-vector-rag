"""Base classes for retrieval."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any]


class BaseRetriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: Query string
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
        -------
            List of retrieval results

        """
