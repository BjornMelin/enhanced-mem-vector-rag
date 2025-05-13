"""Retrieval pipeline implementation."""

import logging
from typing import Any

from dotenv import load_dotenv

from emvr.memory.graph_store import Neo4jMemoryStore
from emvr.memory.memory_manager import MemoryManager
from emvr.memory.vector_store import QdrantMemoryStore
from emvr.retrieval.base import BaseRetriever
from emvr.retrieval.fusion_retriever import FusionRetriever
from emvr.retrieval.hybrid_retriever import HybridRetriever
from emvr.retrieval.knowledge_graph_retriever import KnowledgeGraphRetriever

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Main retrieval pipeline for the EMVR system."""

    def __init__(
        self,
        vector_store: QdrantMemoryStore | None = None,
        graph_store: Neo4jMemoryStore | None = None,
        memory_manager: MemoryManager | None = None,
        retrieval_mode: str = "fusion",
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
        ----
            vector_store: Vector store for semantic search
            graph_store: Graph store for knowledge graph queries
            memory_manager: Memory manager instance
            retrieval_mode: Retrieval mode ("vector", "graph", "hybrid", "fusion")

        """
        # Initialize stores
        self.vector_store = vector_store or QdrantMemoryStore()
        self.graph_store = graph_store or Neo4jMemoryStore()
        self.memory_manager = memory_manager or MemoryManager(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
        )

        # Set retrieval mode
        self.retrieval_mode = retrieval_mode

        # Initialize retrievers (lazy-loaded)
        self._vector_retriever = None
        self._graph_retriever = None
        self._hybrid_retriever = None
        self._fusion_retriever = None

        # Track active retriever
        self._active_retriever = None

        # Initialization status
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the retrieval pipeline."""
        if self._initialized:
            return

        logger.info("Initializing retrieval pipeline")

        # In a real implementation, this would initialize the vector store
        # and any other needed components

        self._initialized = True
        logger.info("Retrieval pipeline initialized")

    @property
    def vector_retriever(self) -> HybridRetriever:
        """Get the vector retriever."""
        if self._vector_retriever is None:
            self._vector_retriever = HybridRetriever(
                vector_store=self.vector_store,
                use_reranking=True,
            )
        return self._vector_retriever

    @property
    def graph_retriever(self) -> KnowledgeGraphRetriever:
        """Get the graph retriever."""
        if self._graph_retriever is None:
            self._graph_retriever = KnowledgeGraphRetriever(
                graph_store=self.graph_store,
                include_text=True,
            )
        return self._graph_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Get the hybrid retriever."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                use_reranking=True,
            )
        return self._hybrid_retriever

    @property
    def fusion_retriever(self) -> FusionRetriever:
        """Get the fusion retriever."""
        if self._fusion_retriever is None:
            self._fusion_retriever = FusionRetriever(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                vector_weight=0.4,
                graph_weight=0.4,
                web_weight=0.2,
                reranking=True,
            )
        return self._fusion_retriever

    @property
    def active_retriever(self) -> BaseRetriever:
        """Get the active retriever based on the current mode."""
        if self.retrieval_mode == "vector":
            return self.vector_retriever
        if self.retrieval_mode == "graph":
            return self.graph_retriever
        if self.retrieval_mode == "hybrid":
            return self.hybrid_retriever
        if self.retrieval_mode == "fusion":
            return self.fusion_retriever
        # Default to fusion if unknown mode
        return self.fusion_retriever

    def set_retrieval_mode(self, mode: str) -> None:
        """
        Set the retrieval mode.

        Args:
        ----
            mode: Retrieval mode ("vector", "graph", "hybrid", "fusion")

        Raises:
        ------
            ValueError: If the mode is invalid

        """
        valid_modes = ["vector", "graph", "hybrid", "fusion"]
        if mode not in valid_modes:
            msg = f"Invalid retrieval mode: {mode}. Must be one of {valid_modes}"
            raise ValueError(
                msg,
            )

        self.retrieval_mode = mode
        logger.info(f"Retrieval mode set to {mode}")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: Query string
            top_k: Number of results to return
            filters: Optional filters to apply
            mode: Override the retrieval mode for this query

        Returns:
        -------
            Dictionary with retrieval results and metadata

        """
        try:
            # Set temporary mode if provided
            original_mode = self.retrieval_mode
            if mode is not None:
                self.set_retrieval_mode(mode)

            # Get retriever
            retriever = self.active_retriever

            # Perform retrieval
            logger.info(
                f"Retrieving documents for query: '{query}' using {self.retrieval_mode} mode",
            )

            results = await retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            # Restore original mode
            if mode is not None:
                self.retrieval_mode = original_mode

            # Format and return results
            return {
                "query": query,
                "mode": self.retrieval_mode,
                "count": len(results),
                "results": [result.dict() for result in results],
            }

        except Exception as e:
            logger.exception(f"Error in retrieval pipeline: {e!s}")
            return {
                "query": query,
                "mode": self.retrieval_mode,
                "count": 0,
                "results": [],
                "error": str(e),
            }

    async def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform hybrid search using fusion retrieval.

        This is a convenience method for the MCP server.

        Args:
        ----
            query: Query string
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
        -------
            Dictionary with search results

        """
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            mode="fusion",
        )

    async def enrich_context(
        self,
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
            # Get retrieval results
            retrieval_results = await self.retrieve(
                query=query,
                top_k=top_k,
                mode="fusion",
            )

            # Extract result texts
            result_texts = []
            for result in retrieval_results.get("results", []):
                result_texts.append(result.get("text", ""))

            # Combine with existing context
            combined_context = context or ""
            if result_texts:
                if combined_context:
                    combined_context += "\n\nRelevant information:\n"
                else:
                    combined_context = "Relevant information:\n"

                combined_context += "\n\n".join(result_texts)

            return {
                "query": query,
                "original_context": context,
                "enriched_context": combined_context,
                "sources_count": len(result_texts),
            }

        except Exception as e:
            logger.exception(f"Error enriching context: {e!s}")
            return {
                "query": query,
                "original_context": context,
                "enriched_context": context,
                "sources_count": 0,
                "error": str(e),
            }


# Singleton instance
retrieval_pipeline = RetrievalPipeline()
