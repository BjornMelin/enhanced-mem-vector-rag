"""Fusion retriever implementation combining multiple retrieval methods."""

import asyncio
import logging
from typing import Any

from emvr.memory.graph_store import Neo4jMemoryStore
from emvr.memory.vector_store import QdrantMemoryStore
from emvr.retrieval.base import BaseRetriever, RetrievalResult
from emvr.retrieval.hybrid_retriever import HybridRetriever
from emvr.retrieval.knowledge_graph_retriever import (
    KnowledgeGraphRetriever as KGRetriever,
)

# Configure logging
logger = logging.getLogger(__name__)


class FusionRetriever(BaseRetriever):
    """Fusion retriever combining vector search, keyword search, and graph context."""

    def __init__(
        self,
        vector_store: QdrantMemoryStore | None = None,
        graph_store: Neo4jMemoryStore | None = None,
        vector_weight: float = 0.4,
        graph_weight: float = 0.4,
        web_weight: float = 0.2,
        top_k_multiplier: int = 3,
        reranking: bool = True,
    ) -> None:
        """
        Initialize the fusion retriever.

        Args:
        ----
            vector_store: Vector store for semantic search
            graph_store: Graph store for knowledge graph queries
            vector_weight: Weight for vector search results (0.0-1.0)
            graph_weight: Weight for graph search results (0.0-1.0)
            web_weight: Weight for web search results (0.0-1.0)
            top_k_multiplier: Multiplier for initial retrieval (for reranking)
            reranking: Whether to apply reranking to combined results

        """
        # Initialize stores
        self.vector_store = vector_store or QdrantMemoryStore()
        self.graph_store = graph_store or Neo4jMemoryStore()

        # Initialize retrievers
        self.vector_retriever = HybridRetriever(
            vector_store=self.vector_store,
            use_reranking=False,  # We'll do our own reranking
        )

        self.graph_retriever = KGRetriever(
            graph_store=self.graph_store,
            include_text=True,
        )

        # Could add web retriever here later
        self.web_retriever = None

        # Weights for results fusion
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.web_weight = web_weight

        # Reranking settings
        self.top_k_multiplier = top_k_multiplier
        self.reranking = reranking

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents using multiple retrieval methods.

        Args:
        ----
            query: Query string
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
        -------
            List of retrieval results

        """
        try:
            logger.info(f"Performing fusion retrieval for query: {query}")

            # Determine how many results to fetch from each source initially
            initial_top_k = top_k * self.top_k_multiplier if self.reranking else top_k

            # Gather results from multiple sources in parallel
            retrieval_tasks = []

            # Vector retrieval task
            if self.vector_weight > 0:
                retrieval_tasks.append(
                    self.vector_retriever.retrieve(
                        query=query,
                        top_k=initial_top_k,
                        filters=filters,
                    ),
                )

            # Graph retrieval task
            if self.graph_weight > 0:
                retrieval_tasks.append(
                    self.graph_retriever.retrieve(
                        query=query,
                        top_k=initial_top_k,
                        filters=filters,
                    ),
                )

            # Web retrieval task (placeholder for future)
            if self.web_weight > 0 and self.web_retriever is not None:
                retrieval_tasks.append(
                    self.web_retriever.retrieve(
                        query=query,
                        top_k=initial_top_k,
                    ),
                )

            # Execute all retrieval tasks in parallel
            results = await asyncio.gather(*retrieval_tasks)

            # Extract results by source (this assumes the same order as retrieval_tasks)
            source_results = {}
            result_idx = 0

            if self.vector_weight > 0:
                source_results["vector"] = results[result_idx]
                result_idx += 1

            if self.graph_weight > 0:
                source_results["graph"] = results[result_idx]
                result_idx += 1

            if self.web_weight > 0 and self.web_retriever is not None:
                source_results["web"] = results[result_idx]
                result_idx += 1

            # Combine and potentially rerank results
            combined_results = self._combine_results(
                query=query,
                source_results=source_results,
                top_k=top_k,
            )

            logger.info(f"Fusion retrieval returned {len(combined_results)} results")
            return combined_results

        except Exception as e:
            logger.exception(f"Error in fusion retrieval: {e!s}")
            return []

    def _combine_results(
        self,
        query: str,
        source_results: dict[str, list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Combine results from multiple sources with weights.

        Args:
        ----
            query: Original query string
            source_results: Dictionary mapping source names to result lists
            top_k: Number of final results to return

        Returns:
        -------
            Combined and reranked list of retrieval results

        """
        # Create a dictionary to track results by ID (to avoid duplicates)
        combined_dict = {}

        # Process vector results
        if "vector" in source_results:
            for result in source_results["vector"]:
                result_id = result.id

                if result_id not in combined_dict:
                    # Copy the result and adjust score
                    combined_dict[result_id] = RetrievalResult(
                        id=result.id,
                        text=result.text,
                        score=(result.score or 0.5) * self.vector_weight,
                        metadata={
                            **result.metadata,
                            "sources": ["vector"],
                        },
                    )
                else:
                    # Update existing result
                    existing = combined_dict[result_id]
                    existing.score += (result.score or 0.5) * self.vector_weight
                    existing.metadata["sources"].append("vector")

        # Process graph results
        if "graph" in source_results:
            for result in source_results["graph"]:
                result_id = result.id

                if result_id not in combined_dict:
                    # Copy the result and adjust score
                    combined_dict[result_id] = RetrievalResult(
                        id=result.id,
                        text=result.text,
                        score=(result.score or 0.5) * self.graph_weight,
                        metadata={
                            **result.metadata,
                            "sources": ["graph"],
                        },
                    )
                else:
                    # Update existing result
                    existing = combined_dict[result_id]
                    existing.score += (result.score or 0.5) * self.graph_weight

                    if "graph" not in existing.metadata["sources"]:
                        existing.metadata["sources"].append("graph")

                    # Enhance text with graph information if not already included
                    if (
                        "source_entity" in result.metadata
                        and "relation" in result.metadata
                    ):
                        source = result.metadata["source_entity"]
                        relation = result.metadata["relation"]
                        target = result.metadata["target_entity"]

                        graph_info = f"\nRelated: {source} --{relation}--> {target}"

                        if graph_info not in existing.text:
                            existing.text += graph_info

        # Process web results
        if "web" in source_results:
            for result in source_results["web"]:
                result_id = result.id

                if result_id not in combined_dict:
                    # Copy the result and adjust score
                    combined_dict[result_id] = RetrievalResult(
                        id=result.id,
                        text=result.text,
                        score=(result.score or 0.5) * self.web_weight,
                        metadata={
                            **result.metadata,
                            "sources": ["web"],
                        },
                    )
                else:
                    # Update existing result
                    existing = combined_dict[result_id]
                    existing.score += (result.score or 0.5) * self.web_weight

                    if "web" not in existing.metadata["sources"]:
                        existing.metadata["sources"].append("web")

        # Convert to list and sort by score
        combined_list = list(combined_dict.values())
        combined_list.sort(
            key=lambda x: x.score if x.score is not None else 0,
            reverse=True,
        )

        # Apply reranking if enabled
        if self.reranking:
            combined_list = self._rerank_results(query, combined_list)

        # Return top-k results
        return combined_list[:top_k]

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank results based on relevance to query and source diversity.

        Note: In a production system, this would use a more sophisticated reranker
        model (e.g., cross-encoder). This is a simplified version for demonstration.

        Args:
        ----
            query: Original query string
            results: List of retrieval results to rerank

        Returns:
        -------
            Reranked list of retrieval results

        """
        query_terms = set(query.lower().split())

        for result in results:
            # Base score from retrieval sources
            base_score = result.score or 0.5

            # Term overlap score
            text_terms = set(result.text.lower().split())
            overlap = len(query_terms.intersection(text_terms))
            overlap_score = min(overlap / max(len(query_terms), 1), 1.0) * 0.3

            # Source diversity bonus (reward results from multiple sources)
            source_count = len(result.metadata.get("sources", []))
            diversity_score = min(source_count / 3, 1.0) * 0.2

            # Combine scores
            result.score = (base_score * 0.5) + overlap_score + diversity_score

        # Sort by adjusted score
        results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

        return results
