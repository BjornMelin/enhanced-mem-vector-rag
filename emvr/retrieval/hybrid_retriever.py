"""Hybrid retriever implementation."""

import os
from typing import Any

import fastembed
from dotenv import load_dotenv
from llama_index.core.schema import NodeWithScore, QueryBundle

from emvr.memory.vector_store import QdrantMemoryStore
from emvr.retrieval.base import BaseRetriever, RetrievalResult

# Load environment variables
load_dotenv()


class HybridRetriever(BaseRetriever):
    """Hybrid retriever using vector search, keyword search, and graph context."""

    def __init__(
        self,
        vector_store: QdrantMemoryStore | None = None,
        embedding_model: str | None = None,
        use_reranking: bool = True,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
        ----
            vector_store: Vector store for retrieval
            embedding_model: Embedding model to use for encoding queries
            use_reranking: Whether to use reranking on results

        """
        # Initialize vector store
        self.vector_store = vector_store or QdrantMemoryStore()

        # Initialize embedding model
        self.embedding_model_name = embedding_model or os.environ.get(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5",
        )

        # Will lazy-load the embedding model when needed
        self._embedding_model = None

        # Reranking flag
        self.use_reranking = use_reranking

    @property
    def embedding_model(self):
        """Get the embedding model, lazy-loading if needed."""
        if self._embedding_model is None:
            self._embedding_model = fastembed.TextEmbedding(
                model_name=self.embedding_model_name,
                max_length=512,
            )
        return self._embedding_model

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
        # Create retriever with filtering
        retriever = self.vector_store.index.as_retriever(
            similarity_top_k=top_k * 2 if self.use_reranking else top_k,
            filters=filters,
        )

        # Create query bundle
        query_bundle = QueryBundle(query)

        # Retrieve similar documents
        nodes = await retriever.aretrieve(query_bundle)

        # Apply reranking if enabled
        if self.use_reranking:
            nodes = self._rerank_nodes(query, nodes, top_k)

        # Convert to retrieval results
        results = []
        for node in nodes[:top_k]:
            results.append(
                RetrievalResult(
                    id=node.node_id,
                    text=node.text,
                    score=node.score if hasattr(node, "score") else None,
                    metadata=node.metadata,
                ),
            )

        return results

    def _rerank_nodes(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_k: int,
    ) -> list[NodeWithScore]:
        """
        Rerank nodes based on relevance to query.

        Note: In a more complex implementation, this would use a dedicated
        reranker model. For simplicity, we're using a basic relevance score.

        Args:
        ----
            query: Query string
            nodes: List of nodes with scores
            top_k: Number of results to return

        Returns:
        -------
            Reranked list of nodes

        """
        # Simple term frequency based reranking
        for node in nodes:
            if not hasattr(node, "score") or node.score is None:
                node.score = 0.0

            # Calculate term overlap - very basic approach
            query_terms = set(query.lower().split())
            text_terms = set(node.text.lower().split())
            overlap = len(query_terms.intersection(text_terms))

            # Adjust score with term overlap (0.1 weight to original score)
            if overlap > 0:
                boost = min(overlap / len(query_terms), 1.0) * 0.5
                node.score = (node.score * 0.5) + boost

        # Sort by adjusted score
        reranked_nodes = sorted(
            nodes,
            key=lambda x: getattr(x, "score", 0.0),
            reverse=True,
        )

        return reranked_nodes[:top_k]
