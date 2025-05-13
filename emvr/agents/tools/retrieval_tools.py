"""
Retrieval tools for EMVR agents.

This module implements tools for interacting with the retrieval pipeline.
"""

import logging
from typing import Any

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from emvr.retrieval.pipeline import retrieval_pipeline

# Configure logging
logger = logging.getLogger(__name__)


# ----- Tool Input/Output Schemas -----


class RetrieveInput(BaseModel):
    """Input schema for hybrid_search tool."""

    query: str = Field(..., description="The search query string")
    limit: int = Field(10, description="Maximum number of results to return")
    rerank: bool = Field(True, description="Whether to rerank results")


class RetrieveAndGenerateInput(BaseModel):
    """Input schema for retrieve_and_generate tool."""

    query: str = Field(..., description="The search query string")
    limit: int = Field(10, description="Maximum number of results to return")
    context_limit: int = Field(
        5,
        description="Maximum number of context documents to include",
    )
    rerank: bool = Field(True, description="Whether to rerank results")


# ----- Retrieval Tools -----


@tool
async def hybrid_search(
    query: str,
    limit: int = 10,
    rerank: bool = True,
) -> dict[str, Any]:
    """
    Perform a hybrid search across vector and graph stores.

    Args:
    ----
        query: The search query string
        limit: Maximum number of results to return
        rerank: Whether to rerank results

    Returns:
    -------
        Dict containing search results

    """
    try:
        # Initialize retrieval pipeline if needed
        await retrieval_pipeline.initialize()

        # Execute the search
        result = await retrieval_pipeline.retrieve(
            query=query,
            limit=limit,
            rerank=rerank,
        )

        return {
            "results": result,
            "status": "success",
        }
    except Exception as e:
        logger.exception(f"Hybrid search failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def vector_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Perform a vector search using embeddings.

    Args:
    ----
        query: The search query string
        limit: Maximum number of results to return

    Returns:
    -------
        Dict containing search results

    """
    try:
        # Initialize retrieval pipeline if needed
        await retrieval_pipeline.initialize()

        # Execute the search
        result = await retrieval_pipeline.hybrid_retriever.vector_search(
            query=query,
            limit=limit,
        )

        return {
            "results": result,
            "status": "success",
        }
    except Exception as e:
        logger.exception(f"Vector search failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def graph_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Perform a graph search using knowledge graph.

    Args:
    ----
        query: The search query string
        limit: Maximum number of results to return

    Returns:
    -------
        Dict containing search results

    """
    try:
        # Initialize retrieval pipeline if needed
        await retrieval_pipeline.initialize()

        # Execute the search
        result = await retrieval_pipeline.graph_retriever.retrieve(
            query=query,
            limit=limit,
        )

        return {
            "results": result,
            "status": "success",
        }
    except Exception as e:
        logger.exception(f"Graph search failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def retrieve_and_generate(
    query: str,
    limit: int = 10,
    context_limit: int = 5,
    rerank: bool = True,
) -> dict[str, Any]:
    """
    Retrieve information and generate a response.

    Args:
    ----
        query: The search query string
        limit: Maximum number of results to return
        context_limit: Maximum number of context documents to include
        rerank: Whether to rerank results

    Returns:
    -------
        Dict containing the generated response and retrieved context

    """
    try:
        # Initialize retrieval pipeline if needed
        await retrieval_pipeline.initialize()

        # Execute the retrieval and generation
        result = await retrieval_pipeline.retrieve_and_generate(
            query=query,
            limit=limit,
            context_limit=context_limit,
            rerank=rerank,
        )

        return {
            "response": result["response"],
            "context": result["context"],
            "sources": result["sources"],
            "status": "success",
        }
    except Exception as e:
        logger.exception(f"Retrieve and generate failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def find_entities(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Extract entities from text and find them in the knowledge graph.

    Args:
    ----
        query: The text to extract entities from
        limit: Maximum number of entities to return

    Returns:
    -------
        Dict containing the found entities

    """
    try:
        # Initialize retrieval pipeline if needed
        await retrieval_pipeline.initialize()

        # Extract entities
        result = await retrieval_pipeline.graph_retriever.extract_entities(
            text=query,
            limit=limit,
        )

        return {
            "entities": result,
            "status": "success",
        }
    except Exception as e:
        logger.exception(f"Entity extraction failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def find_relationships(
    entity_name: str,
    relation_type: str | None = None,
    direction: str = "outgoing",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Find relationships for an entity in the knowledge graph.

    Args:
    ----
        entity_name: The name of the entity to find relationships for
        relation_type: Optional type of relation to filter by
        direction: Direction of relationships ("outgoing", "incoming", or "both")
        limit: Maximum number of relationships to return

    Returns:
    -------
        Dict containing the found relationships

    """
    try:
        # Initialize retrieval pipeline if needed
        await retrieval_pipeline.initialize()

        # Find relationships
        result = await retrieval_pipeline.graph_retriever.find_relationships(
            entity_name=entity_name,
            relation_type=relation_type,
            direction=direction,
            limit=limit,
        )

        return {
            "relationships": result,
            "status": "success",
        }
    except Exception as e:
        logger.exception(f"Relationship finding failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


# ----- Tool Collection -----


def get_retrieval_tools() -> list[BaseTool]:
    """
    Get all retrieval tools.

    Returns
    -------
        List of retrieval tools

    """
    return [
        hybrid_search,
        vector_search,
        graph_search,
        retrieve_and_generate,
        find_entities,
        find_relationships,
    ]
