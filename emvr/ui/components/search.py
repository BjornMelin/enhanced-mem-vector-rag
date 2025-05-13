"""
Search component for EMVR UI.

This module provides functionality for searching and retrieving information.
"""

import logging
from typing import Any

import chainlit as cl

from emvr.retrieval.pipeline import retrieval_pipeline

# Configure logging
logger = logging.getLogger(__name__)


# ----- Search Functions -----


async def perform_search(
    query: str,
    search_type: str = "hybrid",
    limit: int = 10,
    rerank: bool = True,
) -> dict[str, Any]:
    """
    Perform a search.

    Args:
    ----
        query: The search query
        search_type: The type of search (hybrid, vector, graph)
        limit: Maximum number of results to return
        rerank: Whether to rerank results

    Returns:
    -------
        Search results

    """
    try:
        # Get pipeline from session or initialize
        pipeline = cl.user_session.get("retrieval_pipeline")
        if pipeline is None:
            pipeline = retrieval_pipeline
            await pipeline.initialize()
            cl.user_session.set("retrieval_pipeline", pipeline)

        # Perform the appropriate search
        if search_type == "vector":
            result = await pipeline.hybrid_retriever.vector_search(
                query=query,
                limit=limit,
            )
        elif search_type == "graph":
            result = await pipeline.graph_retriever.retrieve(
                query=query,
                limit=limit,
            )
        else:  # hybrid (default)
            result = await pipeline.retrieve(
                query=query,
                limit=limit,
                rerank=rerank,
            )

        return {
            "results": result,
            "status": "success",
        }

    except Exception as e:
        logger.exception(f"Error in search: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


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
        query: The search query
        limit: Maximum number of search results
        context_limit: Maximum number of context documents to include
        rerank: Whether to rerank results

    Returns:
    -------
        Response and context

    """
    try:
        # Get pipeline from session or initialize
        pipeline = cl.user_session.get("retrieval_pipeline")
        if pipeline is None:
            pipeline = retrieval_pipeline
            await pipeline.initialize()
            cl.user_session.set("retrieval_pipeline", pipeline)

        # Retrieve and generate
        return await pipeline.retrieve_and_generate(
            query=query,
            limit=limit,
            context_limit=context_limit,
            rerank=rerank,
        )

    except Exception as e:
        logger.exception(f"Error in retrieve and generate: {e}")
        return {
            "response": f"Error: {e!s}",
            "context": [],
            "sources": [],
            "error": str(e),
            "status": "error",
        }


# ----- UI Components -----


async def show_search_ui() -> None:
    """Show the search UI."""
    # Create search elements
    elements = [
        cl.Text(
            name="query",
            label="Search Query",
            placeholder="What would you like to search for?",
        ),
        cl.Select(
            name="search_type",
            label="Search Type",
            values=[
                {"value": "hybrid", "label": "Hybrid (Vector + Graph)"},
                {"value": "vector", "label": "Vector Search Only"},
                {"value": "graph", "label": "Knowledge Graph Only"},
            ],
            initial="hybrid",
        ),
        cl.Slider(
            name="limit",
            label="Result Limit",
            min=1,
            max=50,
            step=1,
            initial=10,
        ),
        cl.Toggle(
            name="rerank",
            label="Rerank Results",
            initial=True,
        ),
    ]

    await cl.Message(
        content="What would you like to search for?",
        elements=elements,
        author="EMVR",
    ).send()


async def display_search_results(
    results: list[dict[str, Any]],
    query: str,
    search_type: str,
) -> None:
    """
    Display search results in the UI.

    Args:
    ----
        results: Search results
        query: The search query
        search_type: The type of search performed

    """
    if not results:
        await cl.Message(
            content=f"No results found for: '{query}'",
            author="EMVR",
        ).send()
        return

    # Format search results
    result_format = "\n\n".join(
        [
            f"**Result {i + 1}:** {result.get('title', 'Unknown')}\n"
            f"**Source:** {result.get('source', 'Unknown')}\n"
            f"**Score:** {result.get('score', 0):.4f}\n"
            f"**Content:**\n{result.get('content', 'No content available')}"
            for i, result in enumerate(results[:10])
        ],
    )

    # Create result message with elements
    elements = [
        cl.Text(
            name="filter",
            label="Filter Results",
            placeholder="Filter by keyword...",
        ),
        cl.Select(
            name="sort",
            label="Sort By",
            values=[
                {"value": "score", "label": "Relevance Score"},
                {"value": "recency", "label": "Recency"},
                {"value": "alphabetical", "label": "Alphabetical"},
            ],
            initial="score",
        ),
    ]

    header = f"🔍 **Search Results for:** '{query}'\n**Type:** {search_type}\n**Found:** {len(results)} results\n\n"

    await cl.Message(
        content=header + result_format,
        elements=elements,
        author="EMVR",
    ).send()
