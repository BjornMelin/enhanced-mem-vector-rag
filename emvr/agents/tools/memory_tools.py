"""
Memory tools for EMVR agents.

This module implements tools for interacting with the memory system.
"""

import logging
from typing import Any

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from emvr.memory.memory_manager import memory_manager

# Configure logging
logger = logging.getLogger(__name__)


# ----- Tool Input/Output Schemas -----


class SearchNodesInput(BaseModel):
    """Input schema for search_nodes tool."""

    query: str = Field(..., description="The search query string")
    limit: int = Field(10, description="Maximum number of results to return")


class CreateEntitiesInput(BaseModel):
    """Input schema for create_entities tool."""

    entities: list[dict[str, Any]] = Field(
        ...,
        description="List of entities to create",
    )


class CreateRelationsInput(BaseModel):
    """Input schema for create_relations tool."""

    relations: list[dict[str, Any]] = Field(
        ...,
        description="List of relations to create",
    )


class AddObservationsInput(BaseModel):
    """Input schema for add_observations tool."""

    observations: list[dict[str, Any]] = Field(
        ...,
        description="List of observations to add",
    )


class DeleteEntitiesInput(BaseModel):
    """Input schema for delete_entities tool."""

    entity_names: list[str] = Field(..., description="List of entity names to delete")


# ----- Memory Tools -----


@tool
async def search_memory(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Search the memory system for nodes matching the query.

    Args:
    ----
        query: The search query string
        limit: Maximum number of results to return

    Returns:
    -------
        Dict containing search results

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Execute the search
        return await memory_manager.search_nodes(query, limit)

    except Exception as e:
        logger.exception(f"Memory search failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def read_memory_graph() -> dict[str, Any]:
    """
    Read the entire memory graph.

    Returns
    -------
        Dict containing the complete graph structure

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Read the graph
        return await memory_manager.read_graph()

    except Exception as e:
        logger.exception(f"Reading memory graph failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def create_memory_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create new entities in the memory system.

    Args:
    ----
        entities: List of entities to create, each with name, entityType, and observations

    Returns:
    -------
        Dict containing the result of the operation

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Create the entities
        return await memory_manager.create_entities(entities)

    except Exception as e:
        logger.exception(f"Entity creation failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def create_memory_relations(relations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create new relations between entities in the memory system.

    Args:
    ----
        relations: List of relations to create, each with from, to, and relationType

    Returns:
    -------
        Dict containing the result of the operation

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Create the relations
        return await memory_manager.create_relations(relations)

    except Exception as e:
        logger.exception(f"Relation creation failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def add_memory_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Add new observations to existing entities in the memory system.

    Args:
    ----
        observations: List of observations to add, each with entityName and contents

    Returns:
    -------
        Dict containing the result of the operation

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Add the observations
        return await memory_manager.add_observations(observations)

    except Exception as e:
        logger.exception(f"Adding observations failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


@tool
async def delete_memory_entities(entity_names: list[str]) -> dict[str, Any]:
    """
    Delete entities from the memory system.

    Args:
    ----
        entity_names: List of entity names to delete

    Returns:
    -------
        Dict containing the result of the operation

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Delete the entities
        return await memory_manager.delete_entities(entity_names)

    except Exception as e:
        logger.exception(f"Entity deletion failed: {e}")
        return {
            "error": str(e),
            "status": "error",
        }


# ----- Tool Collection -----


def get_memory_tools() -> list[BaseTool]:
    """
    Get all memory tools.

    Returns
    -------
        List of memory tools

    """
    return [
        search_memory,
        read_memory_graph,
        create_memory_entities,
        create_memory_relations,
        add_memory_observations,
        delete_memory_entities,
    ]
