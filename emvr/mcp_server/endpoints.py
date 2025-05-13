"""
MCP server endpoints for the EMVR system.

This module implements the endpoints for the custom 'memory' MCP server.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, MCPServer
from pydantic import BaseModel, Field

from emvr.ingestion.pipeline import ingestion_pipeline
from emvr.memory.memory_manager import memory_manager

# Configure logging
logger = logging.getLogger(__name__)


# ----- Request/Response Models -----


class Entity(BaseModel):
    """Entity schema for memory operations."""

    name: str
    entityType: str
    observations: list[str]


class CreateEntitiesRequest(BaseModel):
    """Request schema for creating entities."""

    entities: list[Entity]


class Relation(BaseModel):
    """Relation schema for memory operations."""

    from_: str
    to: str
    relationType: str


class CreateRelationsRequest(BaseModel):
    """Request schema for creating relations."""

    relations: list[Relation]


class Observation(BaseModel):
    """Observation schema for memory operations."""

    entityName: str
    contents: list[str]


class AddObservationsRequest(BaseModel):
    """Request schema for adding observations."""

    observations: list[Observation]


class DeleteEntitiesRequest(BaseModel):
    """Request schema for deleting entities."""

    entityNames: list[str]


class SearchRequest(BaseModel):
    """Request schema for search operations."""

    query: str
    limit: int | None = 10


# ----- MCP Endpoint Functions -----


async def register_endpoints(mcp: MCPServer) -> None:
    """Register all memory MCP endpoints."""
    # ----- Memory Operations -----

    @mcp.tool()
    async def memory_create_entities(
        entities: Annotated[
            list[dict[str, Any]],
            Field(description="List of entities to create"),
        ],
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Create multiple new entities in the knowledge graph.

        Each entity should have a name, entityType, and observations.
        """
        try:
            await ctx.info(f"Creating {len(entities)} entities")

            # Initialize memory manager
            await memory_manager.initialize()

            # Process the request
            return await memory_manager.create_entities(entities)

        except Exception as e:
            logger.exception(f"Entity creation failed: {e}")
            await ctx.error(f"Failed to create entities: {e}")
            raise

    @mcp.tool()
    async def memory_create_relations(
        relations: Annotated[
            list[dict[str, Any]],
            Field(description="List of relations to create"),
        ],
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Create multiple new relations between entities in the knowledge graph.

        Each relation should have from, to, and relationType fields.
        """
        try:
            await ctx.info(f"Creating {len(relations)} relations")

            # Initialize memory manager
            await memory_manager.initialize()

            # Process the request
            return await memory_manager.create_relations(relations)

        except Exception as e:
            logger.exception(f"Relation creation failed: {e}")
            await ctx.error(f"Failed to create relations: {e}")
            raise

    @mcp.tool()
    async def memory_add_observations(
        observations: Annotated[
            list[dict[str, Any]],
            Field(description="List of observations to add"),
        ],
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Add new observations to existing entities in the knowledge graph.

        Each observation should have an entityName and contents (list of text observations).
        """
        try:
            await ctx.info(f"Adding observations to {len(observations)} entities")

            # Initialize memory manager
            await memory_manager.initialize()

            # Process the request
            return await memory_manager.add_observations(observations)

        except Exception as e:
            logger.exception(f"Adding observations failed: {e}")
            await ctx.error(f"Failed to add observations: {e}")
            raise

    @mcp.tool()
    async def memory_search_nodes(
        query: Annotated[str, Field(description="The search query string")],
        limit: Annotated[
            int,
            Field(description="Maximum number of results to return"),
        ] = 10,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Search for nodes in the knowledge graph based on a query.

        Uses hybrid search across vector and graph stores.
        """
        try:
            await ctx.info(f"Searching nodes with query: {query}")

            # Initialize memory manager
            await memory_manager.initialize()

            # Process the request
            return await memory_manager.search_nodes(query, limit)

        except Exception as e:
            logger.exception(f"Node search failed: {e}")
            await ctx.error(f"Failed to search nodes: {e}")
            raise

    @mcp.tool()
    async def memory_read_graph(
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Read the entire knowledge graph.

        Returns the complete graph structure with entities and relationships.
        """
        try:
            await ctx.info("Reading entire graph")

            # Initialize memory manager
            await memory_manager.initialize()

            # Process the request
            return await memory_manager.read_graph()

        except Exception as e:
            logger.exception(f"Reading graph failed: {e}")
            await ctx.error(f"Failed to read graph: {e}")
            raise

    @mcp.tool()
    async def memory_delete_entities(
        entityNames: Annotated[
            list[str],
            Field(description="List of entity names to delete"),
        ],
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Delete multiple entities and their associated relations from the knowledge graph.

        Use this with caution as it permanently removes entities and their relationships.
        """
        try:
            await ctx.info(f"Deleting {len(entityNames)} entities")

            # Initialize memory manager
            await memory_manager.initialize()

            # Process the request
            return await memory_manager.delete_entities(entityNames)

        except Exception as e:
            logger.exception(f"Entity deletion failed: {e}")
            await ctx.error(f"Failed to delete entities: {e}")
            raise

    # ----- Search Operations -----

    @mcp.tool()
    async def search_hybrid(
        query: Annotated[str, Field(description="The search query string")],
        limit: Annotated[
            int,
            Field(description="Maximum number of results to return"),
        ] = 10,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Perform a hybrid search across vector and graph stores.

        This endpoint combines vector search, keyword search, and graph search
        for comprehensive results.
        """
        try:
            await ctx.info(f"Performing hybrid search with query: {query}")

            # Initialize memory manager
            await memory_manager.initialize()

            # For now, hybrid search is the same as node search
            # In the future, this will be enhanced with additional capabilities
            return await memory_manager.search_nodes(query, limit)

        except Exception as e:
            logger.exception(f"Hybrid search failed: {e}")
            await ctx.error(f"Failed to perform hybrid search: {e}")
            raise

    # ----- Graph Operations -----

    @mcp.tool()
    async def graph_query(
        query: Annotated[str, Field(description="The Cypher query to execute")],
        parameters: Annotated[
            dict[str, Any] | None,
            Field(description="Query parameters"),
        ] = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Execute a Cypher query against the Neo4j graph database.

        Use this for custom graph queries when the standard tools aren't sufficient.
        """
        try:
            await ctx.info(f"Executing graph query: {query}")

            # Initialize memory manager and get Graphiti interface
            await memory_manager.initialize()

            # Execute the query
            return await memory_manager._graphiti.execute_cypher(query, parameters)

        except Exception as e:
            logger.exception(f"Graph query failed: {e}")
            await ctx.error(f"Failed to execute graph query: {e}")
            raise

    # ----- Ingestion Operations -----

    @mcp.tool()
    async def ingest_text(
        content: Annotated[str, Field(description="Text content to ingest")],
        metadata: Annotated[
            dict[str, Any] | None,
            Field(description="Optional metadata for the text"),
        ] = None,
        source_name: Annotated[
            str | None,
            Field(description="Optional source name for the text"),
        ] = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Ingest raw text into the memory system.

        Processes the text through the ingestion pipeline, including text splitting,
        embedding generation, and storage in both vector and graph memory.
        """
        try:
            await ctx.info(f"Ingesting text (length: {len(content)})")

            # Initialize ingestion pipeline
            await ingestion_pipeline.initialize()

            # Process the request
            return await ingestion_pipeline.ingest_text(content, metadata, source_name)

        except Exception as e:
            logger.exception(f"Text ingestion failed: {e}")
            await ctx.error(f"Failed to ingest text: {e}")
            raise

    @mcp.tool()
    async def ingest_file(
        file_path: Annotated[str, Field(description="Path to the file to ingest")],
        metadata: Annotated[
            dict[str, Any] | None,
            Field(description="Optional metadata for the file"),
        ] = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Ingest a file into the memory system.

        Supports various file types through LlamaIndex loaders. The file will be processed
        through the ingestion pipeline and stored in memory.
        """
        try:
            await ctx.info(f"Ingesting file: {file_path}")

            # Initialize ingestion pipeline
            await ingestion_pipeline.initialize()

            # Process the request
            return await ingestion_pipeline.ingest_file(file_path, metadata)

        except Exception as e:
            logger.exception(f"File ingestion failed: {e}")
            await ctx.error(f"Failed to ingest file: {e}")
            raise

    @mcp.tool()
    async def ingest_url(
        url: Annotated[str, Field(description="URL to ingest")],
        metadata: Annotated[
            dict[str, Any] | None,
            Field(description="Optional metadata for the URL content"),
        ] = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Ingest content from a URL into the memory system.

        Downloads and processes the content from the URL through the ingestion pipeline.
        """
        try:
            await ctx.info(f"Ingesting URL: {url}")

            # Initialize ingestion pipeline
            await ingestion_pipeline.initialize()

            # Process the request
            return await ingestion_pipeline.ingest_url(url, metadata)

        except Exception as e:
            logger.exception(f"URL ingestion failed: {e}")
            await ctx.error(f"Failed to ingest URL: {e}")
            raise

    @mcp.tool()
    async def ingest_directory(
        directory_path: Annotated[
            str,
            Field(description="Path to the directory to ingest"),
        ],
        recursive: Annotated[
            bool,
            Field(description="Whether to search subdirectories"),
        ] = True,
        metadata: Annotated[
            dict[str, Any] | None,
            Field(description="Optional metadata for all documents"),
        ] = None,
        exclude_hidden: Annotated[
            bool,
            Field(description="Whether to exclude hidden files/dirs"),
        ] = True,
        file_extensions: Annotated[
            list[str] | None,
            Field(description="List of file extensions to include"),
        ] = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Ingest all files from a directory into the memory system.

        Recursively processes all files from the directory through the ingestion pipeline.
        """
        try:
            await ctx.info(
                f"Ingesting directory: {directory_path} (recursive={recursive})",
            )

            # Initialize ingestion pipeline
            await ingestion_pipeline.initialize()

            # Process the request
            return await ingestion_pipeline.ingest_directory(
                directory_path,
                recursive,
                metadata,
                exclude_hidden,
                file_extensions,
            )

        except Exception as e:
            logger.exception(f"Directory ingestion failed: {e}")
            await ctx.error(f"Failed to ingest directory: {e}")
            raise


# ----- Resources (Static Knowledge) -----


async def register_resources(mcp: MCPServer) -> None:
    """Register all memory MCP resources."""

    @mcp.resource(
        uri="memory://api-guide",
        name="MemoryAPIGuide",
        description="Guide for using the memory API endpoints",
        mime_type="text/markdown",
    )
    async def memory_api_guide() -> str:
        """Return the Memory API usage guide."""
        return """
        # Memory MCP Server API Guide

        This MCP server provides access to knowledge graphs, vector search, memory management, and data ingestion.

        ## Memory Operations

        ### memory_create_entities
        Create multiple entities in the memory system:
        ```python
        result = await memory.create_entities([
            {"name": "EntityName", "entityType": "Component", "observations": ["Observation 1", "Observation 2"]}
        ])
        ```

        ### memory_create_relations
        Create relationships between entities:
        ```python
        result = await memory.create_relations([
            {"from": "EntityA", "to": "EntityB", "relationType": "depends_on"}
        ])
        ```

        ### memory_add_observations
        Add observations to existing entities:
        ```python
        result = await memory.add_observations([
            {"entityName": "EntityName", "contents": ["New observation"]}
        ])
        ```

        ### memory_search_nodes
        Search for nodes matching a query:
        ```python
        results = await memory.search_nodes("Search query", limit=5)
        ```

        ### memory_read_graph
        Get the complete knowledge graph:
        ```python
        graph = await memory.read_graph()
        ```

        ### memory_delete_entities
        Delete entities and their relations:
        ```python
        result = await memory.delete_entities(["EntityName1", "EntityName2"])
        ```

        ## Search Operations

        ### search_hybrid
        Perform hybrid search across vector and graph stores:
        ```python
        results = await search.hybrid("Search query", limit=5)
        ```

        ## Graph Operations

        ### graph_query
        Execute a Cypher query against Neo4j:
        ```python
        results = await graph.query("MATCH (n) RETURN n LIMIT 10")
        ```

        ## Ingestion Operations

        ### ingest_text
        Ingest raw text into memory:
        ```python
        result = await ingest.text("Text content to ingest", metadata={"source": "manual"})
        ```

        ### ingest_file
        Ingest a file into memory:
        ```python
        result = await ingest.file("/path/to/document.txt")
        ```

        ### ingest_url
        Ingest content from a URL:
        ```python
        result = await ingest.url("https://example.com/article")
        ```

        ### ingest_directory
        Ingest all files from a directory:
        ```python
        result = await ingest.directory(
            "/path/to/documents",
            recursive=True,
            file_extensions=["txt", "pdf", "md"]
        )
        ```

        ## Best Practices

        1. Always initialize a session with `memory.read_graph()`
        2. Create entities before creating relations
        3. Use descriptive entity types and relation types
        4. Keep observations concise and factual
        5. Use search_hybrid for semantic searches
        6. Use graph_query only for specialized graph queries
        7. Prefer ingest operations over manual memory creation for large content
        """
