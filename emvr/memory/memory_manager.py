"""Memory manager implementation integrating vector and graph stores."""

from typing import Any

from emvr.memory.base import Entity, MemoryInterface, Relation
from emvr.memory.graph_store import Neo4jMemoryStore
from emvr.memory.vector_store import QdrantMemoryStore


class MemoryManager(MemoryInterface):
    """Memory manager integrating vector and graph stores."""

    def __init__(
        self,
        vector_store: QdrantMemoryStore | None = None,
        graph_store: Neo4jMemoryStore | None = None,
    ) -> None:
        """
        Initialize the memory manager.

        Args:
        ----
            vector_store: Vector store for semantic memory
            graph_store: Graph store for structured memory

        """
        self.vector_store = vector_store or QdrantMemoryStore()
        self.graph_store = graph_store or Neo4jMemoryStore()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize vector and graph stores."""
        if self._initialized:
            return

        # Initialize vector store
        # In a real implementation, this would properly initialize the vector store

        # Initialize graph store
        # In a real implementation, this would properly initialize the graph store

        self._initialized = True

    def close(self) -> None:
        """Close connections and clean up resources."""
        if not self._initialized:
            return

        # Close vector store connections
        # In a real implementation, this would properly close vector store connections

        # Close graph store connections
        # In a real implementation, this would properly close graph store connections

        self._initialized = False

    async def create_entities(self, entities: list[Entity]) -> dict[str, Any]:
        """
        Create multiple new entities in the knowledge graph.

        Args:
        ----
            entities: List of entities to create

        Returns:
        -------
            Dictionary with created entities information

        """
        entities_data = [
            Entity(
                name=entity.name,
                entity_type=entity.entity_type,
                observations=entity.observations,
            )
            for entity in entities
        ]

        return await self.graph_store.create_entities(entities_data)

        # Also index entities in vector store for semantic search
        # This would be implemented based on specific requirements

    async def create_relations(self, relations: list[Relation]) -> dict[str, Any]:
        """
        Create multiple new relations between entities in the knowledge graph.

        Args:
        ----
            relations: List of relations to create

        Returns:
        -------
            Dictionary with created relations information

        """
        relations_data = [
            Relation(
                from_entity=relation.from_entity,
                relation_type=relation.relation_type,
                to_entity=relation.to_entity,
            )
            for relation in relations
        ]

        return await self.graph_store.create_relations(relations_data)

    async def add_observations(
        self,
        entity_name: str,
        observations: list[str],
    ) -> dict[str, Any]:
        """
        Add new observations to an existing entity in the knowledge graph.

        Args:
        ----
            entity_name: Name of the entity
            observations: List of observation texts

        Returns:
        -------
            Dictionary with operation result

        """
        return await self.graph_store.add_observations(entity_name, observations)

    async def delete_entities(self, entity_names: list[str]) -> dict[str, Any]:
        """
        Delete multiple entities and their associated relations from the knowledge graph.

        Args:
        ----
            entity_names: List of entity names to delete

        Returns:
        -------
            Dictionary with operation result

        """
        return await self.graph_store.delete_entities(entity_names)

    async def delete_observations(
        self,
        entity_name: str,
        observations: list[str],
    ) -> dict[str, Any]:
        """
        Delete specific observations from an entity in the knowledge graph.

        Args:
        ----
            entity_name: Name of the entity
            observations: List of observation texts to delete

        Returns:
        -------
            Dictionary with operation result

        """
        return await self.graph_store.delete_observations(entity_name, observations)

    async def delete_relations(self, relations: list[Relation]) -> dict[str, Any]:
        """
        Delete multiple relations from the knowledge graph.

        Args:
        ----
            relations: List of relations to delete

        Returns:
        -------
            Dictionary with operation result

        """
        relations_data = [
            Relation(
                from_entity=relation.from_entity,
                relation_type=relation.relation_type,
                to_entity=relation.to_entity,
            )
            for relation in relations
        ]

        return await self.graph_store.delete_relations(relations_data)

    async def read_graph(self) -> dict[str, Any]:
        """
        Read the entire knowledge graph.

        Returns
        -------
            Dictionary with entities and relations

        """
        return await self.graph_store.read_graph()

    async def search_nodes(self, query: str) -> dict[str, Any]:
        """
        Search for nodes in the knowledge graph based on a query.

        Args:
        ----
            query: Search query string

        Returns:
        -------
            Dictionary with matching entities

        """
        return await self.graph_store.search_nodes(query)

    async def open_nodes(self, names: list[str]) -> dict[str, Any]:
        """
        Open specific nodes in the knowledge graph by their names.

        Args:
        ----
            names: List of entity names

        Returns:
        -------
            Dictionary with matching entities

        """
        return await self.graph_store.open_nodes(names)

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform hybrid search across vector and graph stores.

        Args:
        ----
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search

        Returns:
        -------
            Dictionary with search results

        """
        # Perform vector search
        vector_results = await self.vector_store.hybrid_search(
            query,
            top_k=top_k,
            filters=filters,
        )

        # Perform graph search (simplified here)
        graph_results = await self.graph_store.search_nodes(query)

        # Combine results (in a real implementation, would need more sophisticated fusion)
        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results.get("entities", []),
        }


# Singleton instance
memory_manager = MemoryManager()
