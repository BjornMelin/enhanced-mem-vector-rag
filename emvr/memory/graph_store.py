"""Graph store implementation using Neo4j and Graphiti."""

import os
from typing import Any

from dotenv import load_dotenv

# Temporarily comment out LlamaIndex import
# from llama_index.core.graph_stores import Neo4jGraphStore
from neo4j import AsyncGraphDatabase

from emvr.memory.base import Entity, Relation

# Load environment variables
load_dotenv()


class Neo4jMemoryStore:
    """Graph memory store implementation using Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ) -> None:
        """
        Initialize the Neo4j memory store.

        Args:
        ----
            uri: URI of the Neo4j server (defaults to env var NEO4J_URI)
            username: Username for the Neo4j server (defaults to env var NEO4J_USERNAME)
            password: Password for the Neo4j server (defaults to env var NEO4J_PASSWORD)
            database: Neo4j database name

        """
        self.uri = uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.database = database

        # Initialize Neo4j driver
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

        # Temporarily comment out LlamaIndex graph store
        # self.graph_store = Neo4jGraphStore(
        #     username=self.username,
        #     password=self.password,
        #     url=self.uri,
        #     database=self.database,
        # )

    async def create_entity(self, entity: Entity) -> dict[str, Any]:
        """
        Create a new entity in the knowledge graph.

        Args:
        ----
            entity: Entity to create

        Returns:
        -------
            Created entity information

        """
        query = """
        CREATE (e:`Entity` {name: $name, entity_type: $entity_type})
        RETURN e
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                name=entity.name,
                entity_type=entity.entity_type,
            )
            record = await result.single()
            entity_node = record["e"]

            # Create observations
            for observation in entity.observations:
                await self._add_observation(entity.name, observation)

            return {
                "id": entity_node.id,
                "name": entity_node["name"],
                "entity_type": entity_node["entity_type"],
            }

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
        created = []
        for entity in entities:
            result = await self.create_entity(entity)
            created.append(result)

        return {"created": created}

    async def create_relation(self, relation: Relation) -> dict[str, Any]:
        """
        Create a new relation between entities in the knowledge graph.

        Args:
        ----
            relation: Relation to create

        Returns:
        -------
            Created relation information

        """
        query = """
        MATCH (from:`Entity` {name: $from_entity})
        MATCH (to:`Entity` {name: $to_entity})
        CREATE (from)-[r:`RELATION` {type: $relation_type}]->(to)
        RETURN from, r, to
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                from_entity=relation.from_entity,
                to_entity=relation.to_entity,
                relation_type=relation.relation_type,
            )
            record = await result.single()

            return {
                "from": record["from"]["name"],
                "relation": record["r"]["type"],
                "to": record["to"]["name"],
            }

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
        created = []
        for relation in relations:
            result = await self.create_relation(relation)
            created.append(result)

        return {"created": created}

    async def _add_observation(self, entity_name: str, observation: str) -> None:
        """
        Add an observation to an entity.

        Args:
        ----
            entity_name: Name of the entity
            observation: Observation text

        """
        query = """
        MATCH (e:`Entity` {name: $entity_name})
        CREATE (o:`Observation` {text: $observation})
        CREATE (e)-[r:`HAS_OBSERVATION`]->(o)
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                entity_name=entity_name,
                observation=observation,
            )

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
        for observation in observations:
            await self._add_observation(entity_name, observation)

        return {
            "entity": entity_name,
            "added_observations": len(observations),
        }

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
        query = """
        UNWIND $entity_names AS entity_name
        MATCH (e:`Entity` {name: entity_name})
        OPTIONAL MATCH (e)-[r]-()
        DELETE r, e
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                entity_names=entity_names,
            )

        return {
            "deleted": entity_names,
        }

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
        query = """
        MATCH (e:`Entity` {name: $entity_name})-[r:`HAS_OBSERVATION`]->(o:`Observation`)
        WHERE o.text IN $observations
        DELETE r, o
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                entity_name=entity_name,
                observations=observations,
            )

        return {
            "entity": entity_name,
            "deleted_observations": len(observations),
        }

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
        query = """
        UNWIND $relations AS relation
        MATCH (from:`Entity` {name: relation.from_entity})
                -[r:`RELATION` {type: relation.relation_type}]->
              (to:`Entity` {name: relation.to_entity})
        DELETE r
        """

        relation_data = [
            {
                "from_entity": r.from_entity,
                "relation_type": r.relation_type,
                "to_entity": r.to_entity,
            }
            for r in relations
        ]

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                relations=relation_data,
            )

        return {
            "deleted": len(relations),
        }

    async def read_graph(self) -> dict[str, Any]:
        """
        Read the entire knowledge graph.

        Returns
        -------
            Dictionary with entities and relations

        """
        entities_query = """
        MATCH (e:`Entity`)
        OPTIONAL MATCH (e)-[r:`HAS_OBSERVATION`]->(o:`Observation`)
        RETURN e.name AS name, e.entity_type AS entity_type,
               COLLECT(o.text) AS observations
        """

        relations_query = """
        MATCH (from:`Entity`)-[r:`RELATION`]->(to:`Entity`)
        RETURN from.name AS from_entity, r.type AS relation_type,
               to.name AS to_entity
        """

        entities = []
        relations = []

        async with self.driver.session(database=self.database) as session:
            # Get entities
            result = await session.run(entities_query)
            async for record in result:
                entities.append(
                    {
                        "name": record["name"],
                        "entity_type": record["entity_type"],
                        "observations": record["observations"],
                    },
                )

            # Get relations
            result = await session.run(relations_query)
            async for record in result:
                relations.append(
                    {
                        "from": record["from_entity"],
                        "relation": record["relation_type"],
                        "to": record["to_entity"],
                    },
                )

        return {
            "entities": entities,
            "relations": relations,
        }

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
        search_query = """
        MATCH (e:`Entity`)
        WHERE e.name CONTAINS $query OR e.entity_type CONTAINS $query
        OPTIONAL MATCH (e)-[r:`HAS_OBSERVATION`]->(o:`Observation`)
        WHERE o.text CONTAINS $query
        WITH e, COLLECT(DISTINCT o.text) AS matching_observations
        OPTIONAL MATCH (e)-[r2:`HAS_OBSERVATION`]->(o2:`Observation`)
        RETURN e.name AS name, e.entity_type AS entity_type,
               matching_observations,
               COLLECT(DISTINCT o2.text) AS all_observations
        """

        entities = []

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                search_query,
                query=query,
            )
            async for record in result:
                entities.append(
                    {
                        "name": record["name"],
                        "entity_type": record["entity_type"],
                        "matching_observations": record["matching_observations"],
                        "all_observations": record["all_observations"],
                    },
                )

        return {
            "query": query,
            "entities": entities,
        }

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
        query = """
        UNWIND $names AS name
        MATCH (e:`Entity` {name: name})
        OPTIONAL MATCH (e)-[r:`HAS_OBSERVATION`]->(o:`Observation`)
        RETURN e.name AS name, e.entity_type AS entity_type,
               COLLECT(o.text) AS observations
        """

        entities = []

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                names=names,
            )
            async for record in result:
                entities.append(
                    {
                        "name": record["name"],
                        "entity_type": record["entity_type"],
                        "observations": record["observations"],
                    },
                )

        return {
            "entities": entities,
        }
