"""Knowledge Graph retriever implementation."""

import logging
from typing import Any

from llama_index.core import QueryBundle
from llama_index.core.graph_stores import Neo4jGraphStore
from llama_index.core.query_engine import KnowledgeGraphQueryEngine

from emvr.memory.graph_store import Neo4jMemoryStore
from emvr.retrieval.base import BaseRetriever, RetrievalResult

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeGraphRetriever(BaseRetriever):
    """Knowledge Graph retriever using Neo4j and LlamaIndex."""

    def __init__(
        self,
        graph_store: Neo4jMemoryStore | None = None,
        include_text: bool = True,
    ) -> None:
        """
        Initialize the knowledge graph retriever.

        Args:
        ----
            graph_store: Neo4j memory store
            include_text: Whether to include text in results

        """
        # Initialize graph store
        self.graph_store = graph_store or Neo4jMemoryStore()

        # LlamaIndex graph store for query engine
        self.llama_graph_store = Neo4jGraphStore(
            username=self.graph_store.username,
            password=self.graph_store.password,
            url=self.graph_store.uri,
            database=self.graph_store.database,
        )

        # Configurable parameters
        self.include_text = include_text

        # Initialize query engine
        self._query_engine = None

    @property
    def query_engine(self):
        """Get the knowledge graph query engine, lazy-loading if needed."""
        if self._query_engine is None:
            self._query_engine = KnowledgeGraphQueryEngine(
                graph_store=self.llama_graph_store,
                include_text=self.include_text,
            )
        return self._query_engine

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve knowledge graph paths based on a query.

        Args:
        ----
            query: Query string
            top_k: Number of results to return (used as a limit in Cypher)
            filters: Optional filters to apply (entity types, relation types)

        Returns:
        -------
            List of retrieval results

        """
        try:
            logger.info(f"Performing knowledge graph retrieval for query: {query}")

            # Create query bundle
            QueryBundle(query)

            # Additional Cypher parameters based on filters
            cypher_params = {}
            cypher_where_clause = ""

            if filters:
                # Apply entity type filter
                if "entity_types" in filters:
                    entity_types = filters["entity_types"]
                    if isinstance(entity_types, list) and len(entity_types) > 0:
                        cypher_params["entityTypes"] = entity_types
                        cypher_where_clause += "AND n.entity_type IN $entityTypes "

                # Apply relation type filter
                if "relation_types" in filters:
                    relation_types = filters["relation_types"]
                    if isinstance(relation_types, list) and len(relation_types) > 0:
                        cypher_params["relationTypes"] = relation_types
                        cypher_where_clause += "AND r.type IN $relationTypes "

            # Custom query based on the natural language query
            # This is a simplified approach - in a real implementation,
            # you would use a more sophisticated method to convert natural language to Cypher
            if "implement" in query.lower() or "build" in query.lower():
                cypher_query = f"""
                MATCH (n:Entity)-[r:RELATION]->(m:Entity)
                WHERE r.type = 'implements' OR r.type = 'uses_framework' {cypher_where_clause}
                RETURN n, r, m
                LIMIT {top_k}
                """
            elif "related" in query.lower() or "connection" in query.lower():
                cypher_query = f"""
                MATCH (n:Entity)-[r:RELATION]->(m:Entity)
                WHERE n.name CONTAINS $keyword OR m.name CONTAINS $keyword {cypher_where_clause}
                RETURN n, r, m
                LIMIT {top_k}
                """
                # Extract likely keyword from query
                tokens = query.lower().split()
                for token in tokens:
                    if len(token) > 3 and token not in [
                        "what",
                        "who",
                        "when",
                        "where",
                        "how",
                        "related",
                        "connection",
                    ]:
                        cypher_params["keyword"] = token
                        break
                else:
                    cypher_params["keyword"] = ""
            else:
                # Default query with basic entity matching
                cypher_query = f"""
                MATCH (n:Entity)-[r:RELATION]->(m:Entity)
                WHERE toLower(n.name) CONTAINS toLower($query) OR toLower(m.name) CONTAINS toLower($query) {cypher_where_clause}
                RETURN n, r, m
                LIMIT {top_k}
                """
                cypher_params["query"] = query

            # Execute custom Cypher query
            async with self.graph_store.driver.session(
                database=self.graph_store.database,
            ) as session:
                result = await session.run(cypher_query, **cypher_params)

                # Process results
                retrieval_results = []
                async for record in result:
                    source = record["n"]
                    relation = record["r"]
                    target = record["m"]

                    result_id = f"{source.id}-{relation.id}-{target.id}"
                    result_text = (
                        f"{source['name']} --{relation['type']}--> {target['name']}"
                    )

                    # Get observations for entities if available
                    source_obs_query = """
                    MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
                    RETURN o.text as text
                    LIMIT 3
                    """
                    target_obs_query = """
                    MATCH (e:Entity {name: $name})-[:HAS_OBSERVATION]->(o:Observation)
                    RETURN o.text as text
                    LIMIT 3
                    """

                    source_obs_result = await session.run(
                        source_obs_query,
                        name=source["name"],
                    )
                    target_obs_result = await session.run(
                        target_obs_query,
                        name=target["name"],
                    )

                    source_obs = []
                    async for obs_record in source_obs_result:
                        source_obs.append(obs_record["text"])

                    target_obs = []
                    async for obs_record in target_obs_result:
                        target_obs.append(obs_record["text"])

                    # Add observations to result text if available
                    if source_obs:
                        result_text += (
                            f"\nSource ({source['name']}): {'; '.join(source_obs)}"
                        )
                    if target_obs:
                        result_text += (
                            f"\nTarget ({target['name']}): {'; '.join(target_obs)}"
                        )

                    # Create retrieval result
                    retrieval_results.append(
                        RetrievalResult(
                            id=result_id,
                            text=result_text,
                            score=1.0,  # No scoring in basic graph retrieval
                            metadata={
                                "source_entity": source["name"],
                                "source_type": source["entity_type"],
                                "relation": relation["type"],
                                "target_entity": target["name"],
                                "target_type": target["entity_type"],
                            },
                        ),
                    )

                logger.info(f"Found {len(retrieval_results)} knowledge graph results")
                return retrieval_results

        except Exception as e:
            logger.exception(f"Error in knowledge graph retrieval: {e!s}")
            return []
