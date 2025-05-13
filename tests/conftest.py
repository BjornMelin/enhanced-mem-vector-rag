"""Test configuration and fixtures for the EMVR system."""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

# Create a fake fastmcp module structure
fake_fastmcp = types.ModuleType("fastmcp")
fake_server = types.ModuleType("fastmcp.server")


# Define a fake MCPServer class
class MockMCPServer:
    def __init__(self, name, description=None, version=None):
        self.name = name
        self.description = description
        self.version = version

    async def start_stdio(self):
        """Mock start_stdio method."""
        pass

    async def start_http(self, host=None, port=None):
        """Mock start_http method."""
        pass


# Add the MockMCPServer to the fake module
fake_server.MCPServer = MockMCPServer

# Add the fake server module to fake_fastmcp
sys.modules["fastmcp"] = fake_fastmcp
sys.modules["fastmcp.server"] = fake_server


# Create mock classes for various parts of the system
class MockAsyncMethod(AsyncMock):
    """Mock async method that can be imported and called."""

    pass


class MockClass(MagicMock):
    """Mock class that can be instantiated."""

    pass


# Create a mock settings class for config
class MockSettings:
    """Mock settings class."""

    def __init__(self):
        self.mcp_host = "localhost"
        self.mcp_port = 8080
        self.openai_model = "gpt-4o"


# Create a mock config module
fake_config = types.ModuleType("emvr.config")
fake_config.get_settings = MagicMock(return_value=MockSettings())
sys.modules["emvr.config"] = fake_config

# Create mock modules to avoid import errors
mock_modules = [
    "emvr.agents.orchestration",
    "emvr.agents.supervisors",
    "emvr.agents.supervisors.supervisor",
    "emvr.agents.tools.retrieval_tools",
    "emvr.agents.tools.ingestion_tools",
    "emvr.agents.tools.memory_tools",
    "emvr.retrieval.pipeline",
    "emvr.retrieval.knowledge_graph_retriever",
    "emvr.retrieval.hybrid_retriever",
    "emvr.memory.interfaces",
    "emvr.memory.interfaces.graphiti_interface",
    "emvr.core.embedding",
    "emvr.core.db_connections",
    "emvr.mcp_server.endpoints",
    "emvr.mcp_server.endpoints.agent_endpoints",
    "emvr.ingestion.pipeline",
    "emvr.memory.memory_manager",
]

for mod_name in mock_modules:
    sys.modules[mod_name] = MagicMock()

# Create mock pipeline objects with initialize method
ingestion_mock = MagicMock()
ingestion_mock.initialize = AsyncMock()
retrieval_mock = MagicMock()
retrieval_mock.initialize = AsyncMock()
memory_mock = MagicMock()
memory_mock.initialize = AsyncMock()
memory_mock.close = MagicMock()

# Set up the mocks in their respective modules
sys.modules["emvr.ingestion.pipeline"].ingestion_pipeline = ingestion_mock
sys.modules["emvr.retrieval.pipeline"].retrieval_pipeline = retrieval_mock
sys.modules["emvr.memory.memory_manager"].memory_manager = memory_mock

# Mock specific imported functions/classes
# Orchestration
sys.modules["emvr.agents.orchestration"].get_orchestrator = MagicMock(return_value=None)
sys.modules["emvr.agents.orchestration"].initialize_orchestration = AsyncMock()

# Supervisor/Agent
sys.modules["emvr.agents.supervisors.supervisor"].SupervisorAgent = MockClass

# Tools
sys.modules["emvr.agents.tools.retrieval_tools"].get_retrieval_tools = MagicMock(
    return_value=[],
)
sys.modules["emvr.agents.tools.ingestion_tools"].get_ingestion_tools = MagicMock(
    return_value=[],
)
sys.modules["emvr.agents.tools.memory_tools"].get_memory_tools = MagicMock(
    return_value=[],
)

# Database connections
sys.modules["emvr.core.db_connections"].initialize_connections = MagicMock()
sys.modules["emvr.core.db_connections"].close_connections = MagicMock()

# Endpoints
sys.modules["emvr.mcp_server.endpoints"].register_endpoints = AsyncMock()
sys.modules["emvr.mcp_server.endpoints"].register_resources = AsyncMock()
sys.modules[
    "emvr.mcp_server.endpoints.agent_endpoints"
].register_agent_endpoints = AsyncMock()
sys.modules[
    "emvr.mcp_server.endpoints.agent_endpoints"
].register_agent_resources = AsyncMock()
