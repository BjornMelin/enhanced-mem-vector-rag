"""
Agent orchestration module for EMVR.

This module implements the main orchestration framework for the agent system.
"""

import logging
from typing import Any

from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel

from emvr.agents.base import BaseAgent
from emvr.agents.supervisors import SupervisorAgent
from emvr.agents.workers.specialized import (
    AnalysisAgent,
    CreativeAgent,
    IngestionAgent,
    ResearchAgent,
)
from emvr.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Agent orchestration framework for EMVR.

    This class creates and manages a team of agents, including a supervisor
    and specialized workers.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: list[BaseTool] | None = None,
        custom_agents: dict[str, BaseAgent] | None = None,
    ) -> None:
        """
        Initialize the agent orchestrator.

        Args:
        ----
            llm: Language model to use for all agents
            additional_tools: Additional tools to provide to agents
            custom_agents: Custom agent implementations to use

        """
        self.llm = llm
        self.additional_tools = additional_tools or []
        self.settings = get_settings()

        # Initialize worker agents
        self.workers = self._initialize_workers(custom_agents)

        # Initialize supervisor agent
        self.supervisor = self._initialize_supervisor()

        logger.info("Agent orchestrator initialized with supervisor and workers")

    def _initialize_workers(
        self,
        custom_agents: dict[str, BaseAgent] | None = None,
    ) -> dict[str, BaseAgent]:
        """
        Initialize the worker agents.

        Args:
        ----
            custom_agents: Custom agent implementations to use

        Returns:
        -------
            Dict of worker agents

        """
        workers = {}

        # Use custom implementations if provided
        if custom_agents:
            workers.update(custom_agents)

        # Create default workers if not overridden
        if "research" not in workers:
            workers["research"] = ResearchAgent(
                llm=self.llm,
                additional_tools=self.additional_tools,
                memory_enabled=True,
            )

        if "ingestion" not in workers:
            workers["ingestion"] = IngestionAgent(
                llm=self.llm,
                additional_tools=self.additional_tools,
                memory_enabled=True,
            )

        if "analysis" not in workers:
            workers["analysis"] = AnalysisAgent(
                llm=self.llm,
                additional_tools=self.additional_tools,
                memory_enabled=True,
            )

        if "creative" not in workers:
            workers["creative"] = CreativeAgent(
                llm=self.llm,
                additional_tools=self.additional_tools,
                memory_enabled=True,
            )

        return workers

    def _initialize_supervisor(self) -> SupervisorAgent:
        """
        Initialize the supervisor agent.

        Returns
        -------
            Supervisor agent

        """
        return SupervisorAgent(
            llm=self.llm,
            worker_agents=self.workers,
            additional_tools=self.additional_tools,
            memory_enabled=True,
        )

    async def run(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        """
        Run the agent system on the given input.

        Args:
        ----
            input_text: Input text to process
            kwargs: Additional arguments

        Returns:
        -------
            Dict containing the agent's response and any additional information

        """
        try:
            # Execute the supervisor agent
            return await self.supervisor.run(input_text, **kwargs)

            # Return the result
        except Exception as e:
            logger.exception(f"Agent orchestration failed: {e}")
            return {
                "response": f"I encountered an error: {e!s}",
                "error": str(e),
                "status": "error",
            }

    async def run_worker(
        self,
        worker_name: str,
        input_text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run a specific worker agent on the given input.

        Args:
        ----
            worker_name: Name of the worker agent to run
            input_text: Input text to process
            kwargs: Additional arguments

        Returns:
        -------
            Dict containing the agent's response and any additional information

        """
        try:
            # Check if worker exists
            if worker_name not in self.workers:
                msg = f"Worker agent '{worker_name}' not found"
                raise ValueError(msg)

            # Execute the worker agent
            return await self.workers[worker_name].run(input_text, **kwargs)

            # Return the result
        except Exception as e:
            logger.exception(f"Worker agent execution failed: {e}")
            return {
                "response": f"I encountered an error: {e!s}",
                "error": str(e),
                "status": "error",
            }

    async def shutdown(self) -> None:
        """Shutdown the agent system and clean up resources."""
        # Clean up tasks if needed


# Global orchestrator instance
_orchestrator: AgentOrchestrator | None = None


async def initialize_orchestration(
    llm: BaseLanguageModel,
    additional_tools: list[BaseTool] | None = None,
    custom_agents: dict[str, BaseAgent] | None = None,
) -> AgentOrchestrator:
    """
    Initialize the agent orchestration system.

    Args:
    ----
        llm: Language model to use for all agents
        additional_tools: Additional tools to provide to agents
        custom_agents: Custom agent implementations to use

    Returns:
    -------
        Agent orchestrator instance

    """
    global _orchestrator

    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(
            llm=llm,
            additional_tools=additional_tools,
            custom_agents=custom_agents,
        )

    return _orchestrator


def get_orchestrator() -> AgentOrchestrator | None:
    """
    Get the global orchestrator instance.

    Returns
    -------
        Agent orchestrator instance if initialized, None otherwise

    """
    return _orchestrator
