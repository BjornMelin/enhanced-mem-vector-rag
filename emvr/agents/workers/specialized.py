"""
Specialized worker agents for EMVR.

This module implements specialized worker agents for specific tasks.
"""

import logging

from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel

from emvr.agents.tools.ingestion_tools import get_ingestion_tools
from emvr.agents.tools.memory_tools import get_memory_tools
from emvr.agents.tools.retrieval_tools import get_retrieval_tools
from emvr.agents.workers.worker import WorkerAgent

# Configure logging
logger = logging.getLogger(__name__)


class ResearchAgent(WorkerAgent):
    """
    Research agent specializing in information gathering and synthesis.

    This agent excels at retrieving information, finding connections,
    and synthesizing knowledge.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the research agent.

        Args:
        ----
            llm: Language model to use
            additional_tools: Additional tools for the agent
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are the Research Agent, specializing in information gathering and synthesis. "
                "Your strengths include retrieving relevant information, finding connections between "
                "concepts, and synthesizing knowledge into coherent summaries. "
                "Always cite your sources and provide evidence for your claims. "
                "Think step-by-step to ensure thorough research."
            )

        # Initialize tools
        tools = []
        tools.extend(get_retrieval_tools())
        tools.extend(get_memory_tools())
        if additional_tools:
            tools.extend(additional_tools)

        # Initialize the worker agent
        super().__init__(
            name="Research Agent",
            description="Specializes in information gathering and synthesis",
            specialty="research",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            memory_enabled=memory_enabled,
        )


class IngestionAgent(WorkerAgent):
    """
    Ingestion agent specializing in data processing and storage.

    This agent excels at handling various data formats, extracting
    relevant information, and storing it in the memory system.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the ingestion agent.

        Args:
        ----
            llm: Language model to use
            additional_tools: Additional tools for the agent
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are the Ingestion Agent, specializing in data processing and storage. "
                "Your strengths include handling various data formats, extracting relevant "
                "information, and storing it in the memory system. "
                "Be thorough and methodical in your processing, ensuring all important "
                "information is captured and properly organized."
            )

        # Initialize tools
        tools = []
        tools.extend(get_ingestion_tools())
        tools.extend(get_memory_tools())
        if additional_tools:
            tools.extend(additional_tools)

        # Initialize the worker agent
        super().__init__(
            name="Ingestion Agent",
            description="Specializes in data processing and storage",
            specialty="ingestion",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            memory_enabled=memory_enabled,
        )


class AnalysisAgent(WorkerAgent):
    """
    Analysis agent specializing in data analysis and pattern recognition.

    This agent excels at finding patterns, drawing insights, and
    providing detailed analysis of information.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the analysis agent.

        Args:
        ----
            llm: Language model to use
            additional_tools: Additional tools for the agent
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are the Analysis Agent, specializing in data analysis and pattern recognition. "
                "Your strengths include finding patterns, drawing insights, and providing "
                "detailed analysis of information. "
                "Think critically and consider multiple perspectives when analyzing information. "
                "Provide clear reasoning for your conclusions and identify any limitations "
                "in your analysis."
            )

        # Initialize tools
        tools = []
        tools.extend(get_retrieval_tools())
        tools.extend(get_memory_tools())
        if additional_tools:
            tools.extend(additional_tools)

        # Initialize the worker agent
        super().__init__(
            name="Analysis Agent",
            description="Specializes in data analysis and pattern recognition",
            specialty="analysis",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            memory_enabled=memory_enabled,
        )


class CreativeAgent(WorkerAgent):
    """
    Creative agent specializing in idea generation and creative solutions.

    This agent excels at thinking outside the box, generating novel
    ideas, and finding creative solutions to problems.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        additional_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the creative agent.

        Args:
        ----
            llm: Language model to use
            additional_tools: Additional tools for the agent
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are the Creative Agent, specializing in idea generation and creative solutions. "
                "Your strengths include thinking outside the box, generating novel ideas, "
                "and finding creative solutions to problems. "
                "Don't be constrained by conventional thinking - explore unusual connections "
                "and possibilities. Balance creativity with practicality to ensure your "
                "ideas can be implemented."
            )

        # Initialize tools
        tools = []
        tools.extend(get_retrieval_tools())
        tools.extend(get_memory_tools())
        if additional_tools:
            tools.extend(additional_tools)

        # Initialize the worker agent
        super().__init__(
            name="Creative Agent",
            description="Specializes in idea generation and creative solutions",
            specialty="creativity",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            memory_enabled=memory_enabled,
        )
