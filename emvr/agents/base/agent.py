"""
Base agent implementation for EMVR.

This module implements the base agent class that all agents will inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel

from emvr.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base agent class that all agents will inherit from.

    This class provides common functionality for all agents, including
    initialization, tool registration, and execution.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the base agent.

        Args:
        ----
            name: Agent name
            description: Agent description
            llm: Language model to use
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.memory_enabled = memory_enabled
        self.settings = get_settings()

        # Set default system prompt if not provided
        if system_prompt is None:
            self.system_prompt = (
                f"You are {name}, {description}. "
                "Think step-by-step to determine the best course of action. "
                "Use the tools available to you when necessary."
            )
        else:
            self.system_prompt = system_prompt

        # Initialize the agent with tools
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the agent with its tools and prompt."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )

        # Create the agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.settings.debug_mode,
            handle_parsing_errors=True,
            max_iterations=self.settings.max_agent_iterations or 10,
            return_intermediate_steps=True,
        )

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent.

        Args:
        ----
            tool: Tool to add

        """
        self.tools.append(tool)
        self._initialize_agent()  # Reinitialize with the new tool

    def add_tools(self, tools: list[BaseTool]) -> None:
        """
        Add multiple tools to the agent.

        Args:
        ----
            tools: List of tools to add

        """
        self.tools.extend(tools)
        self._initialize_agent()  # Reinitialize with new tools

    @abstractmethod
    async def run(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        """
        Run the agent on the given input.

        Args:
        ----
            input_text: Input text to process
            kwargs: Additional arguments

        Returns:
        -------
            Dict containing the agent's response and any additional information

        """


class SimpleAgent(BaseAgent):
    """
    Simple agent implementation that directly uses the agent executor.

    This class provides a concrete implementation of the BaseAgent
    that can be used directly for simple use cases.
    """

    async def run(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        """
        Run the agent on the given input.

        Args:
        ----
            input_text: Input text to process
            kwargs: Additional arguments

        Returns:
        -------
            Dict containing the agent's response and any additional information

        """
        try:
            # Get chat history if provided
            chat_history = kwargs.get("chat_history", [])

            # Execute the agent
            result = await self.agent_executor.ainvoke(
                {
                    "input": input_text,
                    "chat_history": chat_history,
                },
            )

            # Return the result
            return {
                "response": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "status": "success",
            }
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            return {
                "response": f"I encountered an error: {e!s}",
                "error": str(e),
                "status": "error",
            }
