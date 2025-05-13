"""
Worker agent implementation for EMVR.

This module implements the worker agent that performs specialized tasks.
"""

import logging
from typing import Any

from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel

from emvr.agents.base import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)


class WorkerAgent(BaseAgent):
    """
    Worker agent that performs specialized tasks.

    This agent is responsible for handling specific types of tasks
    delegated by the supervisor agent.
    """

    def __init__(
        self,
        name: str,
        description: str,
        specialty: str,
        llm: BaseLanguageModel,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the worker agent.

        Args:
        ----
            name: Agent name
            description: Agent description
            specialty: Agent specialty (e.g., "research", "coding", "analysis")
            llm: Language model to use
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        self.specialty = specialty

        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                f"You are {name}, a specialized agent for {specialty}. {description} "
                "Think step-by-step to solve the task assigned to you. "
                "Use the tools available to you when necessary. "
                "Provide a clear and detailed response."
            )

        # Initialize the base agent
        super().__init__(
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            memory_enabled=memory_enabled,
        )

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

            # Get context if provided
            context = kwargs.get("context", [])

            # Create a context string from the context
            context_str = ""
            if context:
                context_str = "Context Information:\n"
                for i, doc in enumerate(context):
                    content = doc.get("content", "")
                    source = doc.get("source", "Unknown")
                    context_str += f"[{i + 1}] From {source}: {content}\n\n"

            # Combine context and input
            full_input = (
                f"{context_str}\n\nTask: {input_text}" if context_str else input_text
            )

            # Execute the agent
            result = await self.agent_executor.ainvoke(
                {
                    "input": full_input,
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
            logger.exception(f"Worker agent execution failed: {e}")
            return {
                "response": f"I encountered an error: {e!s}",
                "error": str(e),
                "status": "error",
            }
