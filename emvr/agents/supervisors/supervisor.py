"""
Supervisor agent implementation for EMVR.

This module implements the supervisor agent that orchestrates worker agents.
"""

import logging
from enum import Enum
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from emvr.agents.base import BaseAgent
from emvr.agents.tools.ingestion_tools import get_ingestion_tools
from emvr.agents.tools.memory_tools import get_memory_tools
from emvr.agents.tools.retrieval_tools import get_retrieval_tools
from emvr.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


# ----- Agent States -----


class AgentState(Enum):
    """States for the supervisor agent workflow."""

    PLANNING = "planning"
    RETRIEVING = "retrieving"
    INGESTING = "ingesting"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    RESPONDING = "responding"


# ----- State Management -----


class SupervisorState(dict[str, Any]):
    """State object for the supervisor agent workflow."""

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Get the messages from the state."""
        return self.get("messages", [])

    @property
    def current_state(self) -> AgentState:
        """Get the current state."""
        return self.get("current_state", AgentState.PLANNING)

    @property
    def plan(self) -> dict[str, Any] | None:
        """Get the current plan."""
        return self.get("plan")

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the retrieved context."""
        return self.get("context", [])

    @property
    def execution_results(self) -> list[dict[str, Any]]:
        """Get the execution results."""
        return self.get("execution_results", [])

    @property
    def reflection(self) -> str | None:
        """Get the reflection."""
        return self.get("reflection")

    @property
    def final_response(self) -> str | None:
        """Get the final response."""
        return self.get("final_response")


# ----- Supervisor Agent -----


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that orchestrates worker agents.

    This agent is responsible for managing the workflow and delegating
    tasks to specialized worker agents.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        worker_agents: dict[str, BaseAgent] | None = None,
        additional_tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        memory_enabled: bool = True,
    ) -> None:
        """
        Initialize the supervisor agent.

        Args:
        ----
            llm: Language model to use
            worker_agents: Worker agents to delegate tasks to
            additional_tools: Additional tools for the supervisor
            system_prompt: System prompt for the agent
            memory_enabled: Whether to enable memory for the agent

        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are the Supervisor Agent, responsible for orchestrating the workflow. "
                "Your job is to break down complex tasks, delegate to specialized agents, "
                "and synthesize their work into a coherent response. "
                "Think step-by-step to determine the best course of action."
            )

        # Initialize basic tools
        self.memory_tools = get_memory_tools()
        self.retrieval_tools = get_retrieval_tools()
        self.ingestion_tools = get_ingestion_tools()

        # Combine all tools
        all_tools = []
        all_tools.extend(self.memory_tools)
        all_tools.extend(self.retrieval_tools)
        all_tools.extend(self.ingestion_tools)
        if additional_tools:
            all_tools.extend(additional_tools)

        # Initialize the base agent
        super().__init__(
            name="Supervisor Agent",
            description="Orchestrates the workflow and delegates tasks to specialized agents",
            llm=llm,
            tools=all_tools,
            system_prompt=system_prompt,
            memory_enabled=memory_enabled,
        )

        # Store worker agents
        self.worker_agents = worker_agents or {}

        # Initialize the workflow
        self._initialize_workflow()

    def _initialize_workflow(self) -> None:
        """Initialize the LangGraph workflow."""
        self.settings = get_settings()

        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools)

        # Define the workflow
        workflow = StateGraph(SupervisorState)

        # Define nodes
        workflow.add_node(AgentState.PLANNING, self._planning_step)
        workflow.add_node(AgentState.RETRIEVING, self._retrieving_step)
        workflow.add_node(AgentState.INGESTING, self._ingesting_step)
        workflow.add_node(AgentState.ANALYZING, self._analyzing_step)
        workflow.add_node(AgentState.EXECUTING, self._executing_step)
        workflow.add_node(AgentState.REFLECTING, self._reflecting_step)
        workflow.add_node(AgentState.RESPONDING, self._responding_step)

        # Define edges
        workflow.add_edge(AgentState.PLANNING, AgentState.RETRIEVING)
        workflow.add_edge(AgentState.RETRIEVING, AgentState.ANALYZING)
        workflow.add_conditional_edges(
            AgentState.ANALYZING,
            self._analyze_condition,
            {
                "needs_ingestion": AgentState.INGESTING,
                "execute": AgentState.EXECUTING,
                "respond": AgentState.RESPONDING,
            },
        )
        workflow.add_edge(AgentState.INGESTING, AgentState.ANALYZING)
        workflow.add_edge(AgentState.EXECUTING, AgentState.REFLECTING)
        workflow.add_edge(AgentState.REFLECTING, AgentState.RESPONDING)
        workflow.add_edge(AgentState.RESPONDING, END)

        # Set the entry point
        workflow.set_entry_point(AgentState.PLANNING)

        # Compile the workflow
        self.workflow = workflow.compile()

    async def _planning_step(self, state: SupervisorState) -> SupervisorState:
        """
        Planning step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Create planning prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are the Planning Agent. Your job is to create a plan for solving "
                            "the user's request. Break down the task into steps, considering what "
                            "information you need and what actions to take."
                        ),
                    ),
                    HumanMessage(content=state.get("input", "")),
                ],
            )

            # Get the plan
            plan_result = await self.llm.ainvoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["plan"] = {
                "steps": plan_result.content.split("\n"),
                "current_step": 0,
            }
            new_state["current_state"] = AgentState.RETRIEVING

            return new_state
        except Exception as e:
            logger.exception(f"Planning step failed: {e}")
            # Return original state with error
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["current_state"] = AgentState.RESPONDING
            return new_state

    async def _retrieving_step(self, state: SupervisorState) -> SupervisorState:
        """
        Retrieving step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Get the query from the input
            query = state.get("input", "")

            # Use hybrid search to get relevant context
            retrieval_result = await self.tool_executor.ainvoke(
                {
                    "tool": "hybrid_search",
                    "tool_input": {
                        "query": query,
                        "limit": 10,
                        "rerank": True,
                    },
                },
            )

            # Update state
            new_state = state.copy()
            new_state["context"] = retrieval_result.get("results", [])
            new_state["current_state"] = AgentState.ANALYZING

            return new_state
        except Exception as e:
            logger.exception(f"Retrieving step failed: {e}")
            # Proceed to analysis even with error
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["context"] = []
            new_state["current_state"] = AgentState.ANALYZING
            return new_state

    async def _analyzing_step(self, state: SupervisorState) -> SupervisorState:
        """
        Analyzing step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Create context string from retrieved documents
            context_str = "\n\n".join(
                [
                    f"Document {i + 1}:\n{doc.get('content', '')}"
                    for i, doc in enumerate(state.get("context", []))
                ],
            )

            # Create analyzing prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are the Analysis Agent. Your job is to analyze the retrieved "
                            "context and determine the next action. There are three possibilities:\n"
                            "1. We need to ingest more information (respond with 'needs_ingestion')\n"
                            "2. We have enough information and need to execute the plan (respond with 'execute')\n"
                            "3. We have enough information to respond directly (respond with 'respond')\n"
                            "Provide your reasoning and then state your decision as a single word."
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f"User query: {state.get('input', '')}\n\n"
                            f"Plan: {state.get('plan', {}).get('steps', [])}\n\n"
                            f"Retrieved context: {context_str}"
                        ),
                    ),
                ],
            )

            # Get the analysis
            analysis_result = await self.llm.ainvoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["analysis"] = analysis_result.content

            return new_state
        except Exception as e:
            logger.exception(f"Analyzing step failed: {e}")
            # Default to execution on error
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["analysis"] = "execute"
            return new_state

    def _analyze_condition(self, state: SupervisorState) -> str:
        """
        Condition function for analyzing step.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Next state

        """
        analysis = state.get("analysis", "")

        if "needs_ingestion" in analysis.lower():
            return "needs_ingestion"
        if "execute" in analysis.lower():
            return "execute"
        return "respond"

    async def _ingesting_step(self, state: SupervisorState) -> SupervisorState:
        """
        Ingesting step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Create ingestion prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are the Ingestion Agent. Your job is to determine what "
                            "information needs to be ingested to answer the user's query. "
                            "This could be a URL, a file, or text provided by the user."
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f"User query: {state.get('input', '')}\n\n"
                            f"Plan: {state.get('plan', {}).get('steps', [])}\n\n"
                            f"Analysis: {state.get('analysis', '')}"
                        ),
                    ),
                ],
            )

            # Get the ingestion plan
            ingestion_result = await self.llm.ainvoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["ingestion_plan"] = ingestion_result.content

            # For now, we'll just add a placeholder for ingestion
            # In a real implementation, we would parse the ingestion plan
            # and execute the appropriate ingestion tools

            new_state["current_state"] = AgentState.ANALYZING

            return new_state
        except Exception as e:
            logger.exception(f"Ingesting step failed: {e}")
            # Proceed to analysis even with error
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["current_state"] = AgentState.ANALYZING
            return new_state

    async def _executing_step(self, state: SupervisorState) -> SupervisorState:
        """
        Executing step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Create context string from retrieved documents
            context_str = "\n\n".join(
                [
                    f"Document {i + 1}:\n{doc.get('content', '')}"
                    for i, doc in enumerate(state.get("context", []))
                ],
            )

            # Create execution prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are the Execution Agent. Your job is to execute the plan "
                            "using the retrieved context. Think step-by-step and provide "
                            "a detailed solution to the user's query."
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f"User query: {state.get('input', '')}\n\n"
                            f"Plan: {state.get('plan', {}).get('steps', [])}\n\n"
                            f"Retrieved context: {context_str}"
                        ),
                    ),
                ],
            )

            # Execute the plan
            execution_result = await self.llm.ainvoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["execution_result"] = execution_result.content
            new_state["current_state"] = AgentState.REFLECTING

            return new_state
        except Exception as e:
            logger.exception(f"Executing step failed: {e}")
            # Proceed to reflection even with error
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["execution_result"] = f"Error during execution: {e!s}"
            new_state["current_state"] = AgentState.REFLECTING
            return new_state

    async def _reflecting_step(self, state: SupervisorState) -> SupervisorState:
        """
        Reflecting step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Create reflection prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are the Reflection Agent. Your job is to reflect on the execution "
                            "of the plan and identify any gaps or areas for improvement. Provide "
                            "an objective assessment of the solution quality."
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f"User query: {state.get('input', '')}\n\n"
                            f"Plan: {state.get('plan', {}).get('steps', [])}\n\n"
                            f"Execution result: {state.get('execution_result', '')}"
                        ),
                    ),
                ],
            )

            # Get the reflection
            reflection_result = await self.llm.ainvoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["reflection"] = reflection_result.content
            new_state["current_state"] = AgentState.RESPONDING

            return new_state
        except Exception as e:
            logger.exception(f"Reflecting step failed: {e}")
            # Proceed to response even with error
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["reflection"] = f"Error during reflection: {e!s}"
            new_state["current_state"] = AgentState.RESPONDING
            return new_state

    async def _responding_step(self, state: SupervisorState) -> SupervisorState:
        """
        Responding step of the workflow.

        Args:
        ----
            state: Current state

        Returns:
        -------
            Updated state

        """
        try:
            # Determine what to include in the final response
            execution_result = state.get("execution_result", "")
            reflection = state.get("reflection", "")

            # Create response prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are the Response Agent. Your job is to create a clear, concise, "
                            "and helpful response to the user's query. Use the execution result "
                            "and reflection to craft your response. Be direct and to the point."
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f"User query: {state.get('input', '')}\n\n"
                            f"Execution result: {execution_result}\n\n"
                            f"Reflection: {reflection}"
                        ),
                    ),
                ],
            )

            # Get the final response
            response_result = await self.llm.ainvoke(prompt)

            # Update state
            new_state = state.copy()
            new_state["final_response"] = response_result.content

            return new_state
        except Exception as e:
            logger.exception(f"Responding step failed: {e}")
            # Provide error message as response
            new_state = state.copy()
            new_state["error"] = str(e)
            new_state["final_response"] = (
                "I apologize, but I encountered an error while preparing your response. "
                f"Error: {e!s}"
            )
            return new_state

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
            # Initialize state
            initial_state = SupervisorState(
                {
                    "input": input_text,
                    "messages": [],
                    "current_state": AgentState.PLANNING,
                },
            )

            # Execute the workflow
            result = await self.workflow.ainvoke(initial_state)

            # Return the result
            return {
                "response": result.get(
                    "final_response",
                    "I'm sorry, I couldn't generate a response.",
                ),
                "workflow_trace": {
                    "plan": result.get("plan"),
                    "context": result.get("context"),
                    "analysis": result.get("analysis"),
                    "execution_result": result.get("execution_result"),
                    "reflection": result.get("reflection"),
                },
                "status": "success",
            }
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            return {
                "response": f"I encountered an error: {e!s}",
                "error": str(e),
                "status": "error",
            }
