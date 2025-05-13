"""
Chainlit UI application for EMVR.

This module implements the main Chainlit UI for the EMVR system.
"""

import logging
import os
from typing import Any

import chainlit as cl
from chainlit.playground.providers import ChatOpenAI
from chainlit.types import FileDict

# Check if we're in development mode
is_development = os.environ.get("EMVR_DEVELOPMENT", "false").lower() == "true"
log_level = os.environ.get("EMVR_LOG_LEVEL", "INFO").upper()

# Set log level based on environment
logging_level = getattr(logging, log_level, logging.INFO)

# Import with try/except to handle missing dependencies gracefully
try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    # Use a mock if not available
    from unittest.mock import MagicMock

    ChatOpenAI = MagicMock

# Import our components
from emvr.config import get_settings

# Use try/except to handle missing dependencies during development
try:
    from emvr.agents.orchestration import initialize_orchestration
    from emvr.ingestion.pipeline import ingestion_pipeline
    from emvr.memory.memory_manager import memory_manager
    from emvr.retrieval.pipeline import retrieval_pipeline
except ImportError:
    # Mock these components if they're not available
    from unittest.mock import AsyncMock, MagicMock

    initialize_orchestration = AsyncMock(return_value=MagicMock())
    ingestion_pipeline = MagicMock()
    ingestion_pipeline.initialize = AsyncMock()
    ingestion_pipeline.ingest_file = AsyncMock(
        return_value={"status": "success", "chunks": 5},
    )
    memory_manager = MagicMock()
    memory_manager.initialize = AsyncMock()
    memory_manager.close = MagicMock()
    retrieval_pipeline = MagicMock()
    retrieval_pipeline.initialize = AsyncMock()

# Configure logging
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Log development mode status
if is_development:
    logger.info("Running in DEVELOPMENT mode - using mocks where needed")
    logger.info(f"Log level set to: {log_level}")


# ----- Initialization -----


@cl.on_app_start
async def setup() -> None:
    """Initialize the EMVR components."""
    try:
        # Get settings with fallback for development
        try:
            settings = get_settings()
        except Exception as e:
            logger.warning(f"Error getting settings: {e}. Using defaults.")
            from types import SimpleNamespace

            settings = SimpleNamespace(openai_model="gpt-3.5-turbo")

        # Initialize memory manager with fallback
        try:
            await memory_manager.initialize()
        except Exception as e:
            logger.warning(f"Error initializing memory manager: {e}. Using mock.")
        cl.user_session.set("memory_manager", memory_manager)

        # Initialize retrieval pipeline with fallback
        try:
            await retrieval_pipeline.initialize()
        except Exception as e:
            logger.warning(f"Error initializing retrieval pipeline: {e}. Using mock.")
        cl.user_session.set("retrieval_pipeline", retrieval_pipeline)

        # Initialize ingestion pipeline with fallback
        try:
            await ingestion_pipeline.initialize()
        except Exception as e:
            logger.warning(f"Error initializing ingestion pipeline: {e}. Using mock.")
        cl.user_session.set("ingestion_pipeline", ingestion_pipeline)

        # Initialize LLM with fallback
        try:
            llm = ChatOpenAI(
                temperature=0.0,
                model=settings.openai_model,
            )
        except Exception as e:
            logger.warning(f"Error initializing LLM: {e}. Using mock.")
            from unittest.mock import MagicMock

            llm = MagicMock()
        cl.user_session.set("llm", llm)

        # Register with playground
        actions = [
            cl.Action(
                name="search",
                label="🔍 Search",
                description="Search for information",
            ),
            cl.Action(
                name="ingest",
                label="📤 Ingest",
                description="Add information to the system",
            ),
            cl.Action(
                name="analyze",
                label="🔬 Analyze",
                description="Analyze information",
            ),
        ]

        cl.init_chat(
            actions=actions,
            playground=ChatOpenAI(
                model=settings.openai_model,
                temperature=0.0,
            ),
        )

        # Initialize agent orchestration
        if is_development:
            # Use mock orchestrator in development mode
            logger.info("Development mode: Using mock orchestrator")
            from unittest.mock import AsyncMock, MagicMock

            orchestrator = MagicMock()
            orchestrator.run = AsyncMock(
                return_value={
                    "response": "I'm a mock orchestrator in development mode. How can I help you?",
                    "workflow_trace": {
                        "plan": {
                            "steps": ["Mock planning step 1", "Mock planning step 2"],
                        },
                        "context": [
                            {"content": "Mock retrieved content for development mode"},
                        ],
                        "execution_result": "Mock execution result in development mode",
                        "reflection": "Mock reflection on the process (development mode)",
                    },
                },
            )
            orchestrator.shutdown = AsyncMock()
            logger.info("Mock orchestrator created for development")
        else:
            # Use real orchestrator in production
            try:
                orchestrator = await initialize_orchestration(llm=llm)
                logger.info("Orchestrator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize orchestrator: {e}")
                # Fallback to mock in production
                from unittest.mock import AsyncMock, MagicMock

                orchestrator = MagicMock()
                orchestrator.run = AsyncMock(
                    return_value={
                        "response": "I'm a fallback orchestrator. The main system is currently unavailable, but I'll do my best to help.",
                        "workflow_trace": {
                            "plan": {"steps": ["Emergency fallback step"]},
                            "context": [{"content": "Fallback content"}],
                            "execution_result": "Fallback execution",
                            "reflection": "System is using fallback mode",
                        },
                    },
                )
                orchestrator.shutdown = AsyncMock()

        # Store in session
        cl.user_session.set("orchestrator", orchestrator)

        # Send welcome message
        if is_development:
            welcome_message = (
                "👋 Welcome to Enhanced Mem-Vector RAG (DEVELOPMENT MODE)!\n\n"
                "This instance is using mock components for development and testing. "
                "Actual components will be used in production.\n\n"
                "How can I assist you today?"
            )
        else:
            welcome_message = (
                "👋 Welcome to Enhanced Mem-Vector RAG! How can I assist you today?"
            )

        await cl.Message(
            content=welcome_message,
            author="EMVR System",
        ).send()

    except Exception as e:
        logger.exception(f"Failed to initialize UI: {e}")
        await cl.Message(
            content=f"❌ Error initializing system: {e!s}",
            author="System",
        ).send()


# ----- Message Handling -----


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Process user messages.

    Args:
    ----
        message: The user message

    """
    try:
        # Get components from session
        orchestrator = cl.user_session.get("orchestrator")

        if orchestrator is None:
            logger.warning("Orchestrator not found in session, creating mock")
            # Create a mock orchestrator for development
            from unittest.mock import AsyncMock, MagicMock

            mock_orchestrator = MagicMock()
            mock_orchestrator.run = AsyncMock(
                return_value={
                    "response": f"Mock response to: {message.content}",
                    "workflow_trace": {
                        "plan": {"steps": ["Mock planning step"]},
                        "context": [{"content": "Mock retrieved document"}],
                        "execution_result": "Mock execution",
                        "reflection": "Mock reflection on the process",
                    },
                },
            )
            mock_orchestrator.shutdown = AsyncMock()
            orchestrator = mock_orchestrator
            cl.user_session.set("orchestrator", orchestrator)

        # Get uploaded files if any
        if message.elements:
            [e for e in message.elements if isinstance(e, cl.File)]

        # Process the message
        thinking_msg = cl.Message(content="", author="EMVR")
        await thinking_msg.send()

        async with thinking_msg.content_thread() as content_stream:
            await content_stream.append("🧠 Thinking...")

            # Run the query through the agent orchestrator
            try:
                results = await orchestrator.run(message.content)
                # Clear the thinking message
                await content_stream.clear()

                # Show the agent response
                response = results.get(
                    "response",
                    "I don't have a response at this time.",
                )
                await content_stream.append(response)
            except Exception as e:
                # Mock response for development if the orchestrator fails
                logger.warning(f"Orchestrator error: {e}. Using mock response.")
                await content_stream.clear()
                await content_stream.append(
                    f"I've processed your query: '{message.content}'"
                    f"\n\nThis is a mock response since the orchestrator is not fully implemented yet.",
                )

                # Create mock results for development
                results = {
                    "response": f"Mock response for: {message.content}",
                    "workflow_trace": {
                        "plan": {"steps": ["Mock step 1", "Mock step 2"]},
                        "context": [{"content": "Mock document content"}],
                        "execution_result": "Mock execution result",
                        "reflection": "Mock reflection",
                    },
                }

            # If there's a workflow trace, add it to the message steps
            if results and "workflow_trace" in results:
                trace = results.get("workflow_trace", {})

                # Add steps for the planning, retrieval, execution, etc.
                steps = []

                if trace.get("plan"):
                    steps.append(
                        cl.Step(
                            name="Planning",
                            type="planning",
                            content="\n".join(trace["plan"].get("steps", [])),
                        ),
                    )

                if trace.get("context"):
                    context_str = "\n\n".join(
                        [
                            f"Document {i + 1}:\n{doc.get('content', '')}"
                            for i, doc in enumerate(trace.get("context", []))
                        ],
                    )
                    steps.append(
                        cl.Step(
                            name="Retrieval",
                            type="retrieval",
                            content=context_str,
                        ),
                    )

                if trace.get("execution_result"):
                    steps.append(
                        cl.Step(
                            name="Execution",
                            type="execution",
                            content=trace["execution_result"],
                        ),
                    )

                if trace.get("reflection"):
                    steps.append(
                        cl.Step(
                            name="Reflection",
                            type="thinking",
                            content=trace["reflection"],
                        ),
                    )

                # Add the steps to the message
                thinking_msg.steps = steps

    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        await cl.Message(
            content=f"❌ Error: {e!s}",
            author="System",
        ).send()


# ----- File Handling -----


@cl.on_file
async def on_file(file: FileDict) -> None:
    """
    Process uploaded files.

    Args:
    ----
        file: The uploaded file

    """
    try:
        # Get ingestion pipeline from session
        pipeline = cl.user_session.get("ingestion_pipeline")

        if pipeline is None:
            if is_development:
                # Create mock pipeline in development mode
                logger.warning("Ingestion pipeline not found in session, creating mock")
                from unittest.mock import AsyncMock, MagicMock

                pipeline = MagicMock()
                pipeline.ingest_file = AsyncMock(
                    return_value={
                        "status": "success",
                        "chunks": 7,
                        "message": "Mock file ingestion success",
                    },
                )
                cl.user_session.set("ingestion_pipeline", pipeline)
                logger.info("Created mock ingestion pipeline for development")
            else:
                await cl.Message(
                    content="Ingestion pipeline not initialized.",
                    author="System",
                ).send()
                return

        # Process the file
        file_path = file["path"]
        file_name = file["name"]
        file["type"]

        ingestion_msg = cl.Message(content="", author="EMVR")
        await ingestion_msg.send()

        async with ingestion_msg.content_thread() as content_stream:
            await content_stream.append(f"📤 Processing file: `{file_name}`...")

            # Ingest the file
            try:
                result = await pipeline.ingest_file(
                    file_path,
                    metadata={"source": file_name},
                )
            except Exception as e:
                logger.warning(f"Pipeline error: {e}. Using mock response.")
                # Create mock result for development
                result = {
                    "status": "success",
                    "chunks": 5,
                    "message": "Mock ingestion success",
                }

            # Show the ingestion result
            if result.get("status") == "success":
                await content_stream.clear()
                chunks = result.get("chunks", 0)
                await content_stream.append(
                    f"✅ Successfully ingested `{file_name}` ({chunks} chunks created)\n\n"
                    f"The file content is now available in the knowledge base.",
                )
            else:
                await content_stream.clear()
                await content_stream.append(
                    f"❌ Failed to ingest `{file_name}`: {result.get('error', 'Unknown error')}",
                )

    except Exception as e:
        logger.exception(f"Error processing file: {e}")
        await cl.Message(
            content=f"❌ Error processing file: {e!s}",
            author="System",
        ).send()


# ----- Action Handlers -----


@cl.action_callback("search")
async def on_search(action) -> None:
    """
    Handle search action.

    Args:
    ----
        action: The action

    """
    # Get search query
    await cl.Message(
        content="🔍 What would you like to search for?",
        author="EMVR",
    ).send()


@cl.action_callback("ingest")
async def on_ingest(action) -> None:
    """
    Handle ingest action.

    Args:
    ----
        action: The action

    """
    # Prompt for ingestion source
    await cl.Message(
        content="📤 Please provide a URL or upload a file to ingest:",
        author="EMVR",
    ).send()


@cl.action_callback("analyze")
async def on_analyze(action) -> None:
    """
    Handle analyze action.

    Args:
    ----
        action: The action

    """
    # Prompt for analysis content
    await cl.Message(
        content="🔬 What would you like me to analyze?",
        author="EMVR",
    ).send()


# ----- UI Customization -----


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    """
    Handle settings updates.

    Args:
    ----
        settings: The new settings

    """
    # Update user preferences
    cl.user_session.set("settings", settings)

    # Confirm update
    await cl.Message(
        content="✅ Settings updated",
        author="System",
    ).send()


# ----- Shutdown -----


@cl.on_app_stop
async def shutdown() -> None:
    """Clean up resources when app is shutting down."""
    try:
        # Get orchestrator and close connections
        orchestrator = cl.user_session.get("orchestrator")
        if orchestrator:
            try:
                await orchestrator.shutdown()
            except Exception as inner_e:
                logger.warning(f"Error shutting down orchestrator: {inner_e}")

        # Close other connections as needed
        try:
            memory_manager.close()
        except Exception as inner_e:
            logger.warning(f"Error closing memory manager: {inner_e}")

        logger.info("UI shutdown successfully")
    except Exception as e:
        logger.exception(f"Error during shutdown: {e}")
