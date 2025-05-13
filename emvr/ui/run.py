"""
Chainlit application runner for EMVR.

This module provides a function to run the Chainlit UI application.
"""

import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_chainlit(
    host: str = "localhost",
    port: int = 8000,
    debug: bool = False,
) -> None:
    """
    Run the Chainlit UI application.

    Args:
    ----
        host: Host to bind to
        port: Port to bind to
        debug: Whether to run in debug mode

    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent.resolve()

        # Set environment variables
        env = os.environ.copy()
        env["CHAINLIT_HOST"] = host
        env["CHAINLIT_PORT"] = str(port)

        # Set development mode for testing
        env["EMVR_DEVELOPMENT"] = "true"

        if debug:
            env["CHAINLIT_DEBUG"] = "true"
            # Enable more verbose logging in debug mode
            env["EMVR_LOG_LEVEL"] = "DEBUG"
        else:
            env["EMVR_LOG_LEVEL"] = "INFO"

        # Determine app path
        app_path = os.path.join(
            project_root,
            "emvr",
            "ui",
            "app.py",
        )

        # Verify app path exists
        if not os.path.exists(app_path):
            logger.error(f"Chainlit app not found at {app_path}")
            raise FileNotFoundError(f"App file not found: {app_path}")

        # Verify chainlit is installed
        try:
            import chainlit

            logger.info(f"Using Chainlit version: {chainlit.__version__}")
        except ImportError:
            logger.error("Chainlit not installed. Run: pip install chainlit==2.5.5")
            raise ImportError("Chainlit not installed")

        # Build command
        cmd = [
            "chainlit",
            "run",
            app_path,
            "--host",
            host,
            "--port",
            str(port),
        ]

        if debug:
            cmd.append("--debug")

        # Run Chainlit
        logger.info("Starting Chainlit UI on %s:%s", host, port)
        logger.info("Development mode enabled - using mock components where needed")

        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(project_root),
            check=False,
            capture_output=debug,  # Capture output in debug mode
        )

        if result.returncode != 0:
            logger.error(f"Chainlit exited with code {result.returncode}")
            if debug and result.stderr:
                logger.error(f"Error output: {result.stderr.decode('utf-8')}")

            raise RuntimeError(f"Chainlit failed with exit code {result.returncode}")

    except Exception as e:
        logger.exception(f"Error running Chainlit: {e}")
        raise


if __name__ == "__main__":
    # When run directly, start the UI
    import argparse

    parser = argparse.ArgumentParser(description="Run EMVR Chainlit UI")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    run_chainlit(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
