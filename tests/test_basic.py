"""Basic tests for the enhanced-mem-vector-rag package."""


# Simple test to verify pytest is working
def test_package_imports():
    """Test that the package can be imported."""
    import emvr

    assert emvr is not None


# Simple test to verify environment
def test_environment():
    """Test the environment setup."""
    import os
    import sys

    # Verify Python version
    assert sys.version_info >= (3, 11), "Python >= 3.11 is required"

    # Verify working directory
    cwd = os.getcwd()
    assert "enhanced-mem-vector-rag" in cwd, "Working in the project directory"

    # Verify pytest is running
    assert "pytest" in sys.modules, "pytest is available"
