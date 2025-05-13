"""Basic tests for the enhanced-mem-vector-rag package."""


# Simple test to verify pytest is working
def test_package_imports():
    """Test that the package can be imported."""
    import emvr

    assert emvr is not None


# Test configuration settings
def test_config_settings():
    """Test configuration settings."""
    from emvr.config import get_settings

    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, "mcp_host")
    assert hasattr(settings, "mcp_port")


# Test memory_manager exists
def test_memory_manager():
    """Test memory_manager is available."""
    from emvr.memory.memory_manager import memory_manager

    assert memory_manager is not None


# Test ingestion_pipeline exists
def test_ingestion_pipeline():
    """Test ingestion_pipeline is available."""
    from emvr.ingestion.pipeline import ingestion_pipeline

    assert ingestion_pipeline is not None


# Test retrieval_pipeline exists
def test_retrieval_pipeline():
    """Test retrieval_pipeline is available."""
    from emvr.retrieval.pipeline import retrieval_pipeline

    assert retrieval_pipeline is not None
