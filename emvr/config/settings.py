"""Settings management for the EMVR system."""

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Settings for the EMVR system.

    This class uses Pydantic to load and validate settings from environment variables.
    """

    # App settings
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_tracing: bool = False
    max_concurrent_requests: int = 5

    # LLM settings
    default_llm_provider: Literal["openai", "anthropic", "cohere"] = "openai"
    default_llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # API keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    cohere_api_key: str | None = None

    # Embedding settings
    embedding_provider: Literal["openai", "cohere", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_device: Literal["cpu", "cuda"] = "cpu"
    vector_dimension: int = 384

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "emvr_vectors"
    qdrant_location: Literal["local", "http"] = "local"

    # Mem0 settings
    mem0_url: str = "http://localhost:7891"
    mem0_api_key: str | None = None
    mem0_memory_id: str = "emvr_memory"

    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "emvr"

    # Graphiti settings
    graphiti_uri: str = "http://localhost:2342"
    graphiti_api_key: str | None = None

    # Supabase settings
    supabase_url: str = "http://localhost:8000"
    supabase_key: str | None = None
    supabase_project: str = "emvr"

    # AWS S3 settings
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-west-2"
    s3_bucket_name: str = "emvr-documents"

    # MCP server settings
    mcp_port: int = Field(default=8080, gt=0, lt=65536)
    mcp_host: str = "0.0.0.0"
    mcp_api_key: str | None = None
    mcp_timeout: int = 30

    # Chunking settings
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    # Chainlit UI settings
    chainlit_host: str = "0.0.0.0"
    chainlit_port: int = Field(default=8501, gt=0, lt=65536)
    chainlit_auth_enabled: bool = False

    # Docker settings
    docker_network_name: str = "emvr-network"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    @field_validator("openai_api_key")
    def validate_openai_api_key(cls, v: str | None, info) -> str | None:
        """Validate OpenAI API key if OpenAI is used."""
        values = info.data
        if values.get("default_llm_provider") == "openai" and not v:
            if os.environ.get("APP_ENV") != "test":
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)
        return v

    @classmethod
    @field_validator("anthropic_api_key")
    def validate_anthropic_api_key(cls, v: str | None, info) -> str | None:
        """Validate Anthropic API key if Anthropic is used."""
        values = info.data
        if values.get("default_llm_provider") == "anthropic" and not v:
            if os.environ.get("APP_ENV") != "test":
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
        return v

    @classmethod
    @field_validator("cohere_api_key")
    def validate_cohere_api_key(cls, v: str | None, info) -> str | None:
        """Validate Cohere API key if Cohere is used."""
        values = info.data
        if values.get("default_llm_provider") == "cohere" and not v:
            if os.environ.get("APP_ENV") != "test":
                msg = "Cohere API key is required when using Cohere provider"
                raise ValueError(msg)
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get settings from environment variables with caching.

    Returns
    -------
        Settings: Validated settings

    """
    return Settings()
