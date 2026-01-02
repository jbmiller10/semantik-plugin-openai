"""OpenAI embedding plugin implementation.

This plugin generates embeddings using OpenAI's text-embedding API.
Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "openai package is required. Install with: pip install openai>=1.0.0"
    ) from e

# These imports work when Semantik is installed
try:
    from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
except ImportError:
    # Fallback for standalone development/testing
    from abc import ABC, abstractmethod

    class BaseEmbeddingPlugin(ABC):  # type: ignore[no-redef]
        """Minimal stub for development without Semantik installed."""

        INTERNAL_NAME: ClassVar[str] = ""
        API_ID: ClassVar[str] = ""
        PROVIDER_TYPE: ClassVar[str] = "local"
        PLUGIN_VERSION: ClassVar[str] = "0.0.0"
        METADATA: ClassVar[dict[str, Any]] = {}

        def __init__(self, config: Any = None) -> None:
            self.config = config

        @classmethod
        @abstractmethod
        def get_definition(cls) -> Any:
            ...

        @classmethod
        @abstractmethod
        def supports_model(cls, model_name: str) -> bool:
            ...

    class EmbeddingProviderDefinition:  # type: ignore[no-redef]
        """Minimal stub for development."""

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)


if TYPE_CHECKING:
    from shared.embedding.types import EmbeddingMode

logger = logging.getLogger(__name__)


class OpenAIEmbeddingPlugin(BaseEmbeddingPlugin):
    """OpenAI embedding provider plugin.

    Generates embeddings using OpenAI's text-embedding API.

    Supported models:
    - text-embedding-3-small: 1536 dimensions, best cost/performance
    - text-embedding-3-large: 3072 dimensions, highest quality
    - text-embedding-ada-002: 1536 dimensions, legacy model

    Configuration:
        The plugin reads configuration from the Semantik plugin state or
        environment variables:

        - api_key: OpenAI API key (required)
            - Via config: {"api_key_env": "OPENAI_API_KEY"}
            - Via env: OPENAI_API_KEY
        - organization: OpenAI organization ID (optional)
            - Via config: {"organization": "org-xxx"}
            - Via env: OPENAI_ORG_ID
        - model: Default model name (optional, default: text-embedding-3-small)
            - Via config: {"model": "text-embedding-3-large"}
    """

    # Required class attributes
    INTERNAL_NAME: ClassVar[str] = "openai_embeddings"
    API_ID: ClassVar[str] = "openai-embeddings"
    PROVIDER_TYPE: ClassVar[str] = "remote"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "OpenAI Embeddings",
        "description": "High-quality embeddings via OpenAI API",
        "author": "semantik",
        "homepage": "https://github.com/jbmiller10/semantik-plugin-openai",
        "best_for": ["general_purpose", "english", "multilingual"],
        "pros": [
            "High-quality embeddings",
            "No local GPU required",
            "Simple API integration",
            "Excellent for general text",
        ],
        "cons": [
            "Requires API key",
            "Per-token pricing",
            "Network latency",
            "Data leaves your infrastructure",
        ],
    }

    # Model name to dimension mapping
    SUPPORTED_MODELS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Model-specific settings
    MODEL_INFO: ClassVar[dict[str, dict[str, Any]]] = {
        "text-embedding-3-small": {
            "max_sequence_length": 8191,
            "description": "Most cost-effective embedding model",
        },
        "text-embedding-3-large": {
            "max_sequence_length": 8191,
            "description": "Highest quality embedding model",
        },
        "text-embedding-ada-002": {
            "max_sequence_length": 8191,
            "description": "Legacy embedding model",
        },
    }

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        """Initialize the OpenAI embedding plugin.

        Args:
            config: Plugin configuration dictionary or VecpipeConfig
            **kwargs: Additional options
        """
        super().__init__(config)
        self._model_name: str | None = None
        self._dimension: int | None = None
        self._initialized: bool = False
        self._client: AsyncOpenAI | None = None

        # Extract config if it's a dict, otherwise use kwargs
        if isinstance(config, dict):
            self._plugin_config = config
        else:
            self._plugin_config = kwargs

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        """Return the canonical definition for this plugin."""
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name=cls.METADATA.get("display_name", "OpenAI Embeddings"),
            description=cls.METADATA.get("description", "OpenAI text embeddings"),
            provider_type=cls.PROVIDER_TYPE,
            supports_quantization=False,  # API returns float
            supports_instruction=False,  # No instruction prefix
            supports_batch_processing=True,
            supports_asymmetric=False,  # OpenAI doesn't use query/doc prefixes
            supported_models=tuple(cls.SUPPORTED_MODELS.keys()),
            default_config={
                "model": "text-embedding-3-small",
            },
            performance_characteristics={
                "latency": "medium",  # Network dependent
                "throughput": "high",  # Batch API is fast
                "memory_usage": "very_low",  # No local model
            },
            is_plugin=True,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this plugin supports the given model name.

        Matches exact model names like 'text-embedding-3-small' or
        prefixed names like 'openai/text-embedding-3-small'.
        """
        # Direct match
        if model_name in cls.SUPPORTED_MODELS:
            return True

        # Match with openai/ prefix
        if model_name.startswith("openai/"):
            actual_name = model_name[7:]  # Remove 'openai/' prefix
            return actual_name in cls.SUPPORTED_MODELS

        return False

    def _get_api_key(self) -> str:
        """Get the OpenAI API key from config or environment."""
        # Check plugin config first
        if self._plugin_config:
            # Direct API key (not recommended, but supported)
            if "api_key" in self._plugin_config:
                return self._plugin_config["api_key"]

            # Reference to environment variable (recommended)
            if "api_key_env" in self._plugin_config:
                env_var = self._plugin_config["api_key_env"]
                key = os.environ.get(env_var)
                if key:
                    return key

        # Fall back to standard environment variable
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key

        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or configure api_key_env in plugin settings."
        )

    def _get_organization(self) -> str | None:
        """Get the OpenAI organization ID from config or environment."""
        if self._plugin_config:
            if "organization" in self._plugin_config:
                return self._plugin_config["organization"]

        return os.environ.get("OPENAI_ORG_ID")

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name, removing any prefix."""
        if model_name.startswith("openai/"):
            return model_name[7:]
        return model_name

    @property
    def is_initialized(self) -> bool:
        """Check if the plugin is initialized."""
        return self._initialized

    async def initialize(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the OpenAI embedding client.

        Args:
            model_name: The model to use for embeddings
            **kwargs: Additional configuration options
        """
        normalized_name = self._normalize_model_name(model_name)

        if normalized_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self._model_name = normalized_name
        self._dimension = self.SUPPORTED_MODELS[normalized_name]

        # Create OpenAI client
        api_key = self._get_api_key()
        organization = self._get_organization()

        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
        )

        self._initialized = True
        logger.info(
            f"OpenAI embedding plugin initialized with model={self._model_name}, "
            f"dimension={self._dimension}"
        )

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 2048,  # OpenAI supports up to 2048 in one request
        *,
        mode: "EmbeddingMode | None" = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (max 2048)
            mode: Ignored - OpenAI doesn't use asymmetric embeddings
            **kwargs: Additional options (ignored)

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "OpenAI plugin not initialized. Call initialize() first."
            )

        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension or 1536)

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = await self._client.embeddings.create(
                    model=self._model_name,
                    input=batch,
                )

                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise RuntimeError(f"Failed to generate embeddings: {e}") from e

        logger.debug(
            f"Generated {len(all_embeddings)} embeddings with dimension {self._dimension}"
        )

        return np.array(all_embeddings, dtype=np.float32)

    async def embed_single(
        self,
        text: str,
        *,
        mode: "EmbeddingMode | None" = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            mode: Ignored - OpenAI doesn't use asymmetric embeddings
            **kwargs: Additional options

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        embeddings = await self.embed_texts([text], mode=mode, **kwargs)
        return embeddings[0]

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        if not self._initialized:
            raise RuntimeError("OpenAI plugin not initialized")
        return self._dimension or 1536

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        if not self._initialized:
            raise RuntimeError("OpenAI plugin not initialized")

        model_info = self.MODEL_INFO.get(self._model_name or "", {})

        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "device": "api",
            "max_sequence_length": model_info.get("max_sequence_length", 8191),
            "quantization": "float32",
            "provider": self.INTERNAL_NAME,
            "description": model_info.get("description", "OpenAI embedding model"),
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._model_name = None
        self._dimension = None

        if self._client is not None:
            # AsyncOpenAI client doesn't require explicit cleanup
            self._client = None

        logger.info("OpenAI embedding plugin cleaned up")

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
        """Check if the plugin can connect to OpenAI.

        Args:
            config: Optional configuration with API key

        Returns:
            True if API key is available, False otherwise
        """
        try:
            # Check if API key is available
            if config and ("api_key" in config or "api_key_env" in config):
                if "api_key" in config:
                    return bool(config["api_key"])
                env_var = config.get("api_key_env", "OPENAI_API_KEY")
                return bool(os.environ.get(env_var))

            # Fall back to environment variable
            return bool(os.environ.get("OPENAI_API_KEY"))
        except Exception:
            return False
