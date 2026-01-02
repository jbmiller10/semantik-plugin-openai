"""Tests for OpenAI embedding plugin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from semantik_plugin_openai import OpenAIEmbeddingPlugin


class TestOpenAIEmbeddingPlugin:
    """Test suite for OpenAIEmbeddingPlugin."""

    def test_class_attributes(self) -> None:
        """Test that required class attributes are defined."""
        assert OpenAIEmbeddingPlugin.INTERNAL_NAME == "openai_embeddings"
        assert OpenAIEmbeddingPlugin.API_ID == "openai-embeddings"
        assert OpenAIEmbeddingPlugin.PROVIDER_TYPE == "remote"
        assert OpenAIEmbeddingPlugin.PLUGIN_VERSION == "1.0.0"

    def test_supported_models(self) -> None:
        """Test supported models configuration."""
        expected_models = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        assert OpenAIEmbeddingPlugin.SUPPORTED_MODELS == expected_models

    def test_supports_model_direct(self) -> None:
        """Test model support detection with direct names."""
        assert OpenAIEmbeddingPlugin.supports_model("text-embedding-3-small")
        assert OpenAIEmbeddingPlugin.supports_model("text-embedding-3-large")
        assert OpenAIEmbeddingPlugin.supports_model("text-embedding-ada-002")
        assert not OpenAIEmbeddingPlugin.supports_model("unknown-model")

    def test_supports_model_with_prefix(self) -> None:
        """Test model support detection with openai/ prefix."""
        assert OpenAIEmbeddingPlugin.supports_model("openai/text-embedding-3-small")
        assert OpenAIEmbeddingPlugin.supports_model("openai/text-embedding-3-large")
        assert not OpenAIEmbeddingPlugin.supports_model("openai/unknown-model")

    def test_get_definition(self) -> None:
        """Test get_definition returns correct metadata."""
        definition = OpenAIEmbeddingPlugin.get_definition()

        assert definition.api_id == "openai-embeddings"
        assert definition.internal_id == "openai_embeddings"
        assert definition.provider_type == "remote"
        assert definition.is_plugin is True
        assert "text-embedding-3-small" in definition.supported_models

    def test_not_initialized_before_init(self) -> None:
        """Test that plugin is not initialized before initialize()."""
        plugin = OpenAIEmbeddingPlugin()
        assert not plugin.is_initialized


class TestOpenAIEmbeddingPluginWithMocks:
    """Test suite with mocked OpenAI API."""

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create a mock OpenAI embeddings response."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        return mock_response

    @pytest.fixture
    def mock_batch_response(self) -> MagicMock:
        """Create a mock OpenAI embeddings response for batch."""
        mock_embeddings = []
        for i in range(3):
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1 * (i + 1)] * 1536
            mock_embeddings.append(mock_embedding)

        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        return mock_response

    @pytest.mark.asyncio
    async def test_initialize_success(self) -> None:
        """Test successful initialization."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI"):
                await plugin.initialize("text-embedding-3-small")

        assert plugin.is_initialized
        assert plugin._model_name == "text-embedding-3-small"
        assert plugin._dimension == 1536

    @pytest.mark.asyncio
    async def test_initialize_with_prefix(self) -> None:
        """Test initialization with openai/ prefix."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI"):
                await plugin.initialize("openai/text-embedding-3-large")

        assert plugin._model_name == "text-embedding-3-large"
        assert plugin._dimension == 3072

    @pytest.mark.asyncio
    async def test_initialize_unsupported_model(self) -> None:
        """Test initialization with unsupported model fails."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="Unsupported model"):
                await plugin.initialize("unknown-model")

    @pytest.mark.asyncio
    async def test_initialize_missing_api_key(self) -> None:
        """Test initialization without API key fails."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            import os

            os.environ.pop("OPENAI_API_KEY", None)

            with pytest.raises(ValueError, match="API key not found"):
                await plugin.initialize("text-embedding-3-small")

    @pytest.mark.asyncio
    async def test_embed_texts_success(
        self, mock_batch_response: MagicMock
    ) -> None:
        """Test embed_texts with mocked API."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI") as mock_client_class:
                mock_client = MagicMock()
                mock_client.embeddings.create = AsyncMock(
                    return_value=mock_batch_response
                )
                mock_client_class.return_value = mock_client

                await plugin.initialize("text-embedding-3-small")
                result = await plugin.embed_texts(["hello", "world", "test"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1536)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_single_success(
        self, mock_openai_response: MagicMock
    ) -> None:
        """Test embed_single with mocked API."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI") as mock_client_class:
                mock_client = MagicMock()
                mock_client.embeddings.create = AsyncMock(
                    return_value=mock_openai_response
                )
                mock_client_class.return_value = mock_client

                await plugin.initialize("text-embedding-3-small")
                result = await plugin.embed_single("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self) -> None:
        """Test embed_texts with empty list returns empty array."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI"):
                await plugin.initialize("text-embedding-3-small")
                result = await plugin.embed_texts([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 1536)

    @pytest.mark.asyncio
    async def test_embed_texts_not_initialized(self) -> None:
        """Test embed_texts before initialization fails."""
        plugin = OpenAIEmbeddingPlugin()

        with pytest.raises(RuntimeError, match="not initialized"):
            await plugin.embed_texts(["hello"])

    @pytest.mark.asyncio
    async def test_get_dimension(self) -> None:
        """Test get_dimension returns correct value."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI"):
                await plugin.initialize("text-embedding-3-large")

        assert plugin.get_dimension() == 3072

    @pytest.mark.asyncio
    async def test_get_model_info(self) -> None:
        """Test get_model_info returns expected structure."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI"):
                await plugin.initialize("text-embedding-3-small")

        info = plugin.get_model_info()

        assert info["model_name"] == "text-embedding-3-small"
        assert info["dimension"] == 1536
        assert info["device"] == "api"
        assert info["provider"] == "openai_embeddings"
        assert "max_sequence_length" in info

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup resets state."""
        plugin = OpenAIEmbeddingPlugin()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("semantik_plugin_openai.provider.AsyncOpenAI"):
                await plugin.initialize("text-embedding-3-small")
                assert plugin.is_initialized

                await plugin.cleanup()
                assert not plugin.is_initialized

    @pytest.mark.asyncio
    async def test_health_check_with_env_key(self) -> None:
        """Test health_check when API key is in environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = await OpenAIEmbeddingPlugin.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_without_key(self) -> None:
        """Test health_check when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("OPENAI_API_KEY", None)

            result = await OpenAIEmbeddingPlugin.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_config_key(self) -> None:
        """Test health_check with API key in config."""
        result = await OpenAIEmbeddingPlugin.health_check({"api_key": "test-key"})
        assert result is True


class TestOpenAIEmbeddingPluginConfig:
    """Test configuration handling."""

    def test_config_from_dict(self) -> None:
        """Test configuration from dictionary."""
        config = {
            "api_key": "test-key",
            "organization": "org-123",
            "model": "text-embedding-3-large",
        }
        plugin = OpenAIEmbeddingPlugin(config=config)

        assert plugin._plugin_config == config

    def test_config_from_kwargs(self) -> None:
        """Test configuration from kwargs when config is not a dict."""
        plugin = OpenAIEmbeddingPlugin(config=None, api_key="test", model="test-model")

        assert plugin._plugin_config.get("api_key") == "test"
        assert plugin._plugin_config.get("model") == "test-model"
