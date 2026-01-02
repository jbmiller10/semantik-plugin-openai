"""OpenAI embeddings plugin for Semantik.

This plugin provides embedding generation using OpenAI's text-embedding models:
- text-embedding-3-small (1536 dimensions)
- text-embedding-3-large (3072 dimensions)
- text-embedding-ada-002 (1536 dimensions)

Usage:
    pip install git+https://github.com/jbmiller10/semantik-plugin-openai.git

    # Set your API key
    export OPENAI_API_KEY=sk-...

    # The plugin auto-registers via entry points
"""

from .provider import OpenAIEmbeddingPlugin

__all__ = ["OpenAIEmbeddingPlugin"]
__version__ = "1.0.0"
