# Semantik OpenAI Embeddings Plugin

Generate high-quality text embeddings using OpenAI's embedding models.

## Supported Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| `text-embedding-3-small` | 1536 | Best cost/performance ratio (default) |
| `text-embedding-3-large` | 3072 | Highest quality embeddings |
| `text-embedding-ada-002` | 1536 | Legacy model |

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/jbmiller10/semantik-plugin-openai.git

# Or pin to a specific version
pip install git+https://github.com/jbmiller10/semantik-plugin-openai.git@v1.0.0
```

## Configuration

### Environment Variable (Recommended)

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

### Docker Compose

Add to your `docker-compose.yml`:

```yaml
services:
  webui:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

### Plugin Configuration (Semantik UI)

Configure via the Semantik plugin settings UI:

```json
{
  "api_key_env": "OPENAI_API_KEY",
  "organization": "org-xxx"
}
```

## Usage

Once installed, the plugin automatically registers with Semantik. You can then:

1. **Create a collection** with OpenAI embeddings in the Semantik UI
2. **Select the model** `openai/text-embedding-3-small` or `openai/text-embedding-3-large`
3. **Index your documents** - embeddings will be generated via the OpenAI API

### Model Selection

When creating a collection or configuring embeddings, use one of:

- `text-embedding-3-small` - Recommended for most use cases
- `text-embedding-3-large` - When you need maximum quality
- `openai/text-embedding-3-small` - Explicit provider prefix

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/jbmiller10/semantik-plugin-openai.git
cd semantik-plugin-openai

# Install in development mode
pip install -e ".[dev]"
```

### Run Tests

```bash
# Set a test API key (tests use mocks, no real API calls)
export OPENAI_API_KEY=test-key

# Run tests
pytest
```

### Local Testing with Semantik

```bash
# Install in your Semantik container
docker compose exec webui pip install -e /path/to/semantik-plugin-openai

# Or install from local wheel
pip install ./dist/semantik_plugin_openai-1.0.0-py3-none-any.whl
```

## API Reference

### OpenAIEmbeddingPlugin

```python
from semantik_plugin_openai import OpenAIEmbeddingPlugin

# Check if model is supported
OpenAIEmbeddingPlugin.supports_model("text-embedding-3-small")  # True

# Get plugin definition
definition = OpenAIEmbeddingPlugin.get_definition()

# Use the plugin
plugin = OpenAIEmbeddingPlugin()
await plugin.initialize("text-embedding-3-small")
embeddings = await plugin.embed_texts(["hello", "world"])
await plugin.cleanup()
```

## Troubleshooting

### "API key not found"

Ensure `OPENAI_API_KEY` is set in your environment or configured in plugin settings.

### "Unsupported model"

Check that you're using one of the supported model names listed above.

### Network Errors

Ensure your Semantik container has network access to `api.openai.com`.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Semantik](https://github.com/jbmiller10/semantik) - Self-hosted semantic search engine
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) - Official documentation
- [Plugin Template](https://github.com/jbmiller10/semantik-plugin-template) - Create your own plugin
