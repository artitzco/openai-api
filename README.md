# OpenAI API Python Explorer (openIA)

A minimalist yet powerful Python interface for the OpenAI API, designed for incremental development and advanced multimodal interactions.

## Features

- **Variadic Chat Interface**: Send multiple message components directly: `chat.chat("Text", Image("local.png"), "More text")`.
- **Node-Based History**: Every interaction is a node with an incremental ID.
- **Context Control**: Activate or deactivate conversation nodes by their ID to manage the model's window.
- **Enhanced Metrics**: Track token usage, models, and active nodes per request with built-in Pandas integration.
- **Multimodal Support**: Handle local images (auto base64 encoding) and remote URLs seamlessly.
- **State Persistence**: Save and load complete Chat sessions (history + metrics) to JSON files.
- **Deep Copy**: Create independent clones of your chat session.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd openIA

# Install dependencies
pip install openai pandas pillow
```

## Quick Start

```python
from openaiapi import Chat, Image

# Initialize (uses OPENAI_API_KEY env var by default)
chat = Chat(model="gpt-4o-mini", system_prompt="You are a helpful assistant.")

# Basic Chat
response = chat.chat("Hello! What can you do?")
print(response)

# Multimodal Chat (Local and URL)
chat.chat(
    "Analyze these images:",
    Image("path/to/local_image.png"),
    "and also this one:",
    Image("https://example.com/image.jpg")
)

# Manage History
df_history = chat.history.to_dataframe()
chat.history.toggle(node_id=1, active=False) # Context management

# Persistence
chat.save("my_session.json")
new_chat = Chat.load("my_session.json")
```

## Project Structure

- `src/openaiapi/`: Core library modules.
  - `client.py`: The main `Chat` class.
  - `history.py`: Logic for node-based conversation management.
  - `metrics.py`: Token usage and request tracking.
  - `content.py`: Multimodal data builders (Images, etc.).
- `notebooks/`: Interactive guides and testing playgrounds.
- `img/`: Storage for sample assets.
- `output/`: Default location for saved session JSONs.

## License

MIT
