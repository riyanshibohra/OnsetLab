# Model backends (Ollama, GGUF, Cloud)
from .base import BaseModel
from .ollama import OllamaModel

__all__ = ["BaseModel", "OllamaModel"]
