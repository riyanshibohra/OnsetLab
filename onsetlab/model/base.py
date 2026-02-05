"""Base class for model backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseModel(ABC):
    """Base class for SLM backends (Ollama, GGUF, Cloud)."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            stop_sequences: Sequences that stop generation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}.
            system_prompt: Optional system prompt.
            stop_sequences: Sequences that stop generation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Assistant response.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
