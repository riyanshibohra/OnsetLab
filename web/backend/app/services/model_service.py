"""OpenRouter model service - FREE small models!"""

import os
import requests
from typing import Optional, List

from ..config import settings


class OpenRouterModel:
    """
    OpenRouter model wrapper - FREE access to small models.
    
    OpenAI-compatible API with 300+ models.
    Get your free API key at: https://openrouter.ai
    """
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model_name = model
        self.openrouter_model = self._get_model_id(model)
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Get your FREE key at https://openrouter.ai"
            )
    
    def _get_model_id(self, model: str) -> str:
        """Map our model names to OpenRouter model IDs."""
        model_config = settings.MODELS.get(model)
        if model_config:
            return model_config["openrouter_id"]
        
        # Default to Llama 3.2 3B free
        return settings.MODELS["llama3.2:3b"]["openrouter_id"]
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text completion using chat endpoint.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens, stop_sequences)
    
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Chat completion via OpenRouter.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://onsetlab.dev",  # Optional, for rankings
            "X-Title": "OnsetLab Playground",
        }
        
        payload = {
            "model": self.openrouter_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # OpenRouter supports up to 4 stop sequences
        if stop_sequences:
            payload["stop"] = stop_sequences[:4]
        
        # Retry once on timeout (MCP tool calls can chain multiple LLM calls)
        last_error = None
        for attempt in range(2):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
                
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt == 0:
                    continue  # retry once
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"OpenRouter API error: {e}")
        
        raise RuntimeError(f"OpenRouter timed out after 2 attempts. Try again.")


# Aliases for compatibility
GroqModel = OpenRouterModel
TogetherModel = OpenRouterModel


def get_model(model_name: str = "llama3.2:3b") -> OpenRouterModel:
    """Factory function to get a model instance."""
    return OpenRouterModel(model_name)
