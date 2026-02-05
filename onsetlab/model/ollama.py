"""Ollama model backend."""

import json
import subprocess
from typing import List, Dict, Any, Optional

from .base import BaseModel


class OllamaModel(BaseModel):
    """
    Ollama model backend for local SLM inference.
    
    Requires Ollama to be installed and running.
    """
    
    def __init__(self, model: str = "phi3.5"):
        """
        Initialize Ollama model.
        
        Args:
            model: Model name (e.g., "phi3.5", "qwen2.5:3b", "llama3.2:3b").
        """
        self._model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama not responding")
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not found. Install from https://ollama.com"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama not responding (timeout)")
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text using Ollama."""
        # Build the full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt
        
        # Call Ollama
        return self._call_ollama(
            full_prompt,
            stop_sequences=stop_sequences,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Chat completion using Ollama."""
        # Build messages array for Ollama
        ollama_messages = []
        
        if system_prompt:
            ollama_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return self._call_ollama_chat(
            ollama_messages,
            stop_sequences=stop_sequences,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def _call_ollama(
        self,
        prompt: str,
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Call Ollama generate API."""
        import requests
        
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running? Try: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")
    
    def _call_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Call Ollama chat API."""
        import requests
        
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running? Try: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")
