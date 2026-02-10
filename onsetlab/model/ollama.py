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
    
    # Recommended models for tool-calling (tested)
    RECOMMENDED = {
        "phi3.5": "Microsoft Phi-3.5 (3.8B) - Default, good balance of speed/quality",
        "qwen2.5:3b": "Alibaba Qwen 2.5 (3B) - Fast, good instruction following",
        "qwen2.5:7b": "Alibaba Qwen 2.5 (7B) - Strong tool calling, needs 8GB+ RAM",
        "qwen3-a3b": "Alibaba Qwen 3 (MoE, 3B active/30B total) - Best tool calling, needs 16GB+ RAM",
        "llama3.2:3b": "Meta Llama 3.2 (3B) - Good general purpose",
    }
    
    @classmethod
    def list_available(cls) -> list:
        """List models available in your local Ollama."""
        import subprocess
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines if line]
        except:
            pass
        return []
    
    @classmethod  
    def print_recommended(cls):
        """Print recommended models for tool-calling."""
        print("\nRecommended models for OnsetLab:")
        print("-" * 50)
        for name, desc in cls.RECOMMENDED.items():
            available = cls.list_available()
            status = "✓ installed" if name in available or name.split(":")[0] in [m.split(":")[0] for m in available] else "  (run: ollama pull " + name + ")"
            print(f"  {name}: {desc}")
            print(f"    {status}")
    
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
        json_mode: bool = False,
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
        
        if json_mode:
            payload["format"] = "json"
        
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
