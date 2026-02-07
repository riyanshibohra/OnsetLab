"""Configuration settings."""

import os
from pathlib import Path
from typing import List

# Load .env file if exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


class Settings:
    """Application settings from environment variables."""
    
    # API Keys - OpenRouter has FREE models!
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:5173,http://localhost:3000"
    ).split(",")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "5"))
    SESSION_EXPIRY_HOURS: int = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
    
    # SLMs via OpenRouter — small, open-source, good at tool calling
    MODELS = {
        "qwen3-a3b": {
            "openrouter_id": "qwen/qwen3-next-80b-a3b-instruct",
            "display_name": "Qwen3 A3B",
            "description": "Best for tool calling — MoE, 3B active params",
            "params": "3B active",
            "badge": "recommended",
        },
        "qwen2.5:7b": {
            "openrouter_id": "qwen/qwen-2.5-7b-instruct",
            "display_name": "Qwen 2.5 7B",
            "description": "Strong all-rounder, best value for 7B",
            "params": "7B",
            "badge": "best value",
        },
        "hermes3:8b": {
            "openrouter_id": "nousresearch/hermes-3-llama-3.1-8b",
            "display_name": "Hermes 3 8B",
            "description": "Excellent function calling, structured output",
            "params": "8B",
            "badge": "function calling",
        },
        "mistral:7b": {
            "openrouter_id": "mistralai/mistral-7b-instruct",
            "display_name": "Mistral 7B",
            "description": "Reliable instruction following, fast",
            "params": "7B",
            "badge": "fastest",
        },
        "gemma3:4b": {
            "openrouter_id": "google/gemma-3-4b-it",
            "display_name": "Gemma 3 4B",
            "description": "Smallest model, good for simple tasks",
            "params": "4B",
            "badge": "",
        },
        "qwen2.5-coder:7b": {
            "openrouter_id": "qwen/qwen-2.5-coder-7b-instruct",
            "display_name": "Qwen 2.5 Coder 7B",
            "description": "Optimized for code + tool use",
            "params": "7B",
            "badge": "code",
        },
    }
    
    # Default model for new sessions
    DEFAULT_MODEL = "qwen3-a3b"
    
    # Same models run locally via Ollama
    LOCAL_MODELS_NOTE = "All run locally via Ollama too"
    
    # Built-in Tools (all 6)
    TOOLS = [
        {
            "name": "Calculator",
            "description": "Math expressions and calculations",
            "enabled_by_default": True,
            "category": "builtin",
        },
        {
            "name": "DateTime",
            "description": "Date/time operations and queries",
            "enabled_by_default": True,
            "category": "builtin",
        },
        {
            "name": "UnitConverter",
            "description": "Convert units (length, weight, temp)",
            "enabled_by_default": True,
            "category": "builtin",
        },
        {
            "name": "TextProcessor",
            "description": "Text ops (count, case, reverse)",
            "enabled_by_default": True,
            "category": "builtin",
        },
        {
            "name": "RandomGenerator",
            "description": "Random numbers and strings",
            "enabled_by_default": True,
            "category": "builtin",
        },
        {
            "name": "CodeExecutor",
            "description": "Run Python/JS/Bash in sandbox",
            "enabled_by_default": False,
            "category": "builtin",
        },
    ]
    
    # MCP Servers available in the playground
    MCP_SERVERS = [
        {
            "name": "GitHub",
            "description": "Issues, PRs, repos, commits",
            "registry_key": "github",
            "requires_token": True,
            "token_label": "Personal Access Token",
            "setup_url": "https://github.com/settings/tokens",
            "token_hint": "ghp_... with 'repo' scope",
        },
        {
            "name": "Slack",
            "description": "Messages, channels, users",
            "registry_key": "slack",
            "requires_token": True,
            "token_label": "User OAuth Token",
            "setup_url": "https://api.slack.com/apps",
            "token_hint": "xoxp-... from a Slack app",
        },
        {
            "name": "Notion",
            "description": "Pages, databases, blocks",
            "registry_key": "notion",
            "requires_token": True,
            "token_label": "Integration Secret",
            "setup_url": "https://www.notion.so/profile/integrations",
            "token_hint": "secret_... from integration",
        },
        {
            "name": "Tavily",
            "description": "Web search and research",
            "registry_key": "tavily",
            "requires_token": True,
            "token_label": "API Key",
            "setup_url": "https://app.tavily.com/",
            "token_hint": "tvly-... from dashboard",
        },
    ]


settings = Settings()
