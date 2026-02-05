"""
OnsetLab Runtime Module
=======================
Packages trained agents for deployment.

Supports:
- Ollama (GGUF) - Simple CLI runtime
- Python script - Direct Python inference
- API-based (OpenAI/Anthropic) - Quick start without training
"""

from .packager import AgentPackager, PackageConfig, RuntimeType
from .api_agent_template import generate_api_agent

__all__ = [
    "AgentPackager",
    "PackageConfig",
    "RuntimeType",
    "generate_api_agent",
]
