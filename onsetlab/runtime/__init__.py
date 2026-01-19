"""
OnsetLab Runtime Module
=======================
Packages trained agents for deployment.

Supports:
- Ollama (GGUF) - Simple CLI runtime
- Python script - Direct Python inference
"""

from .packager import AgentPackager, PackageConfig, RuntimeType

__all__ = [
    "AgentPackager",
    "PackageConfig",
    "RuntimeType",
]
