"""
OnsetLab - Reliable SLM agents with REWOO strategy.

Plan once, execute fast, verify always.
"""

__version__ = "0.1.0"

from .agent import Agent
from .mcp import MCPServer

__all__ = [
    "Agent",
    "MCPServer",
    "__version__",
]
