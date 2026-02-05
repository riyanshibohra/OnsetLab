"""
OnsetLab - Reliable SLM agents with REWOO strategy.

Plan once, execute fast, verify always.
"""

__version__ = "0.1.0"

from .agent import Agent
from .mcp import MCPServer
from .tools import (
    BaseTool,
    Calculator,
    DateTime,
    UnitConverter,
    TextProcessor,
    RandomGenerator,
)

__all__ = [
    # Core
    "Agent",
    "MCPServer",
    # Tools
    "BaseTool",
    "Calculator",
    "DateTime",
    "UnitConverter",
    "TextProcessor",
    "RandomGenerator",
    # Meta
    "__version__",
]
