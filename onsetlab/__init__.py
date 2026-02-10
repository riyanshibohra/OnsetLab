"""
OnsetLab - Reliable SLM agents with hybrid REWOO/ReAct strategy.

Build, benchmark, and package local AI agents.
"""

__version__ = "0.2.0"

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
from .benchmark import Benchmark
from .router import Router, Strategy
from .skills import generate_tool_rules, generate_examples

__all__ = [
    # Core
    "Agent",
    "MCPServer",
    "Benchmark",
    "Router",
    "Strategy",
    # Skills (auto-generated from schemas)
    "generate_tool_rules",
    "generate_examples",
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
