"""
OnsetLab
========
Build AI agents with fine-tuned SLMs and MCP tools.

Usage:
    from onsetlab import AgentBuilder, ToolSchema, MCPServerConfig
    
    builder = AgentBuilder(
        problem_statement="Calendar assistant",
        tools=[...],
        mcp_servers=[...],
        api_key="sk-..."
    )
    agent = builder.build()
    agent.save("./my_agent")
"""

__version__ = "0.1.0"

# Main entry point
from .builder import AgentBuilder, Agent, BuildConfig

# Core schemas
from .utils.schemas import ToolSchema, MCPServerConfig, load_tools_from_file, load_tools_from_json

# Validation
from .utils.validator import Validator, ValidationResult, validate_training_data

# Low-level APIs (for advanced users)
from .synthesis.prompt_generator import PromptGenerator, generate_system_prompt, generate_minimal_prompt
from .synthesis.data_generator import DataGenerator, GeneratorConfig, generate_training_data

__all__ = [
    # Main API (what most users need)
    "AgentBuilder",
    "Agent",
    "BuildConfig",
    "ToolSchema",
    "MCPServerConfig",
    # Helpers
    "load_tools_from_file",
    "load_tools_from_json",
    # Advanced/Low-level
    "Validator",
    "ValidationResult",
    "PromptGenerator",
    "DataGenerator",
]
