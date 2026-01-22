"""
OnsetLab
========
Build AI agents with fine-tuned SLMs and MCP tools.

Usage:
    from onsetlab import AgentBuilder, BuildConfig, ToolSchema
    
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

# =============================================================================
# Main API (what most users need)
# =============================================================================
from .builder import AgentBuilder, Agent, BuildConfig

# Core schemas for defining tools and MCP servers
from .utils.schemas import ToolSchema, MCPServerConfig, load_tools_from_file, load_tools_from_json

# =============================================================================
# Advanced APIs (for users who want more control)
# =============================================================================

# Training configuration
from .training.unsloth_trainer import TrainerConfig, SUPPORTED_MODELS

# Runtime/packaging options
from .runtime.packager import RuntimeType

# Validation
from .utils.validator import Validator, ValidationResult, validate_training_data

# Low-level synthesis APIs
from .synthesis.prompt_generator import PromptGenerator, generate_system_prompt, generate_minimal_prompt
from .synthesis.data_generator_v3 import BatchedDataGenerator, BatchGenConfig, generate_training_data_batched

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Main API (95% of users only need these)
    "AgentBuilder",
    "Agent", 
    "BuildConfig",
    "ToolSchema",
    "MCPServerConfig",
    
    # Helpers
    "load_tools_from_file",
    "load_tools_from_json",
    
    # Training options
    "TrainerConfig",
    "SUPPORTED_MODELS",
    "RuntimeType",
    
    # Advanced/Low-level (for power users)
    "Validator",
    "ValidationResult",
    "validate_training_data",
    "PromptGenerator",
    "generate_system_prompt",
    
    # Batched data generation (v3)
    "BatchedDataGenerator",
    "BatchGenConfig",
    "generate_training_data_batched",
]
