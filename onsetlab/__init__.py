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

# Core schemas for defining tools and servers
from .utils.schemas import (
    ToolSchema, 
    MCPServerConfig,
    APIToolSchema,      # NEW: For direct API endpoints
    APIServerConfig,    # NEW: For API-based services
    load_tools_from_file, 
    load_tools_from_json
)

# =============================================================================
# Advanced APIs (for users who want more control)
# =============================================================================

# Training configuration
from .training.unsloth_trainer import TrainerConfig, SUPPORTED_MODELS

# Runtime/packaging options
from .runtime.packager import RuntimeType

# Validation
from .utils.validator import Validator, ValidationResult, validate_training_data

# Synthesis APIs
from .synthesis.prompts import generate_prompt_for_3b, generate_prompt_for_7b_plus
from .synthesis.data_generator import (
    DataGenerator, 
    DataGenConfig, 
    generate_training_data,
    recommend_dataset_size,
    print_dataset_recommendation,
)

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
    "APIToolSchema",
    "APIServerConfig",
    
    # Helpers
    "load_tools_from_file",
    "load_tools_from_json",
    
    # Training options
    "TrainerConfig",
    "SUPPORTED_MODELS",
    "RuntimeType",
    
    # Validation
    "Validator",
    "ValidationResult",
    "validate_training_data",
    
    # Prompt generation (no LLM needed)
    "generate_prompt_for_3b",
    "generate_prompt_for_7b_plus",
    
    # Data generation
    "DataGenerator",
    "DataGenConfig",
    "generate_training_data",
    "recommend_dataset_size",
    "print_dataset_recommendation",
]
