"""
OnsetLab Synthesis Module
=========================
LLM-powered generation of system prompts and training data.
"""

from .prompt_generator import (
    PromptGenerator,
    PromptConfig,
    generate_system_prompt,
    generate_minimal_prompt,
    # Single-Tool Architecture (v2.0)
    generate_single_tool_prompt,
    get_clarification_examples,
)
from .data_generator import (
    BatchedDataGenerator,
    BatchGenConfig,
    generate_training_data_batched,
    recommend_examples,
)

__all__ = [
    # Prompt generation
    "PromptGenerator",
    "PromptConfig",
    "generate_system_prompt",
    "generate_minimal_prompt",
    # Single-Tool Architecture (v2.0)
    "generate_single_tool_prompt",
    "get_clarification_examples",
    # Data generation
    "BatchedDataGenerator",
    "BatchGenConfig",
    "generate_training_data_batched",
    "recommend_examples",
]
