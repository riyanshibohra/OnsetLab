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
    # Data generation
    "BatchedDataGenerator",
    "BatchGenConfig",
    "generate_training_data_batched",
    "recommend_examples",
]
