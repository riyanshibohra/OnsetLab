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
    DataGenerator,
    GeneratorConfig,
    generate_training_data,
    calculate_recommended_examples,
)

__all__ = [
    # Prompt generation
    "PromptGenerator",
    "PromptConfig",
    "generate_system_prompt",
    "generate_minimal_prompt",
    # Data generation
    "DataGenerator",
    "GeneratorConfig",
    "generate_training_data",
    "calculate_recommended_examples",
]
