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
from .data_generator_v3 import (
    BatchedDataGenerator,
    BatchGenConfig,
    generate_training_data_batched,
)

__all__ = [
    # Prompt generation
    "PromptGenerator",
    "PromptConfig",
    "generate_system_prompt",
    "generate_minimal_prompt",
    # Data generation (v1 - legacy)
    "DataGenerator",
    "GeneratorConfig",
    "generate_training_data",
    "calculate_recommended_examples",
    # Data generation (v3 - batched, recommended)
    "BatchedDataGenerator",
    "BatchGenConfig",
    "generate_training_data_batched",
]
