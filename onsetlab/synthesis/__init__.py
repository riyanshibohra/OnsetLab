"""
OnsetLab Synthesis Module
=========================
Training data and system prompt generation.

Single-Tool Architecture:
- One tool call per turn
- Strong clarification handling
- 75% single-tool, 25% edge cases
- Concise prompts optimized for 3B models
"""

# Prompt generation (concise, template-based)
from .prompts import (
    generate_prompt_for_3b,
    generate_prompt_for_7b_plus,
    get_default_prompt,
)

# Data generation (orchestrator; uses .generators and .validators internally)
from .data_generator import (
    DataGenerator,
    DataGenConfig,
    generate_training_data,
    recommend_dataset_size,
    print_dataset_recommendation,
)

__all__ = [
    # Prompt generation (no LLM needed)
    "generate_prompt_for_3b",
    "generate_prompt_for_7b_plus",
    "get_default_prompt",
    # Data generation
    "DataGenerator",
    "DataGenConfig",
    "generate_training_data",
    "recommend_dataset_size",
    "print_dataset_recommendation",
]
