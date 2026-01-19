"""
OnsetLab Training Module
========================
Fine-tuning SLMs for tool calling using Unsloth.
"""

from .unsloth_trainer import (
    UnslothTrainer,
    TrainerConfig,
    SUPPORTED_MODELS,
)

__all__ = [
    "UnslothTrainer",
    "TrainerConfig",
    "SUPPORTED_MODELS",
]
