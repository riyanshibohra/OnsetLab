"""Shared fixtures and markers for OnsetLab tests."""

import subprocess
import pytest


def _ollama_available() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


OLLAMA_AVAILABLE = _ollama_available()

# Marker for tests that require Ollama
requires_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE,
    reason="Ollama is not installed or not running",
)
