"""
Meta-Agent API
==============
FastAPI server for the meta-agent.
"""

from .server import app, create_app

__all__ = ["app", "create_app"]
