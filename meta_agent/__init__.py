"""
OnsetLab Meta-Agent
===================
Intelligent agent that discovers MCP servers and generates Colab notebooks
for building AI agents.

This is a separate backend service from the onsetlab SDK.

Usage:
    from meta_agent import run_meta_agent
    
    result = await run_meta_agent(
        problem_statement="I need an agent that manages my calendar",
        anthropic_api_key="sk-ant-...",
        tavily_api_key="tvly-..."
    )
    
    print(result["colab_notebook"])
    
Sync Usage:
    from meta_agent import run_meta_agent_sync
    
    result = run_meta_agent_sync(
        problem_statement="I need an agent that manages my calendar",
        anthropic_api_key="sk-ant-...",
        tavily_api_key="tvly-..."
    )
"""

# Main API
from .graph import (
    create_meta_agent_graph,
    run_meta_agent,
    run_meta_agent_sync,
    get_graph,
)

# State schemas
from .state import (
    MetaAgentState,
    MCPServerDiscovery,
    APIServerFallback,
    TokenGuide,
    create_initial_state,
)

__all__ = [
    # Main API
    "create_meta_agent_graph",
    "run_meta_agent",
    "run_meta_agent_sync",
    "get_graph",
    
    # State schemas
    "MetaAgentState",
    "MCPServerDiscovery", 
    "APIServerFallback",
    "TokenGuide",
    "create_initial_state",
]
