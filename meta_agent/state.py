"""
Meta-Agent State Schemas
========================
TypedDict definitions for the LangGraph state.
"""

from typing import TypedDict, Annotated, Optional, Any
from operator import add


class MCPServerDiscovery(TypedDict):
    """Represents a discovered MCP server."""
    service: str                    # "google_calendar"
    package: str                    # "@cocal/google-calendar-mcp"
    auth_type: str                  # "oauth" | "token" | "api_key" | "none"
    env_vars: list[str]             # ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"] - ALL required env vars
    tools: list[dict]               # Extracted tool schemas
    setup_url: Optional[str]        # Link to setup guide
    confidence: float               # 0.0 - 1.0 how confident we are


class APIEndpoint(TypedDict):
    """Detailed API endpoint definition."""
    name: str                       # "send_sms" (function name for tools.py)
    method: str                     # "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
    path: str                       # "/messages" or "/messages/{id}"
    description: str                # What this endpoint does
    parameters: dict                # Query/path parameters (same format as MCP tool params)
    required_params: list[str]      # Required parameter names
    request_body: Optional[dict]    # Body schema for POST/PUT/PATCH
    response_schema: Optional[dict] # Expected response format


class APIServerFallback(TypedDict):
    """Represents a service that needs API implementation (no good MCP found)."""
    service: str                    # "twilio"
    reason: str                     # "No MCP server found"
    
    # API Details
    api_docs_url: Optional[str]     # "https://www.twilio.com/docs/api"
    base_url: str                   # "https://api.twilio.com/v1"
    
    # Authentication
    auth_type: str                  # "bearer" | "api_key" | "basic" | "oauth"
    auth_header: Optional[str]      # "Authorization: Bearer {token}" or "X-API-Key: {key}"
    env_var: str                    # "TWILIO_API_KEY"
    
    # Endpoints (detailed)
    endpoints: list[APIEndpoint]    # List of API endpoints to implement


class TokenGuide(TypedDict):
    """Step-by-step guide for obtaining access tokens."""
    service: str
    auth_type: str
    steps: list[str]                # Step-by-step instructions
    env_var: str                    # Primary env var (for backwards compat)
    env_vars: list[str]             # ALL required env vars


class MetaAgentState(TypedDict, total=False):
    """
    State for the Meta-Agent LangGraph.
    
    This state is passed between nodes and accumulates results
    as the agent discovers MCP servers and generates the notebook.
    
    Note: total=False makes all fields optional, which is needed
    for LangGraph state updates where we only return changed fields.
    """
    # Input (required at start)
    problem_statement: str
    anthropic_api_key: str
    tavily_api_key: str
    
    # Parsed from problem
    identified_services: list[str]           # ["google_calendar", "gmail", "slack"]
    
    # Discovery state (loop control)
    current_service_index: int
    current_service: Optional[str]           # Current service being processed
    search_results: Optional[str]            # Raw search results for current service
    result_quality: Optional[str]            # "good_mcp" | "no_mcp"
    
    # Temporary evaluation results (passed between evaluate_mcp and extract_schemas/mark_as_api)
    _eval_package_name: Optional[str]
    _eval_github_url: Optional[str]
    _eval_auth_type: Optional[str]
    _eval_env_var: Optional[str]
    _eval_trust_level: Optional[str]
    _eval_stars: Optional[int]
    _eval_confidence: Optional[float]
    _eval_reasoning: Optional[str]
    _eval_candidates: list[dict]          # ALL candidates for retry logic
    _eval_candidate_index: int            # Current candidate being tried
    
    # Accumulated results (using Annotated[..., add] for reducer)
    mcp_servers: Annotated[list[MCPServerDiscovery], add]
    api_servers: Annotated[list[APIServerFallback], add]
    tool_schemas: Annotated[list[dict], add]
    
    # Filtered results (NOT accumulated - set by filter_tools, used by generate_notebook)
    filtered_mcp_servers: list[MCPServerDiscovery]
    filtered_api_servers: list[APIServerFallback]
    filtered_tool_schemas: list[dict]
    
    # Output artifacts
    token_guides: list[TokenGuide]
    colab_notebook: str
    colab_notebook_url: Optional[str]
    
    # Error tracking
    errors: Annotated[list[str], add]


def create_initial_state(
    problem_statement: str,
    anthropic_api_key: str,
    tavily_api_key: str,
) -> MetaAgentState:
    """
    Create a properly initialized state for the meta-agent.
    
    Args:
        problem_statement: Description of what the agent should do
        anthropic_api_key: Anthropic API key for Claude LLM calls
        tavily_api_key: Tavily API key for web search
        
    Returns:
        Initialized MetaAgentState
    """
    return {
        # Input
        "problem_statement": problem_statement,
        "anthropic_api_key": anthropic_api_key,
        "tavily_api_key": tavily_api_key,
        
        # Initialize lists and counters
        "identified_services": [],
        "current_service_index": 0,
        "current_service": None,
        "search_results": None,
        "result_quality": None,
        
        # Temp fields
        "_eval_package_name": None,
        "_eval_github_url": None,
        "_eval_auth_type": None,
        "_eval_env_var": None,
        "_eval_trust_level": None,
        "_eval_stars": None,
        "_eval_confidence": None,
        "_eval_reasoning": None,
        "_eval_candidates": [],           # ALL candidates for retry
        "_eval_candidate_index": 0,       # Current candidate index
        
        # Accumulated results (start empty)
        "mcp_servers": [],
        "api_servers": [],
        "tool_schemas": [],
        
        # Filtered results (set by filter_tools)
        "filtered_mcp_servers": [],
        "filtered_api_servers": [],
        "filtered_tool_schemas": [],
        
        # Output artifacts
        "token_guides": [],
        "colab_notebook": "",
        "colab_notebook_url": None,
        
        # Errors
        "errors": [],
    }
