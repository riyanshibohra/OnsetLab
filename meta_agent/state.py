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
    package: str                    # "@cocal/google-calendar-mcp" or "ghcr.io/owner/image"
    server_type: str                # "npm" | "docker" | "go" | "python" | "binary"
    auth_type: str                  # "oauth" | "token" | "api_key" | "none"
    env_vars: list[str]             # ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"] - ALL required env vars
    tools: list[dict]               # Extracted tool schemas
    setup_url: Optional[str]        # Link to setup guide
    confidence: float               # 0.0 - 1.0 how confident we are
    docker_image: Optional[str]     # Docker image (for server_type="docker"|"go"|"python")
    repo_url: Optional[str]         # Source repository URL


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
    State for the Meta-Agent LangGraph (Registry-Based v2.0).
    
    Simplified state for registry-based tool loading with HITL.
    """
    # Input (required at start)
    problem_statement: str
    anthropic_api_key: str
    
    # After analyze_problem
    identified_services: list[str]           # ["github", "slack", "google_calendar"]
    
    # After discover_servers (NEW - from MCP Registry)
    discovered_servers: list[dict]           # Servers found in MCP Registry
    discovery_errors: list[str]              # Services not found
    
    # After verify_servers (NEW - verification results)
    verified_servers: list[dict]             # Verified server configs with scores
    verification_summary: str                # Summary of verification
    
    # After load_registry (legacy or fallback)
    all_tools: list[dict]                    # All tools from registry files
    mcp_servers: list[dict]                  # MCP server configs from registry
    registry_services: list[str]             # Successfully loaded services
    
    # After filter_tools
    filtered_tools: list[dict]               # LLM-selected relevant tools
    
    # HITL (Human-in-the-Loop)
    user_feedback: str                       # User input: "looks good" / "add X" / "remove Y"
    feedback_action: str                     # "approved" / "add_tools" / "remove_tools"
    tools_to_add: list[str]                  # Tool names to add
    tools_to_remove: list[str]               # Tool names to remove
    final_tools: list[dict]                  # User-approved final tool list
    
    # Skill (for guided data generation)
    full_skill: str                          # Detailed skill document for data generation
    condensed_rules: str                     # Short rules for system prompt (~200 tokens)
    
    # Output artifacts
    token_guides: list[TokenGuide]
    colab_notebook: str
    colab_notebook_url: Optional[str]
    
    # Legacy fields (for backwards compatibility with old nodes)
    tool_schemas: list[dict]                 # Same as all_tools
    filtered_tool_schemas: list[dict]        # Same as filtered_tools
    filtered_mcp_servers: list[dict]
    filtered_api_servers: list[dict]
    api_servers: list[dict]
    
    # Error tracking
    errors: Annotated[list[str], add]


def create_initial_state(
    problem_statement: str,
    selected_services: list[str],
    anthropic_api_key: str,
) -> MetaAgentState:
    """
    Create a properly initialized state for the meta-agent.
    
    Args:
        problem_statement: Description of what the agent should do
        selected_services: Services selected by user in UI (e.g., ["github", "slack"])
        anthropic_api_key: Anthropic API key for Claude LLM calls
        
    Returns:
        Initialized MetaAgentState
    """
    return {
        # Input (from UI)
        "problem_statement": problem_statement,
        "anthropic_api_key": anthropic_api_key,
        "identified_services": selected_services,  # From UI selection
        
        # Discovery (NEW)
        "discovered_servers": [],
        "discovery_errors": [],
        "verified_servers": [],
        "verification_summary": "",
        
        # Initialize (legacy)
        "all_tools": [],
        "mcp_servers": [],
        "registry_services": [],
        "filtered_tools": [],
        
        # HITL
        "user_feedback": "",
        "feedback_action": "",
        "tools_to_add": [],
        "tools_to_remove": [],
        "final_tools": [],
        
        # Skill
        "full_skill": "",
        "condensed_rules": "",
        
        # Output
        "token_guides": [],
        "colab_notebook": "",
        "colab_notebook_url": None,
        
        # Legacy (backwards compat)
        "tool_schemas": [],
        "filtered_tool_schemas": [],
        "filtered_mcp_servers": [],
        "filtered_api_servers": [],
        "api_servers": [],
        
        # Errors
        "errors": [],
    }
