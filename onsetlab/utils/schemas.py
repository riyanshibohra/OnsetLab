"""
OnsetLab Schemas
================
Core data classes for tool schemas and MCP server configurations.

These are the primary data structures passed to the AgentBuilder and used
throughout the SDK pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class ToolSchema:
    """
    Represents an MCP tool definition.
    
    This is the standard format for describing tools that the agent can call.
    Compatible with MCP (Model Context Protocol) tool definitions.
    
    Attributes:
        name: Tool identifier (e.g., "list-events", "create-event")
        description: Human-readable description of what the tool does
        parameters: JSON Schema defining the tool's input parameters
        required_params: List of parameter names that are required
    
    Example:
        >>> tool = ToolSchema(
        ...     name="list-events",
        ...     description="List calendar events within a time range",
        ...     parameters={
        ...         "calendarId": {"type": "string", "description": "Calendar ID"},
        ...         "timeMin": {"type": "string", "description": "Start time (ISO)"},
        ...         "timeMax": {"type": "string", "description": "End time (ISO)"},
        ...     },
        ...     required_params=["calendarId"]
        ... )
    """
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    required_params: list = field(default_factory=list)
    
    @classmethod
    def from_mcp(cls, mcp_tool: dict) -> "ToolSchema":
        """
        Create a ToolSchema from an MCP tool definition.
        
        MCP tools have this structure:
        {
            "name": "tool-name",
            "description": "What the tool does",
            "inputSchema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        
        Args:
            mcp_tool: Dictionary from MCP tools/list response
            
        Returns:
            ToolSchema instance
        """
        input_schema = mcp_tool.get("inputSchema", {})
        return cls(
            name=mcp_tool["name"],
            description=mcp_tool.get("description", ""),
            parameters=input_schema.get("properties", {}),
            required_params=input_schema.get("required", [])
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format (for serialization).
        
        Returns:
            Dictionary representation of the tool schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __repr__(self) -> str:
        param_count = len(self.parameters)
        return f"ToolSchema(name='{self.name}', params={param_count})"


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server.
    
    Describes how to connect to and authenticate with an MCP server.
    This information is used when packaging the agent runtime.
    
    Attributes:
        package: NPM package name (e.g., "@cocal/google-calendar-mcp")
        auth_type: Authentication method ("oauth", "token", "api_key", "none")
        env_var: Environment variable name for credentials (optional)
        description: Human-readable description of the server
        setup_url: URL to setup/documentation guide (optional)
    
    Example:
        >>> server = MCPServerConfig(
        ...     package="@cocal/google-calendar-mcp",
        ...     auth_type="oauth",
        ...     env_var="GOOGLE_OAUTH_CREDENTIALS",
        ...     description="Google Calendar integration",
        ...     setup_url="https://github.com/cocal/google-calendar-mcp#setup"
        ... )
    """
    package: str
    auth_type: str = "none"  # "oauth", "token", "api_key", "none"
    env_var: Optional[str] = None
    description: str = ""
    setup_url: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format (for serialization)."""
        result = {
            "package": self.package,
            "auth_type": self.auth_type,
        }
        if self.env_var:
            result["env_var"] = self.env_var
        if self.description:
            result["description"] = self.description
        if self.setup_url:
            result["setup_url"] = self.setup_url
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __repr__(self) -> str:
        return f"MCPServerConfig(package='{self.package}', auth='{self.auth_type}')"


# ============================================================================
# Helper Functions
# ============================================================================

def load_tools_from_file(path: str) -> list[ToolSchema]:
    """
    Load tool schemas from a JSON file.
    
    Args:
        path: Path to JSON file containing array of MCP tool definitions
        
    Returns:
        List of ToolSchema objects
    """
    with open(path) as f:
        tools_data = json.load(f)
    return [ToolSchema.from_mcp(t) for t in tools_data]


def load_tools_from_json(json_str: str) -> list[ToolSchema]:
    """
    Load tool schemas from a JSON string.
    
    Args:
        json_str: JSON string containing array of MCP tool definitions
        
    Returns:
        List of ToolSchema objects
    """
    tools_data = json.loads(json_str)
    return [ToolSchema.from_mcp(t) for t in tools_data]
