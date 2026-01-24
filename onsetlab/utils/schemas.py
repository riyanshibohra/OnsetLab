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
        package: NPM package name (e.g., "@modelcontextprotocol/server-github")
        name: Short name for the server (e.g., "github"). Auto-derived if not set.
        auth_type: Authentication method ("oauth", "token", "api_key", "cookie", "none")
        env_var: Environment variable name for credentials
        description: Human-readable description of the server
        setup_url: URL to setup/documentation guide
        command: Command to run the server (default: "npx")
        args: Arguments for the command (default: ["-y", package])
        tools: List of tool names this server provides
        example_value: Example credential value for .env.example
    
    Example:
        >>> server = MCPServerConfig(
        ...     package="@modelcontextprotocol/server-github",
        ...     name="github",
        ...     auth_type="token",
        ...     env_var="GITHUB_PERSONAL_ACCESS_TOKEN",
        ...     description="GitHub API integration",
        ...     setup_url="https://github.com/settings/tokens",
        ...     tools=["list_issues", "create_issue", "search_repositories"],
        ...     example_value="ghp_xxxxxxxxxxxxxxxxxxxx"
        ... )
    """
    package: str
    name: Optional[str] = None  # Auto-derived from package if not set
    auth_type: str = "none"  # "oauth", "token", "api_key", "cookie", "none"
    env_var: Optional[str] = None
    description: str = ""
    setup_url: Optional[str] = None
    command: str = "npx"  # Command to run server
    args: Optional[list] = None  # Default: ["-y", package]
    tools: Optional[list] = None  # Tool names this server provides
    example_value: Optional[str] = None  # Example credential for .env
    
    def __post_init__(self):
        """Auto-derive name from package if not set."""
        if self.name is None:
            # Extract name from package (e.g., "@org/server-github" -> "github")
            self.name = self.package.split("/")[-1].replace("-mcp-server", "").replace("server-", "").replace("-mcp", "")
        if self.args is None:
            self.args = ["-y", self.package]
        if self.tools is None:
            self.tools = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary format (for serialization)."""
        result = {
            "package": self.package,
            "name": self.name,
            "auth_type": self.auth_type,
            "command": self.command,
            "args": self.args,
            "tools": self.tools or [],
        }
        if self.env_var:
            result["env_var"] = self.env_var
        if self.description:
            result["description"] = self.description
        if self.setup_url:
            result["setup_url"] = self.setup_url
        if self.example_value:
            result["example_value"] = self.example_value
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


# ============================================================================
# API Server Configuration (For services without MCP servers)
# ============================================================================

@dataclass
class APIToolSchema:
    """
    Represents a single API endpoint as a tool.
    
    This is for services that don't have MCP servers available.
    The meta-agent provides this information after analyzing the API.
    
    Attributes:
        name: Tool identifier (e.g., "send_sms", "create_ticket")
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        path: API endpoint path (e.g., "/messages", "/tickets/{id}")
        description: Human-readable description
        parameters: JSON Schema for query/path params (same format as ToolSchema)
        required_params: List of required parameter names
        request_body_schema: JSON Schema for POST/PUT request body
        response_schema: Expected response structure (for documentation)
    
    Example:
        >>> tool = APIToolSchema(
        ...     name="send_sms",
        ...     method="POST",
        ...     path="/messages",
        ...     description="Send an SMS message",
        ...     parameters={
        ...         "to": {"type": "string", "description": "Phone number"},
        ...         "body": {"type": "string", "description": "Message text"}
        ...     },
        ...     required_params=["to", "body"],
        ...     request_body_schema={
        ...         "to": {"type": "string"},
        ...         "body": {"type": "string"}
        ...     }
        ... )
    """
    name: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    path: str    # e.g., "/messages", "/repos/{owner}/{repo}/issues"
    description: str = ""
    parameters: dict = field(default_factory=dict)  # Query/path params
    required_params: list = field(default_factory=list)
    request_body_schema: dict = field(default_factory=dict)  # For POST/PUT
    response_schema: dict = field(default_factory=dict)  # Expected response
    
    def to_tool_schema(self) -> ToolSchema:
        """Convert to standard ToolSchema for training."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required_params=self.required_params
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary format (for serialization)."""
        return {
            "name": self.name,
            "method": self.method,
            "path": self.path,
            "description": self.description,
            "parameters": self.parameters,
            "required_params": self.required_params,
            "request_body_schema": self.request_body_schema,
            "response_schema": self.response_schema
        }
    
    def __repr__(self) -> str:
        return f"APIToolSchema(name='{self.name}', {self.method} {self.path})"


@dataclass
class APIServerConfig:
    """
    Configuration for an API-based service (no MCP server).
    
    For services where MCP servers don't exist or aren't reliable,
    we generate direct API wrapper functions instead.
    
    Attributes:
        name: Short name for the service (e.g., "twilio", "jira")
        base_url: API base URL (e.g., "https://api.twilio.com/2010-04-01")
        auth_type: Authentication method ("bearer", "api_key", "basic", "header")
        auth_env_var: Environment variable for credentials
        auth_header: Header name if auth_type is "header" (e.g., "X-API-Key")
        description: Human-readable description
        setup_url: URL to get API credentials
        tools: List of APIToolSchema for this service
        example_value: Example credential for .env.example
    
    Example:
        >>> server = APIServerConfig(
        ...     name="twilio",
        ...     base_url="https://api.twilio.com/2010-04-01",
        ...     auth_type="basic",
        ...     auth_env_var="TWILIO_API_KEY",
        ...     description="Twilio SMS and Voice API",
        ...     setup_url="https://console.twilio.com/",
        ...     tools=[
        ...         APIToolSchema(name="send_sms", method="POST", path="/messages", ...)
        ...     ]
        ... )
    """
    name: str
    base_url: str
    auth_type: str = "bearer"  # "bearer", "api_key", "basic", "header", "none"
    auth_env_var: Optional[str] = None
    auth_header: Optional[str] = None  # For auth_type="header"
    description: str = ""
    setup_url: Optional[str] = None
    tools: list = field(default_factory=list)  # List[APIToolSchema]
    example_value: Optional[str] = None
    
    def get_tool_schemas(self) -> list[ToolSchema]:
        """Get all tools as standard ToolSchema objects (for training)."""
        return [t.to_tool_schema() for t in self.tools]
    
    def get_tool_names(self) -> list[str]:
        """Get list of tool names."""
        return [t.name for t in self.tools]
    
    def to_dict(self) -> dict:
        """Convert to dictionary format (for serialization)."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "auth_type": self.auth_type,
            "auth_env_var": self.auth_env_var,
            "auth_header": self.auth_header,
            "description": self.description,
            "setup_url": self.setup_url,
            "tools": [t.to_dict() for t in self.tools],
            "example_value": self.example_value
        }
    
    def __repr__(self) -> str:
        return f"APIServerConfig(name='{self.name}', tools={len(self.tools)})"
