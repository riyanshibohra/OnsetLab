"""Pydantic schemas for API requests/responses."""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class PlanStep(BaseModel):
    """A single step in the execution plan."""
    id: str = Field(..., description="Step ID, e.g., '#E1'")
    tool: str = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    result: Optional[str] = Field(None, description="Execution result")
    status: Literal["pending", "running", "done", "error"] = "pending"


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000)
    tools: List[str] = Field(default_factory=list, description="Enabled tool names")
    model: str = Field(default="qwen2.5:7b", description="Model to use")
    github_token: Optional[str] = Field(None, description="GitHub PAT for MCP")


class PipelineTrace(BaseModel):
    """Trace of the agent pipeline — what happened at each stage."""
    router_decision: str = ""          # "TOOL" or "DIRECT"
    router_reason: str = ""            # why the router chose this
    tools_total: int = 0               # total tools available
    tools_filtered: int = 0            # tools selected for planner
    tools_selected: List[str] = Field(default_factory=list)
    tool_rules: str = ""               # auto-generated rules sent to planner
    planner_think: str = ""            # model's reasoning before the plan
    planner_prompt: str = ""           # full prompt sent to model (inspector)
    fallback_used: bool = False        # did ReAct fallback trigger?
    fallback_reason: str = ""          # why fallback was needed


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    answer: str
    plan: List[PlanStep] = Field(default_factory=list)
    results: Dict[str, str] = Field(default_factory=dict)
    strategy: str = "rewoo"  # direct, rewoo, rewoo->react
    slm_calls: int = 0
    requests_remaining: int
    trace: Optional[PipelineTrace] = None


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    requests_used: int
    requests_limit: int
    requests_remaining: int
    created_at: datetime
    model: str
    tools: List[str]


class ExportRequest(BaseModel):
    """Request body for export endpoint."""
    format: Literal["config", "docker", "docker-vllm", "binary"]
    tools: List[str] = Field(default_factory=list)
    model: str = Field(default="qwen2.5:7b")


class ExportResponse(BaseModel):
    """Response from export endpoint."""
    filename: str
    content_type: str
    download_url: Optional[str] = None


class ToolInfo(BaseModel):
    """Tool information for UI."""
    name: str
    description: str
    enabled_by_default: bool
    category: str = "builtin"


class ModelInfo(BaseModel):
    """Model information for UI."""
    id: str
    display_name: str
    description: str
    params: str = ""
    badge: str = ""  # "recommended", "best value", "fastest", etc.


class MCPServerInfo(BaseModel):
    """MCP server information for UI."""
    name: str
    description: str
    registry_key: str
    requires_token: bool
    token_label: str = ""
    setup_url: str = ""
    token_hint: str = ""


class MCPConnectRequest(BaseModel):
    """Request to connect an MCP server."""
    server: str = Field(..., description="Registry key, e.g. 'github'")
    token: str = Field(..., min_length=1, description="Auth token / API key")


class MCPConnectResponse(BaseModel):
    """Response after connecting an MCP server."""
    connected: bool
    server: str
    tools: List[str] = Field(default_factory=list, description="Discovered tool names")


class MCPDisconnectRequest(BaseModel):
    """Request to disconnect an MCP server."""
    server: str = Field(..., description="Registry key")


class MCPStatusServer(BaseModel):
    """Status of a single MCP server connection."""
    connected: bool
    tools: List[str] = Field(default_factory=list)


class MCPStatusResponse(BaseModel):
    """Connection status for all MCP servers in the session."""
    servers: Dict[str, MCPStatusServer] = Field(default_factory=dict)


class RateLimitError(BaseModel):
    """Rate limit exceeded response."""
    error: str = "rate_limit_exceeded"
    message: str = "You've used all 5 free requests!"
    cta: Dict[str, str] = Field(default_factory=lambda: {
        "text": "Run unlimited locally",
        "pip": "pip install onsetlab",
        "github": "https://github.com/riyanshibohra/onsetlab"
    })
