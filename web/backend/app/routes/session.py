"""Session management endpoints."""

from fastapi import APIRouter, Request, Response
from typing import List

from ..models.schemas import SessionInfo, ToolInfo, ModelInfo, MCPServerInfo
from ..services.session_store import session_store
from ..config import settings

router = APIRouter(prefix="/api", tags=["session"])


@router.get("/session", response_model=SessionInfo)
async def get_session(request: Request, response: Response):
    """Get or create session information."""
    session_id = request.cookies.get("session_id")
    session = session_store.get_or_create(session_id)
    
    # Set cookie if new session
    if session_id != session.id:
        response.set_cookie(
            key="session_id",
            value=session.id,
            httponly=True,
            samesite="lax",
            max_age=86400,
        )
    
    return SessionInfo(
        session_id=session.id,
        requests_used=session.requests_used,
        requests_limit=session.requests_limit,
        requests_remaining=session.requests_remaining,
        created_at=session.created_at,
        model=session.model,
        tools=session.tools,
    )


@router.get("/tools", response_model=List[ToolInfo])
async def list_tools():
    """List available built-in tools."""
    return [
        ToolInfo(
            name=tool["name"],
            description=tool["description"],
            enabled_by_default=tool["enabled_by_default"],
            category=tool.get("category", "builtin"),
        )
        for tool in settings.TOOLS
    ]


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available SLM models."""
    return [
        ModelInfo(
            id=model_id,
            display_name=config["display_name"],
            description=config["description"],
            params=config.get("params", ""),
        )
        for model_id, config in settings.MODELS.items()
    ]


@router.get("/mcp-servers", response_model=List[MCPServerInfo])
async def list_mcp_servers():
    """List available MCP servers for the playground."""
    return [
        MCPServerInfo(
            name=server["name"],
            description=server["description"],
            registry_key=server["registry_key"],
            requires_token=server["requires_token"],
            token_label=server.get("token_label", ""),
            setup_url=server.get("setup_url", ""),
            token_hint=server.get("token_hint", ""),
        )
        for server in settings.MCP_SERVERS
    ]
