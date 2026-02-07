"""MCP server connection endpoints."""

import logging
from fastapi import APIRouter, HTTPException, Request, Response

from ..models.schemas import (
    MCPConnectRequest,
    MCPConnectResponse,
    MCPDisconnectRequest,
    MCPStatusResponse,
    MCPStatusServer,
)
from ..services.session_store import session_store
from ..services.mcp_manager import mcp_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


def _get_session_id(request: Request, response: Response) -> str:
    """Get or create a session, set cookie, return session ID."""
    session_id = request.cookies.get("session_id")
    session = session_store.get_or_create(session_id)

    if session_id != session.id:
        response.set_cookie(
            key="session_id",
            value=session.id,
            httponly=True,
            samesite="lax",
            max_age=86400,
        )

    return session.id


@router.post("/connect", response_model=MCPConnectResponse)
async def connect_mcp(
    request: Request,
    response: Response,
    body: MCPConnectRequest,
):
    """
    Connect to an MCP server.

    Spawns the MCP subprocess, performs handshake, discovers tools.
    The connection persists for the session lifetime.
    Token is stored in server-side memory only — never persisted.
    """
    session_id = _get_session_id(request, response)

    try:
        tool_names = mcp_manager.connect(session_id, body.server, body.token)
        return MCPConnectResponse(
            connected=True,
            server=body.server,
            tools=tool_names,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"MCP connection failed: {e}",
        )
    except Exception as e:
        logger.error(f"MCP connect error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error connecting to {body.server}: {e}",
        )


@router.post("/disconnect")
async def disconnect_mcp(
    request: Request,
    response: Response,
    body: MCPDisconnectRequest,
):
    """Disconnect an MCP server (kills subprocess)."""
    session_id = _get_session_id(request, response)
    was_connected = mcp_manager.disconnect(session_id, body.server)

    return {
        "disconnected": was_connected,
        "server": body.server,
    }


@router.get("/status", response_model=MCPStatusResponse)
async def mcp_status(request: Request, response: Response):
    """Get MCP connection status for the current session."""
    session_id = _get_session_id(request, response)
    raw = mcp_manager.get_status(session_id)

    servers = {
        key: MCPStatusServer(connected=info["connected"], tools=info["tools"])
        for key, info in raw.items()
    }

    return MCPStatusResponse(servers=servers)
