"""MCP connection manager — per-session MCP server lifecycle management."""

import logging
import concurrent.futures
from typing import Dict, List, Optional
from threading import Lock

from onsetlab.mcp.server import MCPServer
from onsetlab.tools import BaseTool

logger = logging.getLogger(__name__)

# Map registry key → environment variable name for token
_ENV_VAR_MAP = {
    "github": "GITHUB_PERSONAL_ACCESS_TOKEN",
    "slack": "SLACK_MCP_XOXP_TOKEN",
    "notion": "NOTION_TOKEN",
    "tavily": "TAVILY_API_KEY",
}

# Setup URLs for the token modal
_SETUP_URLS = {
    "github": "https://github.com/settings/tokens",
    "slack": "https://api.slack.com/apps",
    "notion": "https://www.notion.so/profile/integrations",
    "tavily": "https://app.tavily.com/",
}

# Token instructions
_TOKEN_HINTS = {
    "github": "Personal Access Token with 'repo' scope",
    "slack": "User OAuth Token (xoxp-...) from a Slack app",
    "notion": "Integration secret (starts with 'secret_')",
    "tavily": "API key from Tavily dashboard",
}


class MCPConnection:
    """Holds an active MCPServer and its metadata."""

    def __init__(self, server: MCPServer, registry_key: str):
        self.server = server
        self.registry_key = registry_key
        self.tool_names: List[str] = []

    def close(self):
        """Disconnect the MCP server (kills subprocess)."""
        try:
            self.server.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting MCP server {self.registry_key}: {e}")


class MCPManager:
    """
    Manages MCP server connections per session.

    - Thread-safe: all state behind a lock.
    - Tokens are only held in-memory; never persisted.
    - Subprocesses are killed on disconnect / session cleanup.
    """

    def __init__(self):
        # session_id → { registry_key → MCPConnection }
        self._connections: Dict[str, Dict[str, MCPConnection]] = {}
        self._lock = Lock()
        # Thread pool for MCP operations (avoids event-loop conflicts with FastAPI)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="mcp"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(
        self, session_id: str, registry_key: str, token: str
    ) -> List[str]:
        """
        Connect to an MCP server for a given session.

        Runs in a separate thread to avoid event-loop conflicts with FastAPI.

        Args:
            session_id: The session that owns this connection.
            registry_key: Registry key (e.g. "github", "slack").
            token: The user-supplied auth token / API key.

        Returns:
            List of discovered tool names.

        Raises:
            ValueError: Unknown registry key or missing env var mapping.
            RuntimeError: Connection / handshake failure.
        """
        env_var = _ENV_VAR_MAP.get(registry_key)
        if env_var is None:
            raise ValueError(
                f"Unknown MCP server: {registry_key}. "
                f"Available: {', '.join(_ENV_VAR_MAP.keys())}"
            )

        # Disconnect previous connection for this server if any
        self.disconnect(session_id, registry_key)

        # Run the blocking MCP connect in a thread (SyncMCPClient needs its own event loop)
        future = self._executor.submit(
            self._connect_in_thread, session_id, registry_key, token, env_var
        )
        return future.result(timeout=60)

    def _connect_in_thread(
        self, session_id: str, registry_key: str, token: str, env_var: str
    ) -> List[str]:
        """Actual connection logic — runs in a worker thread."""
        env = {env_var: token}
        server = MCPServer.from_registry(registry_key, env=env, timeout=30)

        try:
            server.connect()
        except Exception as e:
            logger.error(f"MCP connect failed for {registry_key}: {e}")
            raise RuntimeError(f"Failed to connect to {registry_key}: {e}")

        tool_names = server.get_tool_names()
        conn = MCPConnection(server=server, registry_key=registry_key)
        conn.tool_names = tool_names

        with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = {}
            self._connections[session_id][registry_key] = conn

        logger.info(
            f"MCP connected: {registry_key} for session {session_id[:8]}… "
            f"({len(tool_names)} tools)"
        )
        return tool_names

    def disconnect(self, session_id: str, registry_key: str) -> bool:
        """
        Disconnect a specific MCP server for a session.

        Returns True if a connection was closed, False if none existed.
        """
        with self._lock:
            conns = self._connections.get(session_id, {})
            conn = conns.pop(registry_key, None)
            if conn:
                # Clean up empty session entry
                if not conns:
                    self._connections.pop(session_id, None)
        if conn:
            # Run disconnect in thread (SyncMCPClient.disconnect uses run_until_complete)
            self._executor.submit(conn.close)
            return True
        return False

    def cleanup_session(self, session_id: str):
        """Kill all MCP subprocesses for a session."""
        with self._lock:
            conns = self._connections.pop(session_id, {})
        # Close in threads to avoid event-loop conflicts
        for conn in conns.values():
            self._executor.submit(conn.close)
        if conns:
            logger.info(
                f"Cleaned up {len(conns)} MCP connection(s) for session {session_id[:8]}…"
            )

    def get_tools(self, session_id: str) -> List[BaseTool]:
        """
        Get all MCP BaseTool instances for a session.

        Returns an empty list if no MCP servers are connected.
        """
        tools: List[BaseTool] = []
        with self._lock:
            conns = self._connections.get(session_id, {})
            for conn in conns.values():
                if conn.server.connected:
                    try:
                        tools.extend(conn.server.get_tools())
                    except Exception as e:
                        logger.warning(
                            f"Failed to get tools from {conn.registry_key}: {e}"
                        )
        return tools

    def get_status(self, session_id: str) -> Dict[str, dict]:
        """
        Get connection status for all MCP servers in a session.

        Returns:
            { registry_key: { connected: bool, tools: [...] } }
        """
        status: Dict[str, dict] = {}
        with self._lock:
            conns = self._connections.get(session_id, {})
            for key, conn in conns.items():
                status[key] = {
                    "connected": conn.server.connected,
                    "tools": conn.tool_names,
                }
        return status

    def is_connected(self, session_id: str, registry_key: str) -> bool:
        """Check if a specific MCP server is connected for a session."""
        with self._lock:
            conns = self._connections.get(session_id, {})
            conn = conns.get(registry_key)
            return conn is not None and conn.server.connected

    @staticmethod
    def get_server_info(registry_key: str) -> dict:
        """Get setup information for a server (for the token modal)."""
        return {
            "env_var": _ENV_VAR_MAP.get(registry_key, ""),
            "setup_url": _SETUP_URLS.get(registry_key, ""),
            "hint": _TOKEN_HINTS.get(registry_key, ""),
        }


# Global singleton
mcp_manager = MCPManager()
