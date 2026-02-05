"""
MCP Server configuration and management for OnsetLab.

Provides easy-to-use interface for connecting to MCP servers.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .client import MCPServerConfig, MCPClient, SyncMCPClient, MCPTool
from .tool_wrapper import MCPToolWrapper, MCPToolLoader


class MCPServer:
    """
    MCP Server connection manager.
    
    Provides a simple interface to connect to MCP servers and use their tools.
    
    Example:
        # Connect to filesystem server
        server = MCPServer.from_registry("filesystem", extra_args=["/path/to/dir"])
        server.connect()
        
        # Get tools
        tools = server.get_tools()
        
        # Call a tool
        result = server.call_tool("list_directory", {"path": "/path/to/dir"})
        
        # Disconnect when done
        server.disconnect()
        
        # Or use context manager
        with MCPServer.from_registry("filesystem", extra_args=["/"]) as server:
            result = server.call_tool("list_directory", {"path": "/"})
    """
    
    def __init__(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
        timeout: int = 30,
    ):
        """
        Configure an MCP server.
        
        Args:
            name: Server name.
            command: Command to run (npx, docker, python, etc.).
            args: Command arguments.
            env: Environment variables (API keys, etc.).
            timeout: Request timeout in seconds.
        """
        self.name = name
        self._config = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            timeout=timeout,
        )
        self._client: Optional[SyncMCPClient] = None
        self._tools: Dict[str, MCPToolWrapper] = {}
    
    @classmethod
    def from_registry(
        cls,
        service_id: str,
        env: Dict[str, str] = None,
        extra_args: List[str] = None,
        timeout: int = 30,
    ) -> "MCPServer":
        """
        Load MCP server config from registry.
        
        Args:
            service_id: Service ID (e.g., "github", "slack", "filesystem").
            env: Environment variables to override.
            extra_args: Extra arguments (e.g., allowed directories for filesystem).
            timeout: Request timeout in seconds.
            
        Returns:
            Configured MCPServer.
        """
        # Find registry file
        registry_dir = Path(__file__).parent / "registry"
        config_file = registry_dir / f"{service_id}.json"
        
        if not config_file.exists():
            available = [f.stem for f in registry_dir.glob("*.json")]
            raise ValueError(
                f"Unknown service: {service_id}. "
                f"Available services: {', '.join(available)}"
            )
        
        config = json.loads(config_file.read_text())
        
        package = config.get("package", {})
        command = package.get("command", "npx")
        args = package.get("args", [])
        
        # Append extra args
        if extra_args:
            args = args + extra_args
        
        return cls(
            name=config.get("name", service_id),
            command=command,
            args=args,
            env=env or {},
            timeout=timeout,
        )
    
    @classmethod
    def list_available_services(cls) -> List[Dict[str, str]]:
        """
        List available MCP services from registry.
        
        Returns:
            List of service info dicts with 'id', 'name', 'description'.
        """
        registry_dir = Path(__file__).parent / "registry"
        services = []
        
        for config_file in registry_dir.glob("*.json"):
            config = json.loads(config_file.read_text())
            services.append({
                "id": config_file.stem,
                "name": config.get("name", config_file.stem),
                "description": config.get("description", ""),
            })
        
        return sorted(services, key=lambda x: x["name"])
    
    def connect(self) -> None:
        """Connect to the MCP server."""
        if self._client and self._client.connected:
            return
        
        self._client = SyncMCPClient(self._config)
        
        try:
            self._client.connect()
        except Exception as e:
            print(f"[MCP ERROR] Failed to connect to {self.name}: {e}")
            raise
        
        # Wrap discovered tools
        self._tools.clear()
        tool_count = len(self._client.tools) if self._client.tools else 0
        
        if tool_count == 0:
            print(f"[MCP WARNING] No tools discovered from {self.name}. Check your token/credentials.")
        
        for mcp_tool in self._client.tools:
            wrapper = MCPToolWrapper(
                mcp_tool=mcp_tool,
                client=self._client,
                server_name="",  # No prefix for single server
            )
            self._tools[mcp_tool.name] = wrapper
    
    def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._client:
            self._client.disconnect()
            self._client = None
        self._tools.clear()
    
    @property
    def connected(self) -> bool:
        """Check if connected to the server."""
        return self._client is not None and self._client.connected
    
    def get_tools(self) -> List[MCPToolWrapper]:
        """
        Get available tools as BaseTool instances.
        
        Returns:
            List of wrapped tools that can be used with the agent.
        """
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[MCPToolWrapper]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
        return list(self._tools.keys())
    
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call.
            params: Tool parameters.
            
        Returns:
            Tool result as string.
        """
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        return tool.execute(**params)
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        tool_count = len(self._tools) if self.connected else 0
        return f"<MCPServer: {self.name} ({status}, {tool_count} tools)>"
