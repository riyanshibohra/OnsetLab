"""
MCP Tool Wrapper for OnsetLab.

Converts MCP tools to BaseTool instances so the agent can use them seamlessly.
"""

from typing import Dict, Any, List, Optional
from ..tools.base import BaseTool
from .client import MCPTool, MCPClient, SyncMCPClient, MCPServerConfig


class MCPToolWrapper(BaseTool):
    """
    Wraps an MCP tool as a BaseTool.
    
    This allows the agent to use MCP tools just like built-in tools.
    """
    
    def __init__(
        self,
        mcp_tool: MCPTool,
        client: SyncMCPClient,
        server_name: str = "",
    ):
        self._mcp_tool = mcp_tool
        self._client = client
        self._server_name = server_name
    
    @property
    def name(self) -> str:
        # Prefix with server name to avoid conflicts
        if self._server_name:
            return f"{self._server_name}_{self._mcp_tool.name}"
        return self._mcp_tool.name
    
    @property
    def description(self) -> str:
        return self._mcp_tool.description
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Convert MCP input schema to our parameter format."""
        schema = self._mcp_tool.input_schema
        
        # MCP uses JSON Schema format
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        params = {}
        for param_name, param_info in properties.items():
            params[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
                "required": param_name in required,
            }
            
            # Include enum if present
            if "enum" in param_info:
                params[param_name]["enum"] = param_info["enum"]
            
            # Include default if present
            if "default" in param_info:
                params[param_name]["default"] = param_info["default"]
        
        return params
    
    def execute(self, **kwargs) -> str:
        """Execute the MCP tool."""
        try:
            result = self._client.call_tool(self._mcp_tool.name, kwargs)
            # Debug: log what we got
            import logging
            logging.getLogger(__name__).debug(f"MCP tool {self._mcp_tool.name} result: {result}")
            if result is None or result == "":
                return "No result returned"
            return str(result)
        except Exception as e:
            import traceback
            return f"Error calling {self._mcp_tool.name}: {e}\n{traceback.format_exc()}"


class MCPToolLoader:
    """
    Loads tools from MCP servers and wraps them as BaseTool instances.
    """
    
    def __init__(self):
        self._clients: Dict[str, SyncMCPClient] = {}
        self._tools: Dict[str, MCPToolWrapper] = {}
    
    def load_server(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
        prefix_tools: bool = True,
    ) -> List[BaseTool]:
        """
        Load tools from an MCP server.
        
        Args:
            name: Server name
            command: Command to run the server
            args: Command arguments
            env: Environment variables
            prefix_tools: Whether to prefix tool names with server name
            
        Returns:
            List of wrapped tools
        """
        config = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
        )
        
        client = SyncMCPClient(config)
        client.connect()
        
        self._clients[name] = client
        
        tools = []
        for mcp_tool in client.tools:
            wrapper = MCPToolWrapper(
                mcp_tool=mcp_tool,
                client=client,
                server_name=name if prefix_tools else "",
            )
            self._tools[wrapper.name] = wrapper
            tools.append(wrapper)
        
        return tools
    
    def load_from_registry(
        self,
        service_id: str,
        env: Dict[str, str] = None,
        extra_args: List[str] = None,
        prefix_tools: bool = True,
    ) -> List[BaseTool]:
        """
        Load tools from an MCP server defined in the registry.
        
        Args:
            service_id: Service ID from registry (e.g., "filesystem")
            env: Environment variables (API keys, etc.)
            extra_args: Extra arguments to pass to the server
            prefix_tools: Whether to prefix tool names with server name
            
        Returns:
            List of wrapped tools
        """
        import json
        from pathlib import Path
        
        # Find registry file
        registry_dir = Path(__file__).parent / "registry"
        config_file = registry_dir / f"{service_id}.json"
        
        if not config_file.exists():
            raise ValueError(f"Unknown service: {service_id}")
        
        config = json.loads(config_file.read_text())
        
        package = config.get("package", {})
        command = package.get("command", "npx")
        args = package.get("args", [])
        
        # Append extra args (e.g., allowed directories for filesystem)
        if extra_args:
            args = args + extra_args
        
        return self.load_server(
            name=service_id,
            command=command,
            args=args,
            env=env or {},
            prefix_tools=prefix_tools,
        )
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a loaded tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all loaded tools."""
        return list(self._tools.values())
    
    def unload_server(self, name: str) -> None:
        """Unload a server and its tools."""
        if name in self._clients:
            self._clients[name].disconnect()
            del self._clients[name]
            
            # Remove tools from this server
            to_remove = [
                tool_name for tool_name, tool in self._tools.items()
                if tool._server_name == name
            ]
            for tool_name in to_remove:
                del self._tools[tool_name]
    
    def unload_all(self) -> None:
        """Unload all servers and tools."""
        for client in self._clients.values():
            client.disconnect()
        self._clients.clear()
        self._tools.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all()
