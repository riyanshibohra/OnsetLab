"""
MCP Client for OnsetLab.

Connects to MCP servers via STDIO transport and enables tool discovery and execution.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30


class MCPClient:
    """
    Client for connecting to MCP servers via STDIO.
    
    MCP uses JSON-RPC 2.0 protocol over stdin/stdout.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._connected = False
        self._tools: Dict[str, MCPTool] = {}
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    @property
    def tools(self) -> List[MCPTool]:
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        return self._tools.get(name)
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return
        
        # Prepare environment
        env = dict(os.environ)
        env.update(self.config.env)
        
        try:
            # Start the MCP server process
            # Use larger buffer limit for servers that send big responses (like GitHub)
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=10 * 1024 * 1024,  # 10MB buffer limit
            )
            
            logger.info(f"Started MCP server: {self.config.name}")
            
            # Initialize the connection (MCP handshake)
            await self._initialize()
            
            # Discover tools
            await self._discover_tools()
            
            self._connected = True
            logger.info(f"Connected to MCP server '{self.config.name}' with {len(self._tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning(f"Error stopping MCP server: {e}")
            finally:
                self._process = None
        
        self._connected = False
        self._tools.clear()
        logger.info(f"Disconnected from MCP server: {self.config.name}")
    
    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Not connected to MCP server")
        
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        # Send request
        async with self._write_lock:
            data = json.dumps(request) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()
        
        # Read response - skip any notifications until we get our response
        async with self._read_lock:
            start_time = asyncio.get_event_loop().time()
            
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining_timeout = self.config.timeout - elapsed
                
                if remaining_timeout <= 0:
                    raise TimeoutError(f"MCP server did not respond within {self.config.timeout}s")
                
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=remaining_timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"MCP server did not respond within {self.config.timeout}s")
                except asyncio.LimitOverrunError as e:
                    logger.warning(f"Response exceeded buffer limit: {e}")
                    raise RuntimeError(f"MCP server response too large: {e}")
                
                if not line:
                    raise ConnectionError("MCP server closed connection")
                
                try:
                    response = json.loads(line.decode())
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from MCP server: {line[:500]}")
                    raise RuntimeError(f"Invalid JSON from MCP server: {e}")
                
                # Check if this is a notification (no id field) - skip it
                if "id" not in response:
                    logger.debug(f"Skipping notification: {response.get('method', 'unknown')}")
                    continue
                
                # Check if this response matches our request
                if response.get("id") == self._request_id:
                    break
                else:
                    # Response for different request? Log and continue
                    logger.warning(f"Got response for different request id: {response.get('id')} (expected {self._request_id})")
                    continue
        
        # Check for errors
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"MCP error: {error.get('message', 'Unknown error')}")
        
        return response.get("result", {})
    
    async def _initialize(self) -> None:
        """Perform MCP initialization handshake."""
        try:
            result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "clientInfo": {
                    "name": "onsetlab",
                    "version": "1.0.0",
                }
            })
            
            logger.debug(f"MCP server capabilities: {result.get('capabilities', {})}")
            
            # Send initialized notification
            if self._process and self._process.stdin:
                notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
                data = json.dumps(notification) + "\n"
                self._process.stdin.write(data.encode())
                await self._process.stdin.drain()
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            # Check stderr for server errors
            if self._process and self._process.stderr:
                try:
                    stderr_data = await asyncio.wait_for(
                        self._process.stderr.read(1000),
                        timeout=1
                    )
                    if stderr_data:
                        logger.error(f"Server stderr: {stderr_data.decode()}")
                except asyncio.TimeoutError:
                    pass
            raise
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        try:
            result = await self._send_request("tools/list")
            
            tools_list = result.get("tools", [])
            if not tools_list:
                logger.warning(f"No tools returned from server. Result: {result}")
            
            for tool_data in tools_list:
                tool = MCPTool.from_dict(tool_data)
                self._tools[tool.name] = tool
                logger.debug(f"Discovered tool: {tool.name}")
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")
        
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        
        logger.debug(f"Tool '{tool_name}' result: {result}")
        
        # Extract content from result
        content = result.get("content", [])
        if content:
            # MCP returns content as a list of content blocks
            texts = []
            for block in content:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    texts.append(f"[Image: {block.get('mimeType', 'unknown')}]")
                elif block.get("type") == "resource":
                    # Handle resource blocks
                    texts.append(f"[Resource: {block.get('resource', {}).get('uri', 'unknown')}]")
                else:
                    # Unknown block type - include as JSON
                    texts.append(json.dumps(block))
            return "\n".join(texts) if texts else str(result)
        
        # No content - return the full result as string
        if result:
            return json.dumps(result, indent=2)
        return "No result returned"
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# Synchronous wrapper for non-async code
class SyncMCPClient:
    """
    Synchronous wrapper for MCPClient.
    
    Provides a simple sync interface for environments without async support.
    """
    
    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._client: Optional[MCPClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def connect(self) -> None:
        """Connect to the MCP server."""
        self._loop = asyncio.new_event_loop()
        self._client = MCPClient(self._config)
        self._loop.run_until_complete(self._client.connect())
    
    def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._client and self._loop:
            self._loop.run_until_complete(self._client.disconnect())
        if self._loop:
            self._loop.close()
            self._loop = None
        self._client = None
    
    @property
    def connected(self) -> bool:
        return self._client.connected if self._client else False
    
    @property
    def tools(self) -> List[MCPTool]:
        return self._client.tools if self._client else []
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        return self._client.get_tool(name) if self._client else None
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self._client or not self._loop:
            raise RuntimeError("Not connected")
        return self._loop.run_until_complete(
            self._client.call_tool(tool_name, arguments)
        )
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
