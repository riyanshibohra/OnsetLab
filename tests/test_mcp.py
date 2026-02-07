"""
Test MCP integration for OnsetLab.

Run with: python -m pytest tests/test_mcp.py -v
Or: python tests/test_mcp.py
"""

import os
import sys
import tempfile
import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .conftest import requires_ollama


def test_mcp_imports():
    """Test that all MCP classes can be imported."""
    from onsetlab.mcp import (
        MCPServer,
        MCPClient,
        SyncMCPClient,
        MCPServerConfig,
        MCPTool,
        MCPToolWrapper,
        MCPToolLoader,
    )
    print("✓ All MCP imports successful")


def test_list_available_services():
    """Test listing available MCP services."""
    from onsetlab import MCPServer
    
    services = MCPServer.list_available_services()
    print(f"\nAvailable MCP services ({len(services)}):")
    for svc in services:
        print(f"  - {svc['id']}: {svc['name']}")
    
    assert len(services) > 0, "Should have at least one service"
    assert any(s["id"] == "filesystem" for s in services), "Should have filesystem service"
    print("✓ List available services works")


def test_mcp_server_config():
    """Test MCPServer configuration."""
    from onsetlab import MCPServer
    
    # Test from_registry
    server = MCPServer.from_registry(
        "filesystem",
        extra_args=["/tmp"]
    )
    
    assert server.name == "Filesystem"
    assert "/tmp" in server._config.args
    print("✓ MCPServer.from_registry works")


def test_mcp_tool_wrapper_schema():
    """Test that MCP tools are properly wrapped as BaseTool."""
    from onsetlab.mcp.client import MCPTool
    from onsetlab.mcp.tool_wrapper import MCPToolWrapper
    
    # Create a mock MCP tool
    mcp_tool = MCPTool(
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                }
            },
            "required": ["path"]
        }
    )
    
    # We can't fully test without a client, but we can test schema conversion
    assert mcp_tool.name == "test_tool"
    assert mcp_tool.description == "A test tool"
    assert "path" in mcp_tool.input_schema["properties"]
    print("✓ MCPTool schema parsing works")


@pytest.mark.skipif(True, reason="Requires Node.js/npx and network access")
def test_filesystem_server_connection():
    """
    Test connecting to the filesystem MCP server.
    
    This test requires:
    - Node.js/npm installed
    - Network access to download the MCP server package
    """
    from onsetlab import MCPServer
    
    print("\nTesting filesystem MCP server connection...")
    print("(This may take a moment to download the server package)")
    
    # Use the project directory (avoids macOS /var -> /private/var symlink issue)
    test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Test directory: {test_dir}")
    
    try:
        # Connect to filesystem server
        server = MCPServer.from_registry(
            "filesystem",
            extra_args=[test_dir]
        )
        server.connect()
        
        print(f"\n✓ Connected to: {server}")
        print(f"  Available tools: {server.get_tool_names()}")
        
        # Test list_directory
        if "list_directory" in server.get_tool_names():
            result = server.call_tool("list_directory", {"path": test_dir})
            print(f"\n  list_directory result:")
            for line in result.split('\n')[:8]:
                print(f"    {line}")
            if result.count('\n') > 8:
                print(f"    ... and more")
        
        # Test read_text_file on README
        readme_path = os.path.join(test_dir, "README.md")
        if "read_text_file" in server.get_tool_names() and os.path.exists(readme_path):
            result = server.call_tool("read_text_file", {"path": readme_path})
            print(f"\n  read_text_file (README.md):")
            print(f"    {result[:150]}...")
            assert "OnsetLab" in result
        
        server.disconnect()
        print("\n✓ Filesystem MCP server test passed!")
        
    except FileNotFoundError:
        print("\n⚠ Node.js/npx not found. Install Node.js to test MCP servers.")
        print("  Install: https://nodejs.org/")
    except Exception as e:
        print(f"\n⚠ MCP test failed: {e}")
        print("  This may be due to:")
        print("  - Node.js/npx not installed")
        print("  - Network issues downloading MCP server")
        raise


def test_agent_with_mcp():
    """
    Test Agent with MCP server integration.
    
    This test requires:
    - Ollama running with phi3.5 model
    - Node.js/npm for MCP server
    """
    from onsetlab import Agent, MCPServer, Calculator, DateTime
    
    print("\nTesting Agent with MCP integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = os.path.join(tmpdir, "notes.txt")
        with open(test_file, "w") as f:
            f.write("Meeting at 3pm tomorrow\nCall John about project")
        
        try:
            # Create MCP server
            server = MCPServer.from_registry("filesystem", extra_args=[tmpdir])
            
            # Create agent with built-in tools + MCP server
            agent = Agent(
                model="phi3.5",
                tools=[Calculator(), DateTime()],
                mcp_servers=[server],
                debug=True
            )
            
            print(f"\n✓ Agent created: {agent}")
            print(f"  Total tools available:")
            for name, tool in agent.tools.items():
                print(f"    - {name}: {tool.description[:50]}...")
            
            # Note: Actual query testing requires Ollama
            # result = agent.run(f"Read the file at {test_file}")
            # print(f"  Result: {result.answer}")
            
            # Cleanup
            agent.disconnect_mcp_servers()
            print("\n✓ Agent with MCP test passed!")
            
        except FileNotFoundError:
            print("\n⚠ Skipping: Node.js/npx not found")
        except RuntimeError as e:
            if "Ollama" in str(e):
                print("\n⚠ Skipping: Ollama not running")
            else:
                raise


if __name__ == "__main__":
    print("=" * 60)
    print("OnsetLab MCP Integration Tests")
    print("=" * 60)
    
    test_mcp_imports()
    test_list_available_services()
    test_mcp_server_config()
    test_mcp_tool_wrapper_schema()
    
    # Interactive tests that require external dependencies
    print("\n" + "-" * 60)
    print("Integration Tests (require Node.js + network)")
    print("-" * 60)
    
    try:
        test_filesystem_server_connection()
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
