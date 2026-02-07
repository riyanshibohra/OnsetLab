#!/usr/bin/env python3
"""Test MCP servers."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onsetlab import Agent, MCPServer
from .conftest import requires_ollama

pytestmark = requires_ollama

# Server configs
SERVERS = {
    "filesystem": {
        "env_vars": [],
        "setup": lambda: MCPServer(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()]
        ),
        "test_query": "List files in current directory",
    },
    "github": {
        "env_vars": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        "setup": lambda: MCPServer(
            name="github",
            command="docker",
            args=["run", "--pull", "always", "-i", "--rm",
                  "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                  "ghcr.io/github/github-mcp-server:latest"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")}
        ),
        "test_query": "Who am I on GitHub?",
    },
    "slack": {
        "env_vars": ["SLACK_MCP_XOXP_TOKEN"],
        "setup": lambda: MCPServer(
            name="slack",
            command="npx",
            args=["-y", "slack-mcp-server", "--transport", "stdio"],
            env={"SLACK_MCP_XOXP_TOKEN": os.environ.get("SLACK_MCP_XOXP_TOKEN", "")}
        ),
        "test_query": "List my Slack channels",
    },
    "notion": {
        "env_vars": ["NOTION_TOKEN"],
        "setup": lambda: MCPServer(
            name="notion",
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={"NOTION_TOKEN": os.environ.get("NOTION_TOKEN", "")}
        ),
        "test_query": "Get my Notion workspace info",
    },
    "tavily": {
        "env_vars": ["TAVILY_API_KEY"],
        "setup": lambda: MCPServer(
            name="tavily",
            command="npx",
            args=["-y", "tavily-mcp@latest"],
            env={"TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", "")}
        ),
        "test_query": "Search for OpenAI",
    },
}


def _run_server_test(service_id, config):
    """Test a single MCP server (called from main(), not discovered by pytest)."""
    print(f"\n{'='*50}")
    print(f"Testing: {service_id.upper()}")
    print("="*50)
    
    # Check env vars
    missing = [v for v in config["env_vars"] if not os.environ.get(v)]
    if missing:
        print(f"[SKIP] Missing: {', '.join(missing)}")
        return "skipped"
    
    agent = Agent("phi3.5", debug=True)
    
    try:
        server = config["setup"]()
        agent.add_mcp_server(server)
        
        tool_count = len(agent._planner.tools)
        print(f"[OK] {tool_count} tools loaded")
        
        # Show tools
        for name in list(agent._planner.tools.keys())[:5]:
            print(f"  - {name}")
        if tool_count > 5:
            print(f"  ... and {tool_count - 5} more")
        
        # Test
        print(f"\nTest: {config['test_query']}")
        result = agent.run(config["test_query"])
        print(f"Answer: {result.answer}")
        
        return "success"
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return "failed"
    finally:
        agent.disconnect_mcp_servers()


def main():
    print("OnsetLab MCP Test")
    print("="*50)
    
    # Which server to test
    test_only = ["notion"]  # Change this to test different servers
    
    for service_id, config in SERVERS.items():
        if test_only and service_id not in test_only:
            continue
        _run_server_test(service_id, config)


if __name__ == "__main__":
    main()
