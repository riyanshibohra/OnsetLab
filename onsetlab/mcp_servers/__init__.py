"""
MCP Server Schemas for OnsetLab

Comprehensive tool definitions for the 5 core MCP servers:
- Google Calendar
- GitHub  
- Slack
- Gmail
- Notion
"""

from .registry import (
    MCP_SERVERS,
    get_server_config,
    get_all_tools,
    get_tools_for_server,
    GOOGLE_CALENDAR_TOOLS,
    GITHUB_TOOLS,
    SLACK_TOOLS,
    GMAIL_TOOLS,
    NOTION_TOOLS,
)

__all__ = [
    "MCP_SERVERS",
    "get_server_config",
    "get_all_tools",
    "get_tools_for_server",
    "GOOGLE_CALENDAR_TOOLS",
    "GITHUB_TOOLS",
    "SLACK_TOOLS",
    "GMAIL_TOOLS",
    "NOTION_TOOLS",
]
