"""
Meta-Agent Tools
================
LangChain tools for MCP discovery and web search.
"""

from .tavily_search import (
    tavily_search_tool,
    tavily_search,
    search_for_mcp_server,
)
from .github_tools import (
    fetch_github_readme,
    fetch_github_file,
    fetch_github_file_sync,
    fetch_github_file_async,
    extract_package_json,
    parse_github_url,
)
from .npm_tools import (
    validate_npm_package,
    get_npm_package_info,
    is_mcp_package,
)

__all__ = [
    # Tavily search
    "tavily_search_tool",
    "tavily_search",
    "search_for_mcp_server",
    
    # GitHub tools
    "fetch_github_readme",
    "fetch_github_file",
    "fetch_github_file_sync",
    "fetch_github_file_async",
    "extract_package_json",
    "parse_github_url",
    
    # NPM tools
    "validate_npm_package",
    "get_npm_package_info",
    "is_mcp_package",
]
