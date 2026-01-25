"""
Tavily Search Tool
==================
Web search using Tavily API for discovering MCP servers.
"""

import os
from typing import Optional
from langchain_core.tools import tool
from tavily import TavilyClient


def create_tavily_client(api_key: Optional[str] = None) -> TavilyClient:
    """Create a Tavily client with the given or environment API key."""
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY not provided and not found in environment")
    return TavilyClient(api_key=key)


def tavily_search(query: str, api_key: Optional[str] = None, max_results: int = 5) -> str:
    """
    Search the web using Tavily API.
    
    Args:
        query: Search query (e.g., "MCP server for Google Calendar")
        api_key: Tavily API key (optional, uses env var if not provided)
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    client = create_tavily_client(api_key)
    
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
    )
    
    # Format results
    results = []
    
    # Include the AI-generated answer if available
    if response.get("answer"):
        results.append(f"Summary: {response['answer']}\n")
    
    # Format each search result
    for i, result in enumerate(response.get("results", []), 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")[:500]  # Truncate long content
        
        results.append(f"{i}. {title}\n   URL: {url}\n   {content}\n")
    
    return "\n".join(results) if results else "No results found."


@tool
def tavily_search_tool(query: str) -> str:
    """
    Search the web for MCP servers, documentation, and API information.
    
    Use this tool to find:
    - MCP servers for specific services (e.g., "MCP server for Google Calendar")
    - GitHub repositories with MCP implementations
    - API documentation for services
    
    Args:
        query: Search query describing what you're looking for
        
    Returns:
        Search results with titles, URLs, and content snippets
    """
    # Note: When used as a LangChain tool, the api_key needs to be
    # set via environment variable or passed through the node
    return tavily_search(query)


def search_for_mcp_server(service: str, api_key: Optional[str] = None) -> str:
    """
    Specialized search for MCP servers for a given service.
    
    Performs multiple targeted searches prioritizing:
    1. NPM packages with service name
    2. GitHub repos with MCP tools for the service
    3. Community maintained servers
    
    Args:
        service: Service name (e.g., "google_calendar", "slack", "github")
        api_key: Tavily API key
        
    Returns:
        Combined search results with trust indicators
    """
    # Normalize service name for search
    service_name = service.replace("_", " ").title()
    service_lower = service.lower().replace("_", "-")
    
    # Prioritized search queries - from most targeted to broadest
    queries = [
        # 1. NPM packages with exact service name
        f"site:npmjs.com @*/{service_lower}-mcp mcp server",
        
        # 2. NPM packages with service in name
        f"site:npmjs.com mcp server {service_lower} API",
        
        # 3. GitHub repos - service-specific MCP server
        f"site:github.com {service_lower} mcp server API tools",
        
        # 4. GitHub repos - MCP with service name
        f"site:github.com mcp {service_name} server 2024 2025",
        
        # 5. General search
        f"{service_name} MCP model context protocol server npm package github",
    ]
    
    all_results = []
    
    # Add trust level context
    trust_levels = [
        "📦 NPM EXACT MATCH",
        "📦 NPM SEARCH",
        "⭐ GITHUB REPOS",
        "⭐ GITHUB MCP SERVER",
        "🔍 GENERAL SEARCH",
    ]
    
    for i, query in enumerate(queries):
        try:
            # Get more results for each query
            result = tavily_search(query, api_key=api_key, max_results=4)
            trust = trust_levels[i] if i < len(trust_levels) else "🔍 GENERAL"
            all_results.append(f"--- {trust} ---\nSearch: {query}\n{result}")
        except Exception as e:
            all_results.append(f"--- Search: {query} ---\nError: {str(e)}")
    
    return "\n\n".join(all_results)
