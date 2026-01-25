"""
Search MCP Servers Node
=======================
Uses Tavily to search for MCP servers for the current service.
"""

from meta_agent.state import MetaAgentState
from meta_agent.tools.tavily_search import search_for_mcp_server


def search_mcp_servers(state: MetaAgentState) -> dict:
    """
    Search for MCP servers for the current service.
    
    Uses Tavily to search:
    - "MCP server {service} modelcontextprotocol GitHub"
    - "{service} MCP model context protocol npm package"
    - "awesome-mcp {service} server"
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with search_results and current_service
    """
    identified_services = state["identified_services"]
    current_index = state["current_service_index"]
    tavily_api_key = state["tavily_api_key"]
    
    # Check if we have services to process
    if not identified_services or current_index >= len(identified_services):
        print("⚠️ No more services to search for")
        return {
            "current_service": None,
            "search_results": None,
        }
    
    # Get current service
    current_service = identified_services[current_index]
    print(f"\n🔍 Searching MCP servers for: {current_service} ({current_index + 1}/{len(identified_services)})")
    
    try:
        # Use our specialized MCP search function
        search_results = search_for_mcp_server(
            service=current_service,
            api_key=tavily_api_key
        )
        
        # Log search results for debugging
        print(f"   Found results ({len(search_results)} chars)")
        
        # Show key URLs found in results
        import re
        github_urls = re.findall(r'https?://github\.com/[^\s\)\"\']+', search_results)
        npm_urls = re.findall(r'https?://(?:www\.)?npmjs\.com/[^\s\)\"\']+', search_results)
        
        if github_urls:
            print(f"   📦 GitHub repos found:")
            # Deduplicate and show first 5
            seen = set()
            for url in github_urls[:10]:
                clean_url = url.rstrip('.,;:')
                if clean_url not in seen:
                    seen.add(clean_url)
                    # Warn if archived
                    if 'archived' in clean_url.lower():
                        print(f"      ⚠️ {clean_url} (ARCHIVED!)")
                    else:
                        print(f"      • {clean_url}")
                if len(seen) >= 5:
                    break
        
        if npm_urls:
            print(f"   📋 NPM packages found:")
            seen = set()
            for url in npm_urls[:5]:
                clean_url = url.rstrip('.,;:')
                if clean_url not in seen:
                    seen.add(clean_url)
                    print(f"      • {clean_url}")
        
        return {
            "current_service": current_service,
            "search_results": search_results,
        }
        
    except Exception as e:
        print(f"❌ Search failed for {current_service}: {e}")
        return {
            "current_service": current_service,
            "search_results": f"Search failed: {str(e)}",
            "errors": [f"Search failed for {current_service}: {e}"],
        }
