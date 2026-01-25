"""
Compile Results Node
====================
Aggregates all discovered MCP servers and API fallbacks.
"""

from meta_agent.state import MetaAgentState


def compile_results(state: MetaAgentState) -> dict:
    """
    Compile and deduplicate all discovery results.
    
    Steps:
    1. Deduplicate tool schemas by name
    2. Validate all MCP server configs
    3. Print summary of discoveries
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update (mainly cleanup/validation)
    """
    mcp_servers = state.get("mcp_servers", [])
    api_servers = state.get("api_servers", [])
    tool_schemas = state.get("tool_schemas", [])
    identified_services = state.get("identified_services", [])
    
    print("\n" + "=" * 60)
    print("📊 DISCOVERY RESULTS")
    print("=" * 60)
    
    # Summary stats
    print(f"\n🎯 Services requested: {len(identified_services)}")
    print(f"   {', '.join(identified_services)}")
    
    print(f"\n✅ MCP Servers found: {len(mcp_servers)}")
    for server in mcp_servers:
        tool_count = len(server.get("tools", []))
        print(f"   • {server.get('service')}: {server.get('package')} ({tool_count} tools)")
    
    print(f"\n🔧 API Fallbacks: {len(api_servers)}")
    for api in api_servers:
        print(f"   • {api.get('service')}: {api.get('reason', 'No MCP found')[:50]}")
    
    # Deduplicate tool schemas by name
    seen_tools = set()
    unique_tools = []
    for tool in tool_schemas:
        tool_name = tool.get("name")
        if tool_name and tool_name not in seen_tools:
            seen_tools.add(tool_name)
            unique_tools.append(tool)
    
    print(f"\n🔧 Total unique tools: {len(unique_tools)}")
    
    # List tools by category
    if unique_tools:
        print("   Tools discovered:")
        for tool in unique_tools[:10]:  # Show first 10
            desc = tool.get("description", "")[:40]
            print(f"   • {tool.get('name')}: {desc}")
        if len(unique_tools) > 10:
            print(f"   ... and {len(unique_tools) - 10} more")
    
    # Check for any errors
    errors = state.get("errors", [])
    if errors:
        print(f"\n⚠️ Errors encountered: {len(errors)}")
        for error in errors[:5]:
            print(f"   • {error}")
    
    print("\n" + "=" * 60)
    
    # Return deduplicated tool schemas
    # Note: We return as a replacement, not addition (don't use reducer)
    return {
        # Clear temporary evaluation fields
        "_eval_package_name": None,
        "_eval_github_url": None,
        "_eval_auth_type": None,
        "_eval_confidence": None,
        "_eval_reasoning": None,
        "current_service": None,
        "search_results": None,
        "result_quality": None,
    }
