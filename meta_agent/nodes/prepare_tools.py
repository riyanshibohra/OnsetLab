"""
Prepare Tools Node
==================
Converts verified servers from discovery into the format expected by filter_tools.
"""

from meta_agent.state import MetaAgentState


def prepare_tools(state: MetaAgentState) -> dict:
    """
    Convert verified_servers into all_tools format for filter_tools.
    
    This bridges the gap between:
    - Discovery output: verified_servers with server configs
    - Filter input: all_tools list with tool schemas
    
    For discovered servers, we generate placeholder tool entries
    since actual tools are discovered at runtime.
    
    Args:
        state: Current MetaAgentState with verified_servers
        
    Returns:
        State update with all_tools and mcp_servers
    """
    verified = state.get("verified_servers", [])
    
    if not verified:
        print("\n⚠️ No verified servers to prepare")
        return {"all_tools": [], "mcp_servers": []}
    
    print(f"\n📦 Preparing {len(verified)} verified servers for filtering...")
    
    all_tools = []
    mcp_servers = []
    registry_services = []
    
    for item in verified:
        service = item.get("service", "unknown")
        server = item.get("server", {})
        verification = item.get("verification", {})
        
        if not verification.get("verified"):
            print(f"   ⚠️ Skipping unverified: {service}")
            continue
        
        # Build MCP server config
        server_config = {
            "name": server.get("name", service),
            "service": service,
            "description": server.get("description", ""),
            "package": None,
            "docker_image": None,
            "command": None,
            "args": [],
            "env_vars": [e.get("name") for e in server.get("env_vars", []) if e.get("name")],
            "transport": server.get("transport", "stdio"),
            "remote_url": server.get("remote_url"),
        }
        
        # Set install info based on type
        install = server.get("install", {})
        if install:
            if install.get("type") == "npm":
                server_config["package"] = install.get("package")
                server_config["command"] = "npx"
                server_config["args"] = ["-y", install.get("package")]
            elif install.get("type") == "docker":
                server_config["docker_image"] = install.get("image")
                server_config["command"] = "docker"
                server_config["args"] = ["run", install.get("image")]
        
        mcp_servers.append(server_config)
        registry_services.append(service)
        
        # Generate tool entries
        # If we extracted tools from README, use those
        extracted = verification.get("extracted_tools", [])
        
        if extracted:
            for tool in extracted:
                all_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": {},  # Will be discovered at runtime
                    "required_params": [],
                    "_service": service,
                    "_discovered": True,  # Mark as discovered (not from curated registry)
                })
        else:
            # No tools extracted - create a placeholder based on service
            # The actual tools will be discovered at runtime when server starts
            all_tools.append({
                "name": f"{service}_tools",
                "description": f"Tools from {server.get('name', service)} - discovered at runtime",
                "parameters": {},
                "required_params": [],
                "_service": service,
                "_discovered": True,
                "_placeholder": True,  # Mark as placeholder
            })
        
        print(f"   ✅ {service}: {len(extracted) or 1} tools")
    
    print(f"\n   📊 Prepared {len(all_tools)} tools from {len(mcp_servers)} servers")
    
    return {
        "all_tools": all_tools,
        "mcp_servers": mcp_servers,
        "registry_services": registry_services,
    }
