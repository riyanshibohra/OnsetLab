"""
Load Registry Node
==================
Loads tools from registry JSON files for selected services.
"""

import json
import os
from pathlib import Path
from meta_agent.state import MetaAgentState


def load_registry(state: MetaAgentState) -> dict:
    """
    Load tools from registry files for selected services.
    
    Always includes memory (built-in), plus any services selected
    by the user in the UI.
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with:
        - all_tools: List of all tool dicts from registry
        - mcp_servers: List of MCP server configs
        - registry_services: Services successfully loaded
        - errors: Any services not found in registry
    """
    identified_services = state.get("identified_services", [])
    
    print(f"\n📚 Loading registry for {len(identified_services)} services...")
    
    # Get registry directory
    registry_dir = Path(__file__).parent.parent / "registry"
    
    all_tools = []
    mcp_servers = []
    registry_services = []
    errors = []
    
    # Always load memory (built-in)
    memory_path = registry_dir / "_builtin_memory.json"
    if memory_path.exists():
        with open(memory_path, 'r') as f:
            memory_data = json.load(f)
            # Add service context to memory tools
            for tool in memory_data["tools"]:
                tool["_service"] = "memory"
                tool["_builtin"] = True  # Mark as built-in (always include)
            all_tools.extend(memory_data["tools"])
            print(f"   ✅ Memory (built-in): {len(memory_data['tools'])} tools")
            registry_services.append("memory")
    else:
        errors.append("Built-in memory registry not found!")
    
    # Load each identified service
    for service in identified_services:
        registry_path = registry_dir / f"{service}.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                service_data = json.load(f)
                tools = service_data.get("tools", [])
                
                # Add service context to each tool
                for tool in tools:
                    tool["_service"] = service
                
                all_tools.extend(tools)
                mcp_servers.append(service_data)
                registry_services.append(service)
                
                print(f"   ✅ {service.title()}: {len(tools)} tools")
        else:
            error_msg = f"Service '{service}' not found in registry"
            errors.append(error_msg)
            print(f"   ❌ {service.title()}: Not in registry")
    
    print(f"\n   📦 Loaded {len(all_tools)} total tools from {len(registry_services)} services")
    
    return {
        "all_tools": all_tools,
        "mcp_servers": mcp_servers,
        "registry_services": registry_services,
        "errors": errors,
    }
