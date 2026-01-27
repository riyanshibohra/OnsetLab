"""
Generate Token Guides Node
==========================
Creates step-by-step instructions for obtaining access tokens.
Reads pre-written guides from registry files - no LLM needed.
"""

from meta_agent.state import MetaAgentState, TokenGuide


def generate_token_guides(state: MetaAgentState) -> dict:
    """
    Generate token setup guides for each MCP server.
    
    Reads pre-written guides from registry files (auth.guide).
    No LLM calls - guides are pre-written for each service.
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with token_guides list (for UI display)
    """
    # Use filtered servers if available, otherwise fall back to unfiltered
    mcp_servers = state.get("filtered_mcp_servers") or state.get("mcp_servers", [])
    
    print("\n📚 Loading token setup guides from registry...")
    
    token_guides = []
    
    for server in mcp_servers:
        service = server.get("service") or server.get("service_id") or server.get("name", "unknown").lower()
        auth = server.get("auth", {})
        
        auth_type = auth.get("type", "unknown")
        env_vars = auth.get("env_vars", [])
        setup_url = auth.get("setup_url", "")
        guide_data = auth.get("guide", {})
        
        # Skip if no auth required
        if auth_type == "none" and not env_vars:
            print(f"   ⏭️ {service}: No auth required")
            continue
        
        # Get steps from pre-written guide or create fallback
        steps = guide_data.get("steps", [])
        notes = guide_data.get("notes", "")
        
        if not steps:
            # Fallback if guide not in registry
            steps = [
                f"Visit {setup_url or 'the service developer portal'}",
                "Create a new application or integration",
                "Generate API credentials (token/key)",
            ]
            for ev in env_vars:
                steps.append(f"Set environment variable: {ev}")
        
        guide: TokenGuide = {
            "service": service,
            "auth_type": auth_type,
            "steps": steps,
            "env_var": env_vars[0] if env_vars else f"{service.upper()}_TOKEN",
            "env_vars": env_vars,
            "setup_url": setup_url,
            "notes": notes,
        }
        
        token_guides.append(guide)
        print(f"   ✅ {service}: {len(steps)} steps loaded")
    
    print(f"\n   📦 Loaded {len(token_guides)} token guides")
    
    return {
        "token_guides": token_guides,
    }
