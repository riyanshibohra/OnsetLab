"""
Generate Token Guides Node
==========================
Creates step-by-step instructions for obtaining access tokens.
Uses LLM to intelligently filter and deduplicate env vars.
"""

import json
import anthropic

from meta_agent.state import MetaAgentState, TokenGuide


FILTER_ENV_VARS_PROMPT = """You are filtering environment variables for an MCP server.

Given a list of environment variables, keep ALL credentials the user needs to obtain.

## KEEP EVERYTHING EXCEPT:

1. **Transport/Server config** - Remove these:
   - PORT, HOST, URL, MCP_TRANSPORT, NODE_ENV, DEBUG, LOG_*
   
2. **Generic auth without service prefix** - Remove these:
   - AUTH_TOKEN, TOKEN, API_KEY (alone, without service name prefix)
   
3. **Duplicates** - If two vars are the same credential, keep shorter one:
   - *_TOKEN and *_ACCESS_TOKEN → keep *_TOKEN
   - *_KEY and *_API_KEY → keep *_API_KEY

## KEEP ALL OF THESE:
- Any *_TOKEN, *_API_KEY, *_SECRET with service prefix
- Any *_ID (team IDs, workspace IDs, org IDs are often REQUIRED)
- When uncertain, KEEP IT - better to ask user for extra var than miss a required one

## RESPONSE FORMAT (JSON):
{
    "required_credentials": ["VAR1", "VAR2", "VAR3"],
    "removed": [
        {"var": "PORT", "reason": "transport config"},
        {"var": "AUTH_TOKEN", "reason": "generic, no service prefix"}
    ]
}"""


TOKEN_GUIDE_PROMPT = """You are an expert at creating clear, step-by-step authentication setup guides.

Given a service and its required environment variables, create a detailed guide for obtaining each credential.

The guide should be:
1. Clear and actionable (numbered steps)
2. Include specific URLs where possible
3. Cover ALL required environment variables
4. Explain what each env var is for

Respond with a JSON object in this exact format:
{
    "steps": [
        "Step 1: Go to https://...",
        "Step 2: Click on...",
        "Step 3: Create a new...",
        "Step 4: Copy the token and set: export ENV_VAR=your_token"
    ],
    "env_var_notes": {
        "ENV_VAR_1": "What this is and where to get it",
        "ENV_VAR_2": "What this is and where to get it"
    },
    "notes": "Any additional notes or warnings"
}

Be specific to the service - use real URLs and accurate instructions.
Cover ALL environment variables listed, not just the first one."""


def filter_env_vars_with_llm(
    service: str,
    env_vars: list,
    client: anthropic.Anthropic
) -> list:
    """
    Use LLM to intelligently filter and deduplicate env vars.
    
    Args:
        service: Service name
        env_vars: Raw list of env vars from extraction
        client: Anthropic client
        
    Returns:
        Filtered list of actual credential env vars
    """
    if not env_vars:
        return [f"{service.upper()}_TOKEN"]
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"""{FILTER_ENV_VARS_PROMPT}

Service: {service}

Environment Variables Found:
{json.dumps(env_vars, indent=2)}

Filter these to only the required credentials. Respond with JSON."""}
            ],
        )
        
        response_text = response.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        required = result.get("required_credentials", [])
        removed = result.get("removed", [])
        
        if removed:
            removed_names = [r.get("var", r) if isinstance(r, dict) else r for r in removed[:3]]
            print(f"      🗑️ Removed: {', '.join(removed_names)}")
        
        return required if required else [f"{service.upper()}_TOKEN"]
        
    except Exception as e:
        print(f"      ⚠️ LLM filter failed: {e}, using fallback")
        # Fallback: basic filter
        return _basic_filter(env_vars)


def _basic_filter(env_vars: list) -> list:
    """Fallback basic filter if LLM fails."""
    SKIP_PATTERNS = ["_PORT", "_HOST", "_DEBUG", "_LOG", "MCP_TRANSPORT", "NODE_ENV"]
    filtered = [
        ev for ev in env_vars
        if not any(p in ev.upper() for p in SKIP_PATTERNS)
    ]
    return filtered if filtered else env_vars[:1]


def generate_token_guides(state: MetaAgentState) -> dict:
    """
    Generate token setup guides for each MCP server and API server.
    
    Uses LLM to intelligently filter env vars and generate guides.
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with token_guides list AND cleaned env vars for servers
    """
    # Use filtered servers if available, otherwise fall back to unfiltered
    mcp_servers = state.get("filtered_mcp_servers") or state.get("mcp_servers", [])
    api_servers = state.get("filtered_api_servers") or state.get("api_servers", [])
    api_key = state["anthropic_api_key"]
    
    print("\n📚 Generating token setup guides...")
    print("   🧠 Using LLM to filter env vars...")
    
    token_guides = []
    client = anthropic.Anthropic(api_key=api_key)
    
    # Track cleaned servers (with filtered env vars)
    cleaned_mcp_servers = []
    cleaned_api_servers = []
    
    # Process MCP servers
    all_servers = []
    for server in mcp_servers:
        service = server.get("service", "unknown")
        
        # Get all env vars (handle both list and single)
        env_vars = server.get("env_vars", [])
        if not env_vars and server.get("env_var"):
            env_vars = [server.get("env_var")]
        
        print(f"   📦 {service}: {len(env_vars)} env vars found")
        
        # Use LLM to filter to actual credentials only
        credential_vars = filter_env_vars_with_llm(service, env_vars, client)
        
        print(f"      ✅ Keeping: {', '.join(credential_vars)}")
        
        # Create cleaned server copy
        cleaned_server = dict(server)
        cleaned_server["env_vars"] = credential_vars
        cleaned_mcp_servers.append(cleaned_server)
        
        all_servers.append({
            "service": service,
            "auth_type": server.get("auth_type"),
            "package": server.get("package", ""),
            "setup_url": server.get("setup_url", ""),
            "env_vars": credential_vars,
            "source": "mcp"
        })
    
    for api in api_servers:
        service = api.get("service", "unknown")
        
        env_vars = api.get("env_vars", [])
        if not env_vars and api.get("env_var"):
            env_vars = [api.get("env_var")]
        
        print(f"   📦 {service}: {len(env_vars)} env vars found")
        
        credential_vars = filter_env_vars_with_llm(service, env_vars, client)
        
        print(f"      ✅ Keeping: {', '.join(credential_vars)}")
        
        # Create cleaned API copy
        cleaned_api = dict(api)
        cleaned_api["env_vars"] = credential_vars
        cleaned_api_servers.append(cleaned_api)
        
        all_servers.append({
            "service": service,
            "auth_type": api.get("auth_type"),
            "package": "",
            "setup_url": api.get("api_docs_url", ""),
            "env_vars": credential_vars,
            "source": "api"
        })
    
    for server in all_servers:
        service = server.get("service", "unknown")
        auth_type = server.get("auth_type", "unknown")
        package = server.get("package", "")
        setup_url = server.get("setup_url", "")
        source = server.get("source", "mcp")
        env_vars = server.get("env_vars", [])
        
        source_label = "MCP" if source == "mcp" else "API"
        env_vars_str = ", ".join(env_vars) if env_vars else "auto-detect"
        print(f"   📖 Creating guide for: {service} ({auth_type}) [{source_label}]")
        print(f"      📋 Env vars: {env_vars_str}")
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": f"""{TOKEN_GUIDE_PROMPT}

Service: {service}
Package: {package}
Auth Type: {auth_type}
Setup URL: {setup_url}
Required Environment Variables: {env_vars if env_vars else ['Detect based on service']}

Create a step-by-step guide for obtaining ALL the required credentials.
Make sure to explain how to get EACH environment variable listed.
Respond with JSON."""}
                ],
            )
            
            response_text = response.content[0].text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            
            steps = result.get("steps", [])
            env_var_notes = result.get("env_var_notes", {})
            
            # Use provided env_vars or fall back to detected
            final_env_vars = env_vars if env_vars else [f"{service.upper()}_TOKEN"]
            
            guide: TokenGuide = {
                "service": service,
                "auth_type": auth_type,
                "steps": steps,
                "env_var": final_env_vars[0] if final_env_vars else f"{service.upper()}_TOKEN",
                "env_vars": final_env_vars,  # All required env vars
            }
            
            token_guides.append(guide)
            print(f"      ✅ Generated {len(steps)} steps for {len(final_env_vars)} env var(s)")
            
        except Exception as e:
            print(f"      ❌ Failed to generate guide: {e}")
            
            # Create a basic fallback guide
            fallback_env_vars = env_vars if env_vars else [f"{service.upper()}_TOKEN"]
            fallback_guide: TokenGuide = {
                "service": service,
                "auth_type": auth_type,
                "steps": [
                    f"1. Visit the {service.replace('_', ' ').title()} developer portal",
                    "2. Create a new application or integration",
                    "3. Generate API credentials (token/key)",
                ] + [f"{i+4}. Set environment variable: export {ev}=your_value" for i, ev in enumerate(fallback_env_vars)],
                "env_var": fallback_env_vars[0],
                "env_vars": fallback_env_vars,
            }
            token_guides.append(fallback_guide)
    
    print(f"\n   ✅ Generated {len(token_guides)} token guides")
    
    # Return both token guides AND cleaned servers with filtered env vars
    return {
        "token_guides": token_guides,
        # Override filtered servers with cleaned env vars
        "filtered_mcp_servers": cleaned_mcp_servers,
        "filtered_api_servers": cleaned_api_servers,
    }
