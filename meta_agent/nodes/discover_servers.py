"""
MCP Server Discovery
====================
Discovers and fetches MCP servers from the official registry and other sources.
"""

import json
import re
import httpx
from typing import Optional
from meta_agent.state import MetaAgentState


# =============================================================================
# Registry API Configuration
# =============================================================================

MCP_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"

# Service name aliases (user might say "calendar" but server is "google-calendar")
SERVICE_ALIASES = {
    "calendar": ["google-calendar", "google_calendar", "gcal", "calendar"],
    "github": ["github", "gh"],
    "slack": ["slack"],
    "notion": ["notion"],
    "discord": ["discord"],
    "linear": ["linear"],
    "postgres": ["postgres", "postgresql", "pg"],
    "mysql": ["mysql"],
    "sqlite": ["sqlite"],
    "filesystem": ["filesystem", "fs", "file"],
    "brave": ["brave", "brave-search"],
    "tavily": ["tavily"],
    "google-drive": ["google-drive", "gdrive", "drive"],
    "gmail": ["gmail", "google-mail"],
    "twitter": ["twitter", "x"],
    "reddit": ["reddit"],
    "jira": ["jira", "atlassian"],
    "confluence": ["confluence"],
    "airtable": ["airtable"],
    "asana": ["asana"],
    "trello": ["trello"],
    "todoist": ["todoist"],
    "stripe": ["stripe"],
    "shopify": ["shopify"],
    "salesforce": ["salesforce", "sfdc"],
    "hubspot": ["hubspot"],
    "zendesk": ["zendesk"],
    "intercom": ["intercom"],
}


# =============================================================================
# Registry Fetching
# =============================================================================

def fetch_all_servers() -> list[dict]:
    """
    Fetch all servers from the official MCP Registry.
    
    Returns list of server objects with full metadata.
    """
    all_servers = []
    cursor = None
    
    try:
        with httpx.Client(timeout=30.0) as client:
            while True:
                url = MCP_REGISTRY_URL
                if cursor:
                    url += f"?cursor={cursor}"
                
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                
                servers = data.get("servers", [])
                all_servers.extend(servers)
                
                # Check for pagination
                metadata = data.get("metadata", {})
                cursor = metadata.get("nextCursor")
                
                if not cursor or len(servers) == 0:
                    break
                    
        print(f"   📡 Fetched {len(all_servers)} servers from MCP Registry")
        return all_servers
        
    except Exception as e:
        print(f"   ⚠️ Failed to fetch from MCP Registry: {e}")
        return []


def search_registry_for_service(service_name: str, all_servers: list[dict]) -> list[dict]:
    """
    Search fetched servers for a specific service.
    
    Args:
        service_name: The service to search for (e.g., "slack", "github")
        all_servers: List of all servers from registry
        
    Returns:
        List of matching server objects, sorted by relevance
    """
    service_lower = service_name.lower().strip()
    
    # Get aliases for this service
    search_terms = SERVICE_ALIASES.get(service_lower, [service_lower])
    if service_lower not in search_terms:
        search_terms.append(service_lower)
    
    matches = []
    
    for server_data in all_servers:
        server = server_data.get("server", {})
        meta = server_data.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {})
        
        # Skip non-active servers
        if meta.get("status") != "active":
            continue
        
        name = server.get("name", "").lower()
        description = server.get("description", "").lower()
        repo_url = server.get("repository", {}).get("url", "").lower() if isinstance(server.get("repository"), dict) else ""
        
        # Score based on how well it matches
        score = 0
        matched_term = None
        
        for term in search_terms:
            # Exact service name at end of server name (highest priority)
            # e.g., "app.linear/linear" for "linear", "com.github/github" for "github"
            if name.endswith(f"/{term}") or name.endswith(f"/{term}-mcp") or name.endswith(f"/{term}-server"):
                score += 200
                matched_term = term
            # Service name as main identifier (second priority)
            elif f"/{term}" in name or f"-{term}" in name:
                score += 150
                matched_term = term
            # Service name in the path (official repos often have service name)
            elif term in name and "io.github" not in name:  # Avoid random github.io repos
                score += 100
                matched_term = term
            # Description mentions service specifically
            elif f" {term} " in f" {description} " or description.startswith(term) or f"for {term}" in description:
                score += 50
                matched_term = term
        
        # Bonus for official-looking servers
        if score > 0:
            # Check if from official org (e.g., "app.linear", "com.slack")
            if any(name.startswith(prefix) for prefix in ["app.", "com.", "io."]) and matched_term in name:
                score += 75
            
            # Check if repo is from the service's own org
            if matched_term and repo_url:
                if f"github.com/{matched_term}" in repo_url:
                    score += 100  # Official repo!
            
            # Penalty for generic github.io repos matching accidentally
            if "io.github" in name and matched_term not in name.split("/")[-1]:
                score -= 50
            
            matches.append({
                "server": server,
                "meta": meta,
                "score": score
            })
    
    # Sort by score (highest first), then by most recently updated
    matches.sort(key=lambda x: (x["score"], x["meta"].get("updatedAt", "")), reverse=True)
    
    return matches


# =============================================================================
# Server Data Extraction
# =============================================================================

def extract_server_config(server_data: dict) -> dict:
    """
    Extract useful configuration from a registry server entry.
    
    Returns a normalized config dict with install info, env vars, etc.
    """
    server = server_data.get("server", {})
    meta = server_data.get("meta", {})
    
    config = {
        "name": server.get("name", ""),
        "description": server.get("description", ""),
        "version": server.get("version", ""),
        "repository": server.get("repository", {}).get("url", ""),
        "website": server.get("websiteUrl", ""),
        "status": meta.get("status", "unknown"),
        "updated_at": meta.get("updatedAt", ""),
        "install": None,
        "env_vars": [],
        "transport": None,
        "remote_url": None,
    }
    
    # Extract package/install info
    packages = server.get("packages", [])
    if packages:
        pkg = packages[0]  # Use first package
        registry_type = pkg.get("registryType", "")
        identifier = pkg.get("identifier", "")
        
        if registry_type == "npm":
            config["install"] = {
                "type": "npm",
                "package": identifier,
                "command": f"npx -y {identifier}"
            }
        elif registry_type == "oci":
            config["install"] = {
                "type": "docker",
                "image": identifier,
                "command": f"docker run {identifier}"
            }
        
        # Extract transport
        transport = pkg.get("transport", {})
        config["transport"] = transport.get("type", "stdio")
        
        # Extract env vars
        env_vars = pkg.get("environmentVariables", [])
        for env in env_vars:
            config["env_vars"].append({
                "name": env.get("name", ""),
                "description": env.get("description", ""),
                "required": env.get("isRequired", False),
                "secret": env.get("isSecret", False)
            })
    
    # Extract remote URL if available (for hosted servers)
    remotes = server.get("remotes", [])
    if remotes:
        remote = remotes[0]
        config["remote_url"] = remote.get("url", "")
        if not config["transport"]:
            config["transport"] = remote.get("type", "")
    
    return config


# =============================================================================
# Main Discovery Node
# =============================================================================

def discover_servers(state: MetaAgentState) -> dict:
    """
    LangGraph node: Discover MCP servers for identified services.
    
    Takes identified_services from state, searches the MCP Registry,
    and returns discovered server configurations.
    
    Args:
        state: Current MetaAgentState with identified_services
        
    Returns:
        State update with discovered_servers list
    """
    identified_services = state.get("identified_services", [])
    
    if not identified_services:
        print("\n⚠️ No services identified, skipping discovery")
        return {"discovered_servers": [], "discovery_errors": ["No services identified"]}
    
    print(f"\n🔍 Discovering MCP servers for {len(identified_services)} services...")
    
    # Fetch all servers from registry once
    all_servers = fetch_all_servers()
    
    if not all_servers:
        return {
            "discovered_servers": [],
            "discovery_errors": ["Failed to fetch from MCP Registry"]
        }
    
    discovered = []
    errors = []
    
    for service in identified_services:
        print(f"\n   🔎 Searching for: {service}")
        
        # Search registry
        matches = search_registry_for_service(service, all_servers)
        
        if matches:
            # Take the best match
            best_match = matches[0]
            config = extract_server_config(best_match)
            
            print(f"      ✅ Found: {config['name']} (v{config['version']})")
            if config.get("repository"):
                print(f"         📦 {config['repository']}")
            
            discovered.append({
                "service": service,
                "server": config,
                "alternatives": [extract_server_config(m) for m in matches[1:3]]  # Include top 2 alternatives
            })
        else:
            print(f"      ❌ No MCP server found for: {service}")
            errors.append(f"No MCP server found for: {service}")
    
    print(f"\n   📊 Discovered {len(discovered)}/{len(identified_services)} services")
    
    return {
        "discovered_servers": discovered,
        "discovery_errors": errors
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_popular_servers(all_servers: list[dict], limit: int = 20) -> list[dict]:
    """
    Get a list of popular/well-known servers from the registry.
    
    Filters for servers that are likely to be well-maintained and useful.
    """
    # Look for servers from known organizations or with specific keywords
    popular_keywords = [
        "github", "slack", "notion", "linear", "google", "postgres",
        "filesystem", "brave", "tavily", "discord", "stripe", "shopify"
    ]
    
    popular = []
    
    for server_data in all_servers:
        server = server_data.get("server", {})
        meta = server_data.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {})
        
        if meta.get("status") != "active":
            continue
        
        name = server.get("name", "").lower()
        
        for keyword in popular_keywords:
            if keyword in name:
                config = extract_server_config(server_data)
                popular.append(config)
                break
        
        if len(popular) >= limit:
            break
    
    return popular


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the discovery
    print("Testing MCP Server Discovery...\n")
    
    # Fetch all servers
    servers = fetch_all_servers()
    print(f"Total servers in registry: {len(servers)}")
    
    # Test searching for specific services
    test_services = ["github", "slack", "linear", "notion", "calendar"]
    
    for service in test_services:
        print(f"\n--- Searching for: {service} ---")
        matches = search_registry_for_service(service, servers)
        
        if matches:
            for i, match in enumerate(matches[:3]):
                config = extract_server_config(match)
                print(f"  {i+1}. {config['name']} (score: {match['score']})")
                print(f"     {config['description'][:80]}...")
                if config.get("install"):
                    print(f"     Install: {config['install']['command']}")
        else:
            print("  No matches found")
