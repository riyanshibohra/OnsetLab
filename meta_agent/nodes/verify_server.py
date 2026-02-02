"""
MCP Server Verification
=======================
Verifies that discovered MCP servers are valid and extracts tool information.
"""

import json
import re
import httpx
from typing import Optional
from datetime import datetime, timedelta
from meta_agent.state import MetaAgentState


# =============================================================================
# Verification Functions
# =============================================================================

def verify_npm_package(package_name: str) -> dict:
    """
    Verify an NPM package exists and get its metadata.
    
    Args:
        package_name: NPM package name (e.g., "@slack/mcp-server")
        
    Returns:
        Dict with verification status and metadata
    """
    try:
        # Clean package name (remove version if present)
        clean_name = package_name.split("@")[-1] if package_name.startswith("@") else package_name
        if "@" in clean_name:
            clean_name = package_name.rsplit("@", 1)[0]
        else:
            clean_name = package_name
            
        url = f"https://registry.npmjs.org/{clean_name}"
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            
            if response.status_code == 404:
                return {
                    "exists": False,
                    "error": "Package not found on NPM"
                }
            
            response.raise_for_status()
            data = response.json()
            
            # Extract useful info
            latest_version = data.get("dist-tags", {}).get("latest", "")
            time_data = data.get("time", {})
            last_modified = time_data.get(latest_version) or time_data.get("modified", "")
            
            return {
                "exists": True,
                "name": data.get("name", ""),
                "latest_version": latest_version,
                "description": data.get("description", ""),
                "last_modified": last_modified,
                "homepage": data.get("homepage", ""),
                "repository": data.get("repository", {}).get("url", ""),
                "weekly_downloads": None  # Would need additional API call
            }
            
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }


def verify_github_repo(repo_url: str) -> dict:
    """
    Verify a GitHub repository exists and check its activity.
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Dict with verification status and repo metadata
    """
    try:
        # Extract owner/repo from URL
        match = re.search(r"github\.com[/:]([^/]+)/([^/\.]+)", repo_url)
        if not match:
            return {
                "exists": False,
                "error": "Could not parse GitHub URL"
            }
        
        owner, repo = match.groups()
        repo = repo.replace(".git", "")
        
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(api_url)
            
            if response.status_code == 404:
                return {
                    "exists": False,
                    "error": "Repository not found"
                }
            
            response.raise_for_status()
            data = response.json()
            
            # Check if archived or disabled
            if data.get("archived"):
                return {
                    "exists": True,
                    "active": False,
                    "warning": "Repository is archived"
                }
            
            if data.get("disabled"):
                return {
                    "exists": True,
                    "active": False,
                    "warning": "Repository is disabled"
                }
            
            # Check last update
            pushed_at = data.get("pushed_at", "")
            last_push = None
            days_since_update = None
            
            if pushed_at:
                try:
                    last_push = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                    days_since_update = (datetime.now(last_push.tzinfo) - last_push).days
                except:
                    pass
            
            return {
                "exists": True,
                "active": True,
                "name": data.get("full_name", ""),
                "description": data.get("description", ""),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
                "last_push": pushed_at,
                "days_since_update": days_since_update,
                "default_branch": data.get("default_branch", "main"),
                "license": data.get("license", {}).get("spdx_id") if data.get("license") else None
            }
            
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }


def extract_tools_from_readme(repo_url: str) -> list[dict]:
    """
    Attempt to extract tool information from a repository's README.
    
    This is a best-effort extraction - may not work for all repos.
    """
    try:
        # Extract owner/repo from URL
        match = re.search(r"github\.com[/:]([^/]+)/([^/\.]+)", repo_url)
        if not match:
            return []
        
        owner, repo = match.groups()
        repo = repo.replace(".git", "")
        
        # Try to fetch README
        readme_urls = [
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
            f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
        ]
        
        readme_content = None
        with httpx.Client(timeout=10.0) as client:
            for url in readme_urls:
                response = client.get(url)
                if response.status_code == 200:
                    readme_content = response.text
                    break
        
        if not readme_content:
            return []
        
        # Try to extract tool information from README
        tools = []
        
        # Look for tool sections (common patterns)
        tool_patterns = [
            r"##\s*Tools?\s*\n([\s\S]*?)(?=\n##|\Z)",
            r"###\s*Available\s+Tools?\s*\n([\s\S]*?)(?=\n###|\n##|\Z)",
            r"\*\*Tools?\*\*:?\s*\n([\s\S]*?)(?=\n\*\*|\n##|\Z)",
        ]
        
        for pattern in tool_patterns:
            match = re.search(pattern, readme_content, re.IGNORECASE)
            if match:
                tools_section = match.group(1)
                
                # Extract individual tools (look for list items or headers)
                tool_matches = re.findall(
                    r"[-*]\s*`?(\w+)`?\s*[-:]\s*(.+?)(?=\n[-*]|\n\n|\Z)",
                    tools_section
                )
                
                for name, desc in tool_matches:
                    tools.append({
                        "name": name.strip(),
                        "description": desc.strip()[:200],
                        "source": "readme"
                    })
                
                if tools:
                    break
        
        return tools
        
    except Exception as e:
        print(f"      ⚠️ Could not extract tools from README: {e}")
        return []


# =============================================================================
# Main Verification Function
# =============================================================================

def verify_single_server(server_config: dict) -> dict:
    """
    Verify a single MCP server configuration.
    
    Verification Strategy (practical approach):
    1. If in official MCP Registry with status=active → baseline trust (30 points)
    2. If NPM package exists → additional trust (30 points)
    3. If GitHub repo active → additional trust (20 points)
    4. If has remote URL (hosted) → bonus (15 points)
    5. Tool extraction from README is OPTIONAL bonus
    
    We don't require tool extraction to pass - tools are discovered at runtime.
    
    Returns:
        Verification result with scores and warnings
    """
    result = {
        "verified": False,
        "score": 0,
        "warnings": [],
        "npm_info": None,
        "github_info": None,
        "extracted_tools": [],
        "max_score": 100,
        "install_ready": False,  # Can we generate install command?
    }
    
    name = server_config.get("name", "unknown")
    
    # Base score: If it's in the official registry, it has some validity
    if server_config.get("status") == "active":
        result["score"] += 30
    
    # Check NPM package
    install_info = server_config.get("install", {})
    if install_info and install_info.get("type") == "npm":
        package_name = install_info.get("package", "")
        if package_name:
            npm_result = verify_npm_package(package_name)
            result["npm_info"] = npm_result
            
            if npm_result.get("exists"):
                result["score"] += 30
                result["install_ready"] = True
                
                # Check how recently updated
                last_modified = npm_result.get("last_modified", "")
                if last_modified:
                    try:
                        mod_date = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                        days_old = (datetime.now(mod_date.tzinfo) - mod_date).days
                        if days_old < 30:
                            result["score"] += 5
                        elif days_old > 365:
                            result["warnings"].append(f"NPM package not updated in {days_old} days")
                    except:
                        pass
            else:
                result["warnings"].append(f"NPM package not found: {package_name}")
    
    # Docker installs are also valid
    if install_info and install_info.get("type") == "docker":
        result["score"] += 20  # Docker images are usually valid if in registry
        result["install_ready"] = True
    
    # Check GitHub repo (optional - not required for verification)
    repo_url = server_config.get("repository", "")
    if repo_url and "github.com" in repo_url:
        github_result = verify_github_repo(repo_url)
        result["github_info"] = github_result
        
        if github_result.get("exists"):
            result["score"] += 15
            
            if github_result.get("active"):
                result["score"] += 5
            
            # Bonus for popular repos
            stars = github_result.get("stars", 0)
            if stars > 100:
                result["score"] += 10
            elif stars > 10:
                result["score"] += 5
        # Don't penalize if GitHub not accessible - many official servers don't expose repos
    
    # Check for remote URL (hosted server - very reliable)
    remote_url = server_config.get("remote_url", "")
    if remote_url:
        result["score"] += 20
        result["install_ready"] = True  # Hosted servers don't need install
    
    # Try to extract tools from README (OPTIONAL - just a bonus)
    if repo_url and result["score"] < 80:  # Only bother if we need more confidence
        tools = extract_tools_from_readme(repo_url)
        if tools:
            result["extracted_tools"] = tools
            result["score"] += 5  # Small bonus
    
    # Determine if verified - lower threshold since tools are discovered at runtime
    # If it's in the registry and we can install it, it's probably good
    result["verified"] = result["score"] >= 30 and result["install_ready"]
    
    return result


# =============================================================================
# LangGraph Node
# =============================================================================

def verify_servers(state: MetaAgentState) -> dict:
    """
    LangGraph node: Verify discovered MCP servers.
    
    Takes discovered_servers from state and verifies each one.
    
    Args:
        state: Current MetaAgentState with discovered_servers
        
    Returns:
        State update with verified_servers list
    """
    discovered = state.get("discovered_servers", [])
    
    if not discovered:
        print("\n⚠️ No servers to verify")
        return {"verified_servers": [], "verification_summary": "No servers to verify"}
    
    print(f"\n✅ Verifying {len(discovered)} discovered servers...")
    
    verified = []
    
    for item in discovered:
        service = item.get("service", "unknown")
        server_config = item.get("server", {})
        
        print(f"\n   🔍 Verifying: {server_config.get('name', service)}")
        
        result = verify_single_server(server_config)
        
        # Print results
        status = "✅" if result["verified"] else "⚠️"
        print(f"      {status} Score: {result['score']}/{result['max_score']}")
        
        if result.get("warnings"):
            for warning in result["warnings"][:2]:
                print(f"      ⚠️  {warning}")
        
        if result.get("extracted_tools"):
            print(f"      📦 Found {len(result['extracted_tools'])} tools in README")
        
        verified.append({
            "service": service,
            "server": server_config,
            "verification": result,
            "alternatives": item.get("alternatives", [])
        })
    
    # Summary
    verified_count = sum(1 for v in verified if v["verification"]["verified"])
    print(f"\n   📊 Verified {verified_count}/{len(verified)} servers")
    
    summary = f"{verified_count}/{len(verified)} servers verified successfully"
    
    return {
        "verified_servers": verified,
        "verification_summary": summary
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing MCP Server Verification...\n")
    
    # Test with a known server
    test_configs = [
        {
            "name": "io.github.modelcontextprotocol/filesystem",
            "description": "Filesystem access",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "install": {
                "type": "npm",
                "package": "@modelcontextprotocol/server-filesystem"
            }
        },
        {
            "name": "slack-mcp-server",
            "description": "Slack integration",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "install": {
                "type": "npm",
                "package": "slack-mcp-server"
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Verifying: {config['name']} ---")
        result = verify_single_server(config)
        print(f"Verified: {result['verified']}")
        print(f"Score: {result['score']}/{result['max_score']}")
        print(f"Warnings: {result['warnings']}")
        if result.get("extracted_tools"):
            print(f"Tools found: {len(result['extracted_tools'])}")
            for tool in result["extracted_tools"][:3]:
                print(f"  - {tool['name']}: {tool['description'][:50]}...")
