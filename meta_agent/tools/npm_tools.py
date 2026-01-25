"""
NPM Tools
=========
Tools for validating and getting info about NPM packages.
"""

from typing import Optional
from langchain_core.tools import tool
import httpx


def get_npm_package_info(package_name: str) -> dict:
    """
    Get information about an NPM package from the registry.
    
    Also checks for deprecation notices and extracts new repo URLs.
    
    Args:
        package_name: NPM package name (e.g., "@modelcontextprotocol/server-github")
        
    Returns:
        Dictionary with package information including deprecation status
    """
    import re
    
    # NPM registry URL
    # Scoped packages need URL encoding for the @
    encoded_name = package_name.replace("/", "%2F")
    url = f"https://registry.npmjs.org/{encoded_name}"
    
    with httpx.Client() as client:
        try:
            response = client.get(url, timeout=10.0)
            
            if response.status_code == 404:
                return {
                    "exists": False,
                    "package": package_name,
                    "error": "Package not found"
                }
            
            if response.status_code != 200:
                return {
                    "exists": False,
                    "package": package_name,
                    "error": f"HTTP {response.status_code}"
                }
            
            data = response.json()
            
            # Extract useful information
            latest_version = data.get("dist-tags", {}).get("latest", "unknown")
            version_info = data.get("versions", {}).get(latest_version, {})
            
            # Check for deprecation in latest version
            deprecated = version_info.get("deprecated", "")
            is_deprecated = bool(deprecated)
            
            # Also check readme for deprecation notices
            readme = data.get("readme", "")
            
            # Try to extract new repo URL from deprecation notice or readme
            new_repo_url = None
            deprecation_text = deprecated or readme[:2000]  # Check first 2000 chars of readme
            
            # Look for GitHub URLs in deprecation notice
            github_patterns = [
                r'moved to[:\s]+(?:the\s+)?(?:https?://)?github\.com/([^\s\)\"\']+)',
                r'(?:https?://)?github\.com/([^\s\)\"\']+)(?:\s+repo)',
                r'new (?:location|repo|repository)[:\s]+(?:https?://)?github\.com/([^\s\)\"\']+)',
            ]
            
            for pattern in github_patterns:
                match = re.search(pattern, deprecation_text, re.IGNORECASE)
                if match:
                    new_repo_url = f"https://github.com/{match.group(1).rstrip('.,;:')}"
                    break
            
            # Get repository URL
            repo = data.get("repository", {})
            if isinstance(repo, dict):
                repo_url = repo.get("url", "")
            else:
                repo_url = str(repo) if repo else ""
            
            # Clean up repo URL (remove git+ prefix and .git suffix)
            if repo_url.startswith("git+"):
                repo_url = repo_url[4:]
            if repo_url.endswith(".git"):
                repo_url = repo_url[:-4]
            if repo_url.startswith("git://"):
                repo_url = repo_url.replace("git://", "https://")
            
            # If deprecated and we found a new URL, prefer that
            final_repo_url = new_repo_url if new_repo_url else repo_url
            
            result = {
                "exists": True,
                "package": package_name,
                "version": latest_version,
                "description": data.get("description", ""),
                "repository": final_repo_url,
                "original_repository": repo_url if new_repo_url else None,
                "homepage": data.get("homepage", ""),
                "keywords": data.get("keywords", []),
                "author": data.get("author", {}),
                "license": data.get("license", ""),
                "dependencies": list(version_info.get("dependencies", {}).keys())[:10],
                "deprecated": is_deprecated,
                "deprecation_message": deprecated if deprecated else None,
                "new_repo_url": new_repo_url,
            }
            
            return result
            
        except httpx.TimeoutException:
            return {
                "exists": False,
                "package": package_name,
                "error": "Request timed out"
            }
        except Exception as e:
            return {
                "exists": False,
                "package": package_name,
                "error": str(e)
            }


@tool
def validate_npm_package(package_name: str) -> str:
    """
    Check if an NPM package exists and get its metadata.
    
    Use this to verify MCP server packages are real and get their:
    - GitHub repository URL
    - Description
    - Latest version
    - Dependencies
    
    Args:
        package_name: NPM package name (e.g., "@modelcontextprotocol/server-github")
        
    Returns:
        Package information as formatted string
    """
    info = get_npm_package_info(package_name)
    
    if not info.get("exists"):
        return f"Package '{package_name}' not found: {info.get('error', 'Unknown error')}"
    
    lines = [
        f"Package: {info['package']}",
        f"Version: {info.get('version', 'unknown')}",
        f"Description: {info.get('description', 'No description')}",
    ]
    
    if info.get("repository"):
        lines.append(f"Repository: {info['repository']}")
    
    if info.get("homepage"):
        lines.append(f"Homepage: {info['homepage']}")
    
    if info.get("keywords"):
        lines.append(f"Keywords: {', '.join(info['keywords'][:5])}")
    
    if info.get("dependencies"):
        lines.append(f"Dependencies: {', '.join(info['dependencies'])}")
    
    return "\n".join(lines)


def is_mcp_package(package_name: str) -> bool:
    """
    Check if a package appears to be an MCP server package.
    
    Args:
        package_name: NPM package name
        
    Returns:
        True if the package appears to be an MCP server
    """
    info = get_npm_package_info(package_name)
    
    if not info.get("exists"):
        return False
    
    # Check keywords
    keywords = [k.lower() for k in info.get("keywords", [])]
    if "mcp" in keywords or "model-context-protocol" in keywords:
        return True
    
    # Check description
    description = info.get("description", "").lower()
    if "mcp" in description or "model context protocol" in description:
        return True
    
    # Check package name
    if "mcp" in package_name.lower():
        return True
    
    # Check if it's in the official MCP namespace
    if package_name.startswith("@modelcontextprotocol/"):
        return True
    
    return False


def validate_mcp_package_health(package_name: str) -> dict:
    """
    Validate that an MCP package is properly configured and likely to work.
    
    Checks for:
    - Package exists
    - Not deprecated
    - Has @modelcontextprotocol/sdk as dependency (required for MCP servers)
    - Has reasonable dependency count (not empty, not bloated)
    
    Args:
        package_name: NPM package name
        
    Returns:
        Dictionary with health status:
        {
            "healthy": bool,
            "issues": list[str],
            "warnings": list[str],
            "score": int (0-100, higher is better),
            "recommendation": str
        }
    """
    info = get_npm_package_info(package_name)
    
    issues = []
    warnings = []
    score = 100
    
    # Check existence
    if not info.get("exists"):
        return {
            "healthy": False,
            "issues": [f"Package not found: {info.get('error', 'Unknown')}"],
            "warnings": [],
            "score": 0,
            "recommendation": "Package does not exist on npm"
        }
    
    # Check deprecation
    if info.get("deprecated"):
        issues.append(f"Package is deprecated: {info.get('deprecation_message', 'No message')}")
        score -= 50
    
    # Get dependencies
    dependencies = info.get("dependencies", [])
    
    # Check for @modelcontextprotocol/sdk dependency
    has_mcp_sdk = any(
        dep in dependencies 
        for dep in ["@modelcontextprotocol/sdk", "@anthropic-ai/mcp-sdk"]
    )
    
    if not has_mcp_sdk:
        # This is the bug the user encountered!
        issues.append(
            "Missing @modelcontextprotocol/sdk dependency - package may fail at runtime"
        )
        score -= 40
    
    # Check for empty dependencies (suspicious for MCP server)
    if len(dependencies) == 0:
        warnings.append("No dependencies listed - may have bundling issues")
        score -= 10
    
    # Prefer official packages
    is_official = package_name.startswith("@modelcontextprotocol/")
    if is_official:
        score += 20  # Bonus for official
    
    # Generate recommendation
    if score >= 80:
        recommendation = "✅ Package looks healthy"
    elif score >= 50:
        recommendation = "⚠️ Package may have issues, use with caution"
    else:
        recommendation = "❌ Package has critical issues, consider alternatives"
    
    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "score": min(100, max(0, score)),
        "recommendation": recommendation,
        "has_mcp_sdk": has_mcp_sdk,
        "is_official": is_official,
        "dependencies": dependencies,
    }


