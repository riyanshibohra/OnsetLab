"""
GitHub Tools
============
Tools for fetching content from GitHub repositories.
"""

import re
import base64
from typing import Optional
from langchain_core.tools import tool
import httpx


def parse_github_url(url: str) -> tuple[str, str, str, str]:
    """
    Parse a GitHub URL to extract owner, repo, branch, and subpath.
    
    Args:
        url: GitHub URL (e.g., "https://github.com/owner/repo" or 
             "https://github.com/owner/repo/tree/main/src/subdir")
        
    Returns:
        Tuple of (owner, repo, branch, subpath)
        - subpath is empty string if URL points to repo root
    """
    # Handle URLs with /tree/branch/path format (subdirectory links)
    tree_match = re.search(
        r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+?)/?$",
        url
    )
    if tree_match:
        owner, repo, branch, subpath = tree_match.groups()
        return owner, repo, branch, subpath.rstrip("/")
    
    # Handle URLs with /blob/branch/path format (file links)
    blob_match = re.search(
        r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)/?$",
        url
    )
    if blob_match:
        owner, repo, branch, subpath = blob_match.groups()
        return owner, repo, branch, subpath.rstrip("/")
    
    # Handle basic repo URLs
    patterns = [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)",
        r"^([^/]+)/([^/]+)$",  # owner/repo format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2).rstrip("/"), "main", ""
    
    raise ValueError(f"Could not parse GitHub URL: {url}")


async def fetch_github_file_async(
    repo_url: str,
    file_path: str = "README.md",
    branch: str = None
) -> str:
    """
    Fetch a file from a GitHub repository using the GitHub API.
    
    Args:
        repo_url: GitHub repository URL or owner/repo string
                  Can include subdirectory: https://github.com/owner/repo/tree/main/src/subdir
        file_path: Path to file within repo/subdir (default: README.md)
        branch: Branch to fetch from (default: auto-detected or main)
        
    Returns:
        File content as string
    """
    owner, repo, url_branch, subpath = parse_github_url(repo_url)
    branch = branch or url_branch
    
    # If URL pointed to a subdirectory, prepend it to file_path
    if subpath:
        file_path = f"{subpath}/{file_path}"
    
    # Try GitHub API first
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    async with httpx.AsyncClient() as client:
        # Try specified branch first, then master
        for b in [branch, "master"]:
            try:
                response = await client.get(
                    api_url,
                    params={"ref": b},
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("encoding") == "base64":
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        return content
                    return data.get("content", "")
                    
            except Exception:
                continue
        
        # Fallback to raw content URL
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        try:
            response = await client.get(raw_url, timeout=10.0)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass
        
        # Try master branch for raw URL
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}"
        response = await client.get(raw_url, timeout=10.0)
        if response.status_code == 200:
            return response.text
            
        raise Exception(f"Could not fetch {file_path} from {repo_url}")


def fetch_github_file_sync(
    repo_url: str,
    file_path: str = "README.md",
    branch: str = None
) -> str:
    """
    Synchronous version of fetch_github_file_async.
    
    Handles subdirectory URLs like:
    https://github.com/owner/repo/tree/main/src/subdir
    """
    owner, repo, url_branch, subpath = parse_github_url(repo_url)
    branch = branch or url_branch
    
    # If URL pointed to a subdirectory, prepend it to file_path
    if subpath:
        file_path = f"{subpath}/{file_path}"
    
    with httpx.Client() as client:
        # Try specified branch first, then master
        for b in [branch, "master"]:
            # Try raw URL first (simpler)
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{b}/{file_path}"
            try:
                response = client.get(raw_url, timeout=10.0, follow_redirects=True)
                if response.status_code == 200:
                    return response.text
            except Exception:
                continue
        
        raise Exception(f"Could not fetch {file_path} from {repo_url}")


@tool
def fetch_github_readme(repo_url: str) -> str:
    """
    Fetch the README content from a GitHub repository.
    
    Use this to get documentation about MCP servers, including:
    - Tool descriptions and capabilities
    - Installation instructions
    - Authentication requirements
    - Usage examples
    
    Args:
        repo_url: GitHub repository URL (e.g., "https://github.com/owner/repo")
        
    Returns:
        README.md content as string
    """
    # Try common README filenames
    for filename in ["README.md", "readme.md", "Readme.md", "README"]:
        try:
            return fetch_github_file_sync(repo_url, filename)
        except Exception:
            continue
    
    return f"Could not find README in {repo_url}"


@tool
def fetch_github_file(repo_url: str, file_path: str) -> str:
    """
    Fetch a specific file from a GitHub repository.
    
    Use this to get source code files that may contain tool definitions,
    such as TypeScript files with MCP tool schemas.
    
    Args:
        repo_url: GitHub repository URL
        file_path: Path to file within repo (e.g., "src/tools.ts", "package.json")
        
    Returns:
        File content as string
    """
    return fetch_github_file_sync(repo_url, file_path)


def extract_package_json(repo_url: str) -> Optional[dict]:
    """
    Fetch and parse package.json from a GitHub repository.
    
    Handles subdirectory URLs - will look for package.json in the subdirectory.
    
    Args:
        repo_url: GitHub repository URL (can include subdirectory path)
        
    Returns:
        Parsed package.json as dict, or None if not found
    """
    import json
    
    try:
        # fetch_github_file_sync now handles subdirectory URLs automatically
        content = fetch_github_file_sync(repo_url, "package.json")
        return json.loads(content)
    except Exception:
        return None
