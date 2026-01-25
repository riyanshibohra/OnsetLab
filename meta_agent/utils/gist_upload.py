"""
GitHub Gist Upload Utility
==========================
Upload notebooks to GitHub Gist for easy Colab sharing.
"""

import os
import json
from typing import Optional
import httpx


def upload_to_gist(
    notebook_json: str,
    filename: str = "onsetlab_agent_builder.ipynb",
    description: str = "OnsetLab Agent Builder Notebook",
    github_token: Optional[str] = None,
    public: bool = True,
) -> dict:
    """
    Upload a notebook to GitHub Gist.
    
    Args:
        notebook_json: The notebook content as JSON string
        filename: Name for the notebook file
        description: Gist description
        github_token: GitHub personal access token (or uses GITHUB_TOKEN env var)
        public: Whether the gist should be public
        
    Returns:
        Dictionary with:
        - gist_id: The Gist ID
        - gist_url: URL to the Gist
        - raw_url: URL to the raw notebook file
        - colab_url: Direct link to open in Colab
        
    Raises:
        ValueError: If no GitHub token is provided
        Exception: If upload fails
    """
    token = github_token or os.getenv("GITHUB_TOKEN")
    
    if not token:
        raise ValueError(
            "GitHub token required for Gist upload. "
            "Set GITHUB_TOKEN environment variable or pass github_token parameter."
        )
    
    # Prepare the Gist payload
    payload = {
        "description": description,
        "public": public,
        "files": {
            filename: {
                "content": notebook_json
            }
        }
    }
    
    # Upload to GitHub Gist API
    with httpx.Client() as client:
        response = client.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code not in (200, 201):
            error_detail = response.text[:500]
            raise Exception(
                f"Failed to create Gist: HTTP {response.status_code} - {error_detail}"
            )
        
        data = response.json()
        
        gist_id = data["id"]
        gist_url = data["html_url"]
        
        # Get raw URL for the notebook file
        raw_url = data["files"][filename]["raw_url"]
        
        # Generate Colab URL
        # Format: https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}
        owner = data["owner"]["login"]
        colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"
        
        return {
            "gist_id": gist_id,
            "gist_url": gist_url,
            "raw_url": raw_url,
            "colab_url": colab_url,
            "owner": owner,
        }


def get_colab_url(gist_url: str) -> str:
    """
    Convert a GitHub Gist URL to a Colab URL.
    
    Args:
        gist_url: GitHub Gist URL (e.g., https://gist.github.com/user/abc123)
        
    Returns:
        Colab URL for the notebook
    """
    # Parse the gist URL to extract owner and ID
    # Format: https://gist.github.com/{owner}/{gist_id}
    parts = gist_url.rstrip("/").split("/")
    
    if len(parts) < 2:
        raise ValueError(f"Invalid Gist URL: {gist_url}")
    
    gist_id = parts[-1]
    owner = parts[-2]
    
    # Assume the notebook filename
    filename = "onsetlab_agent_builder.ipynb"
    
    return f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"


async def upload_to_gist_async(
    notebook_json: str,
    filename: str = "onsetlab_agent_builder.ipynb",
    description: str = "OnsetLab Agent Builder Notebook",
    github_token: Optional[str] = None,
    public: bool = True,
) -> dict:
    """
    Async version of upload_to_gist.
    """
    token = github_token or os.getenv("GITHUB_TOKEN")
    
    if not token:
        raise ValueError(
            "GitHub token required for Gist upload. "
            "Set GITHUB_TOKEN environment variable or pass github_token parameter."
        )
    
    payload = {
        "description": description,
        "public": public,
        "files": {
            filename: {
                "content": notebook_json
            }
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code not in (200, 201):
            error_detail = response.text[:500]
            raise Exception(
                f"Failed to create Gist: HTTP {response.status_code} - {error_detail}"
            )
        
        data = response.json()
        
        gist_id = data["id"]
        gist_url = data["html_url"]
        raw_url = data["files"][filename]["raw_url"]
        owner = data["owner"]["login"]
        colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"
        
        return {
            "gist_id": gist_id,
            "gist_url": gist_url,
            "raw_url": raw_url,
            "colab_url": colab_url,
            "owner": owner,
        }


def update_gist(
    gist_id: str,
    notebook_json: str,
    filename: str = "onsetlab_agent_builder.ipynb",
    github_token: Optional[str] = None,
) -> dict:
    """
    Update an existing Gist with new content.
    
    Args:
        gist_id: The Gist ID to update
        notebook_json: New notebook content
        filename: Name of the file to update
        github_token: GitHub personal access token
        
    Returns:
        Dictionary with gist info
    """
    token = github_token or os.getenv("GITHUB_TOKEN")
    
    if not token:
        raise ValueError("GitHub token required for Gist update.")
    
    payload = {
        "files": {
            filename: {
                "content": notebook_json
            }
        }
    }
    
    with httpx.Client() as client:
        response = client.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to update Gist: HTTP {response.status_code}")
        
        data = response.json()
        
        return {
            "gist_id": data["id"],
            "gist_url": data["html_url"],
            "raw_url": data["files"][filename]["raw_url"],
        }
