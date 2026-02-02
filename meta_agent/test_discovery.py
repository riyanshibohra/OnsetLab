#!/usr/bin/env python3
"""
Test MCP Server Discovery and Verification
==========================================
Quick test to ensure the discovery pipeline works.
"""

import sys
sys.path.insert(0, ".")

from meta_agent.nodes.discover_servers import (
    fetch_all_servers,
    search_registry_for_service,
    extract_server_config
)
from meta_agent.nodes.verify_server import verify_single_server


def test_discovery():
    """Test the full discovery + verification pipeline."""
    print("=" * 60)
    print("Testing MCP Server Discovery & Verification")
    print("=" * 60)
    
    # Step 1: Fetch all servers from registry
    print("\n📡 Step 1: Fetching from MCP Registry...")
    all_servers = fetch_all_servers()
    
    if not all_servers:
        print("❌ Failed to fetch servers from registry")
        return False
    
    print(f"✅ Fetched {len(all_servers)} servers")
    
    # Step 2: Test searching for common services
    test_services = ["github", "slack", "notion", "linear", "filesystem"]
    
    print(f"\n🔍 Step 2: Searching for {len(test_services)} services...")
    
    results = {}
    for service in test_services:
        print(f"\n   Searching for: {service}")
        matches = search_registry_for_service(service, all_servers)
        
        if matches:
            best = matches[0]
            config = extract_server_config(best)
            results[service] = config
            print(f"   ✅ Found: {config['name']}")
            print(f"      Description: {config['description'][:60]}...")
            if config.get("install"):
                print(f"      Install: {config['install'].get('command', 'N/A')}")
        else:
            results[service] = None
            print(f"   ❌ No match found")
    
    # Step 3: Verify the found servers
    print(f"\n✅ Step 3: Verifying discovered servers...")
    
    for service, config in results.items():
        if not config:
            continue
            
        print(f"\n   Verifying: {config['name']}")
        verification = verify_single_server(config)
        
        status = "✅" if verification["verified"] else "⚠️"
        print(f"   {status} Score: {verification['score']}/{verification['max_score']}")
        
        npm_info = verification.get("npm_info") or {}
        if npm_info.get("exists"):
            print(f"      📦 NPM: {npm_info.get('latest_version', 'N/A')}")
        
        github_info = verification.get("github_info") or {}
        if github_info.get("exists"):
            stars = github_info.get("stars", 0)
            print(f"      ⭐ GitHub: {stars} stars")
        
        if verification.get("warnings"):
            for w in verification["warnings"][:2]:
                print(f"      ⚠️  {w}")
        
        if verification.get("extracted_tools"):
            print(f"      🔧 Tools: {len(verification['extracted_tools'])} found")
    
    # Summary
    print("\n" + "=" * 60)
    found = sum(1 for v in results.values() if v is not None)
    print(f"Summary: Found {found}/{len(test_services)} services")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_discovery()
    sys.exit(0 if success else 1)
