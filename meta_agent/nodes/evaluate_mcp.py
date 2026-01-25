"""
Evaluate MCP Results Node
=========================
LLM evaluates search results to determine if a good MCP server exists.
"""

import json
import anthropic

from meta_agent.state import MetaAgentState


EVALUATE_MCP_PROMPT = """You are an expert at finding MCP (Model Context Protocol) servers for specific services.

## YOUR TASK:
Find MCP servers that provide tools for the SPECIFIC SERVICE requested.

## CRITICAL - MATCH THE SERVICE:
- The MCP server must provide API tools for the EXACT service requested
- The package/repo name should contain the service name
- DO NOT return generic tools (e.g., generic git for a platform-specific API)
- DO NOT return unrelated services

## CANDIDATE TYPES:

1. **NPM Packages** (start with @):
   - Format: @scope/service-mcp-server or @scope/service-mcp
   
2. **GitHub Repos** (owner/repo format):
   - Format: owner/service-mcp-server

Include BOTH types as candidates when available.

## PRIORITY ORDER:

1. **Well-maintained community packages** with the service name in package name
2. **GitHub repos** with high stars and recent commits
3. **Any active MCP server** for this service

## AVOID:

❌ Archived repos
❌ Generic tools that don't match the specific service API
❌ Wrong services (package name should match requested service)
❌ SDKs/clients (@modelcontextprotocol/sdk is NOT a server)
❌ Deprecated packages

## RESPONSE FORMAT (JSON):

{
    "quality": "good_mcp" | "no_mcp",
    "candidates": [
        {
            "package_name": "@scope/package-name OR owner/repo-name",
            "github_url": "https://github.com/...",
            "trust_level": "official" | "high_stars" | "community",
            "stars": number or null,
            "auth_type": "token" | "oauth" | "api_key" | "unknown",
            "env_var": "SERVICE_TOKEN",
            "confidence": 0.0-1.0,
            "reasoning": "Why this matches the service"
        }
    ],
    "reasoning": "Explanation of search results"
}

Return UP TO 5 candidates ranked by relevance to the SPECIFIC SERVICE.
Be STRICT about service matching - only return servers for the requested service."""


def evaluate_mcp_results(state: MetaAgentState) -> dict:
    """
    Evaluate search results to determine if a good MCP server exists.
    
    Uses LLM to analyze Tavily search results and determine:
    - Is there a viable MCP server?
    - What's the NPM package name?
    - What's the GitHub URL?
    - What auth type does it require?
    
    Sets result_quality to: "good_mcp" | "no_mcp"
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with result_quality and extracted info
    """
    current_service = state.get("current_service")
    search_results = state.get("search_results", "")
    api_key = state["anthropic_api_key"]
    
    if not current_service:
        print("⚠️ No current service to evaluate")
        return {"result_quality": "no_mcp"}
    
    print(f"🤔 Evaluating MCP results for: {current_service}")
    print(f"   📋 Using LLM to evaluate search results...")
    
    # Use LLM to analyze search results
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": f"""{EVALUATE_MCP_PROMPT}

Service: {current_service}

Search Results:
{search_results}

Evaluate whether a good MCP server exists for this service.

Respond with valid JSON only."""}
            ],
        )
        
        # Parse the response
        response_text = response.content[0].text
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        
        quality = result.get("quality", "no_mcp")
        candidates = result.get("candidates", [])
        reasoning = result.get("reasoning", "")
        
        # Log the decision
        if quality == "good_mcp" and candidates:
            print(f"   ✅ Found {len(candidates)} candidate(s):")
            for i, cand in enumerate(candidates):
                trust_emoji = {
                    "official": "🔒",
                    "high_stars": "⭐",
                    "curated": "📋",
                    "community": "👥",
                }.get(cand.get("trust_level", ""), "🔍")
                
                stars_info = f" ({cand.get('stars')}⭐)" if cand.get("stars") else ""
                print(f"      {i+1}. {trust_emoji} {cand.get('package_name')}{stars_info}")
                if cand.get("github_url"):
                    print(f"         🔗 {cand.get('github_url')}")
            
            # Use first candidate as primary
            primary = candidates[0]
            package_name = primary.get("package_name")
            github_url = primary.get("github_url")
            auth_type = primary.get("auth_type", "unknown")
            env_var = primary.get("env_var")
            trust_level = primary.get("trust_level", "unknown")
            stars = primary.get("stars")
            confidence = primary.get("confidence", 0.5)
        else:
            print(f"   ❌ No good MCP found")
            package_name = None
            github_url = None
            auth_type = "unknown"
            env_var = None
            trust_level = "unknown"
            stars = None
            confidence = 0.0
        
        if reasoning:
            print(f"   💭 {reasoning[:100]}...")
        
        # Store evaluation results in state for next node
        # Store ALL candidates for retry logic
        return {
            "result_quality": quality,
            # Primary candidate
            "_eval_package_name": package_name,
            "_eval_github_url": github_url,
            "_eval_auth_type": auth_type,
            "_eval_env_var": env_var,
            "_eval_trust_level": trust_level,
            "_eval_stars": stars,
            "_eval_confidence": confidence,
            "_eval_reasoning": reasoning,
            # All candidates for retry
            "_eval_candidates": candidates,
            "_eval_candidate_index": 0,
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse LLM response: {e}")
        return {
            "result_quality": "no_mcp",
            "errors": [f"Failed to evaluate MCP results: {e}"],
        }
    except Exception as e:
        print(f"❌ Error calling Claude: {e}")
        return {
            "result_quality": "no_mcp",
            "errors": [f"Claude error during evaluation: {e}"],
        }
