"""
Filter Tools Node
=================
Filters discovered tools to only include those relevant to the problem statement.
Limits to max 15-20 tools to keep the agent focused.
"""

import json
import anthropic

from meta_agent.state import MetaAgentState


FILTER_TOOLS_PROMPT = """You are an expert at selecting the most relevant tools for a specific task.

Given a problem statement and a list of discovered tools, select ONLY the tools that are directly relevant to solving the problem.

## RULES:

1. Select tools that DIRECTLY help with the problem statement
2. Maximum 15-20 tools (prefer fewer, more focused tools)
3. Exclude tools that are:
   - Too generic (like basic git operations unless specifically needed)
   - Not mentioned or implied in the problem
   - Redundant (if two tools do similar things, pick the better one)
   
4. Prioritize tools that:
   - Are explicitly mentioned in the problem
   - Enable core functionality described
   - Are essential for the workflow

## SELECTION CRITERIA:

✅ KEEP tools that:
- Directly relate to actions in the problem statement
- Are core to the described workflow
- Enable the main functionality

❌ EXCLUDE tools that:
- Are tangentially related but not needed
- Handle advanced/admin features not mentioned
- Are generic/low-level operations

## RESPONSE FORMAT (JSON):

{
    "selected_tools": [
        {
            "name": "tool_name",
            "reason": "Brief reason why this tool is needed"
        }
    ],
    "excluded_count": number,
    "reasoning": "Overall explanation of selection strategy"
}

Be strict - only include tools that are CLEARLY needed for the problem."""


def filter_tools(state: MetaAgentState) -> dict:
    """
    Filter discovered tools to only those relevant to the problem statement.
    
    Takes all tool_schemas and mcp_servers, filters tools based on
    relevance to problem_statement, and updates the state.
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with filtered tool_schemas and updated mcp_servers
    """
    problem_statement = state.get("problem_statement", "")
    tool_schemas = state.get("tool_schemas", [])
    mcp_servers = state.get("mcp_servers", [])
    api_servers = state.get("api_servers", [])
    api_key = state["anthropic_api_key"]
    
    print("\n🔍 Filtering tools for relevance...")
    print(f"   📝 Problem: {problem_statement[:100]}...")
    print(f"   📦 Total tools before filtering: {len(tool_schemas)}")
    
    if len(tool_schemas) <= 20:
        print(f"   ✅ Already within limit ({len(tool_schemas)} tools), skipping filter")
        # Still set filtered fields to the full list (no filtering needed)
        return {
            "filtered_mcp_servers": mcp_servers,
            "filtered_api_servers": api_servers,
            "filtered_tool_schemas": tool_schemas,
        }
    
    # Format tools for LLM
    tools_list = []
    for tool in tool_schemas:
        tools_list.append({
            "name": tool.get("name"),
            "description": tool.get("description", "")[:100],
        })
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": f"""{FILTER_TOOLS_PROMPT}

Problem Statement:
{problem_statement}

Available Tools ({len(tools_list)} total):
{json.dumps(tools_list, indent=2)}

Select only the tools that are relevant to this specific problem.
Respond with valid JSON."""}
            ],
        )
        
        response_text = response.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        
        selected_tool_names = {t["name"] for t in result.get("selected_tools", [])}
        excluded_count = result.get("excluded_count", 0)
        reasoning = result.get("reasoning", "")
        
        # Filter tool_schemas
        filtered_schemas = [
            tool for tool in tool_schemas
            if tool.get("name") in selected_tool_names
        ]
        
        # Also filter tools in mcp_servers
        filtered_mcp_servers = []
        for server in mcp_servers:
            server_tools = server.get("tools", [])
            filtered_server_tools = [
                t for t in server_tools
                if t.get("name") in selected_tool_names
            ]
            # Create updated server with filtered tools
            filtered_server = dict(server)
            filtered_server["tools"] = filtered_server_tools
            filtered_mcp_servers.append(filtered_server)
        
        # Filter tools in api_servers
        filtered_api_servers = []
        for server in api_servers:
            endpoints = server.get("endpoints", [])
            filtered_endpoints = [
                e for e in endpoints
                if e.get("name") in selected_tool_names
            ]
            filtered_server = dict(server)
            filtered_server["endpoints"] = filtered_endpoints
            filtered_api_servers.append(filtered_server)
        
        print(f"   ✅ Filtered to {len(filtered_schemas)} relevant tools")
        print(f"   🗑️ Excluded {len(tool_schemas) - len(filtered_schemas)} irrelevant tools")
        
        if reasoning:
            print(f"   💭 Strategy: {reasoning[:100]}...")
        
        # Show selected tools
        print(f"   📋 Selected tools:")
        for tool_info in result.get("selected_tools", [])[:10]:
            print(f"      • {tool_info['name']}: {tool_info.get('reason', '')[:40]}")
        if len(result.get("selected_tools", [])) > 10:
            print(f"      ... and {len(result.get('selected_tools', [])) - 10} more")
        
        return {
            "filtered_tool_schemas": filtered_schemas,
            "filtered_mcp_servers": filtered_mcp_servers,
            "filtered_api_servers": filtered_api_servers,
        }
        
    except Exception as e:
        print(f"   ❌ Failed to filter tools: {e}")
        print(f"   ⚠️ Using all {len(tool_schemas)} tools")
        # On error, use unfiltered results
        return {
            "filtered_mcp_servers": mcp_servers,
            "filtered_api_servers": api_servers,
            "filtered_tool_schemas": tool_schemas,
        }
