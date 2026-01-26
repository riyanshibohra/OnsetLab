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
    
    Can operate in two modes:
    1. Initial filtering: Use LLM to select tools from all_tools
    2. Feedback mode: Adjust existing filtered_tools based on user's add/remove requests
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with filtered_tools (renamed from filtered_tool_schemas for consistency)
    """
    problem_statement = state.get("problem_statement", "")
    all_tools = state.get("all_tools", [])  # From load_registry
    tool_schemas = state.get("tool_schemas", [])  # Legacy field, prefer all_tools
    mcp_servers = state.get("mcp_servers", [])
    api_servers = state.get("api_servers", [])
    api_key = state["anthropic_api_key"]
    
    # Check if coming from feedback loop
    tools_to_add = state.get("tools_to_add", [])
    tools_to_remove = state.get("tools_to_remove", [])
    
    # Use all_tools if available (from new registry flow), else fall back to tool_schemas (legacy)
    source_tools = all_tools if all_tools else tool_schemas
    
    # FEEDBACK MODE: User requested changes
    if tools_to_remove:
        print(f"\n➖ Removing {len(tools_to_remove)} tools...")
        current_tools = state.get("filtered_tools", [])
        filtered_tools = [t for t in current_tools if t["name"] not in tools_to_remove]
        print(f"   ✅ Removed. Now have {len(filtered_tools)} tools")
        
        return {
            "filtered_tools": filtered_tools,
            "filtered_tool_schemas": filtered_tools,  # Legacy field
            "tools_to_remove": []  # Clear for next iteration
        }
    
    if tools_to_add:
        print(f"\n➕ Adding {len(tools_to_add)} tools...")
        current_tools = state.get("filtered_tools", [])
        
        for tool_name in tools_to_add:
            tool = next((t for t in source_tools if t["name"] == tool_name), None)
            if tool and tool not in current_tools:
                current_tools.append(tool)
                print(f"   ✅ Added {tool_name}")
            elif not tool:
                print(f"   ⚠️ Tool '{tool_name}' not found in registry")
        
        return {
            "filtered_tools": current_tools,
            "filtered_tool_schemas": current_tools,  # Legacy field
            "tools_to_add": []  # Clear for next iteration
        }
    
    # INITIAL MODE: LLM filters from scratch
    print("\n🔍 Filtering tools for relevance...")
    print(f"   📝 Problem: {problem_statement[:100]}...")
    print(f"   📦 Total tools before filtering: {len(source_tools)}")
    
    if len(source_tools) <= 20:
        print(f"   ✅ Already within limit ({len(source_tools)} tools), skipping filter")
        return {
            "filtered_tools": source_tools,
            "filtered_tool_schemas": source_tools,  # Legacy
            "filtered_mcp_servers": mcp_servers,
            "filtered_api_servers": api_servers,
        }
    
    # Format tools for LLM
    tools_list = []
    for tool in source_tools:
        tools_list.append({
            "name": tool.get("name"),
            "description": tool.get("description", "")[:100],
            "service": tool.get("_service", "unknown")
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
        reasoning = result.get("reasoning", "")
        
        # Filter tools
        filtered_tools = [
            tool for tool in source_tools
            if tool.get("name") in selected_tool_names
        ]
        
        print(f"   ✅ Filtered to {len(filtered_tools)} relevant tools")
        print(f"   🗑️ Excluded {len(source_tools) - len(filtered_tools)} tools")
        
        if reasoning:
            print(f"   💭 Strategy: {reasoning[:100]}...")
        
        # Show selected tools by service
        by_service = {}
        for tool in filtered_tools:
            service = tool.get("_service", "unknown")
            by_service[service] = by_service.get(service, 0) + 1
        
        print(f"   📋 Selected by service:")
        for service, count in sorted(by_service.items()):
            print(f"      • {service}: {count} tools")
        
        return {
            "filtered_tools": filtered_tools,
            "filtered_tool_schemas": filtered_tools,  # Legacy field for backwards compat
            "filtered_mcp_servers": mcp_servers,
            "filtered_api_servers": api_servers,
        }
        
    except Exception as e:
        print(f"   ❌ Failed to filter tools: {e}")
        print(f"   ⚠️ Using all {len(source_tools)} tools")
        return {
            "filtered_tools": source_tools,
            "filtered_tool_schemas": source_tools,
            "filtered_mcp_servers": mcp_servers,
            "filtered_api_servers": api_servers,
        }
