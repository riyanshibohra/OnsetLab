"""
Process Feedback Node
=====================
Processes user feedback on selected tools (HITL).
"""

import json
import anthropic
from meta_agent.state import MetaAgentState


FEEDBACK_PROMPT = """You are analyzing user feedback about a list of selected tools.

The user has reviewed the tools and provided feedback. Your job is to understand what they want:
1. "looks good" / "approve" / "yes" / "continue" → User approves, proceed
2. "add X" / "include X" / "also need X" → User wants to add tool(s)
3. "remove X" / "don't need X" / "exclude X" → User wants to remove tool(s)

CURRENT TOOLS:
{current_tools}

ALL AVAILABLE TOOLS FROM REGISTRY:
{all_tools}

USER FEEDBACK:
{feedback}

Respond with JSON:
{{
    "action": "approved" | "add_tools" | "remove_tools",
    "tools_to_add": ["tool_name1", "tool_name2"],  // Empty if not adding
    "tools_to_remove": ["tool_name1"],              // Empty if not removing
    "reasoning": "Brief explanation"
}}

IMPORTANT:
- For "add", find exact tool names from ALL AVAILABLE TOOLS
- For "remove", find exact tool names from CURRENT TOOLS
- Only return tool names that actually exist
- If feedback is approval (looks good), return action="approved" with empty lists"""


def process_feedback(state: MetaAgentState) -> dict:
    """
    Process user feedback on selected tools.
    
    Understands:
    - "looks good" / "approve" → proceed
    - "add list_issues" → add that tool
    - "remove search_repositories" → remove that tool
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with feedback_action and tool changes
    """
    feedback = state.get("user_feedback", "").strip()
    current_tools = state.get("filtered_tools", [])
    all_tools = state.get("all_tools", [])
    api_key = state["anthropic_api_key"]
    
    print(f"\n💬 Processing user feedback: '{feedback}'")
    
    # Quick check for simple approvals
    if not feedback or feedback.lower() in ["looks good", "approve", "yes", "continue", "ok", "good"]:
        print("   ✅ User approved!")
        return {
            "feedback_action": "approved",
            "final_tools": current_tools,
            "tools_to_add": [],
            "tools_to_remove": []
        }
    
    # Use LLM to parse complex feedback
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Format tools for prompt
        current_tools_str = "\n".join([f"- {t['name']}: {t['description'][:60]}" for t in current_tools])
        all_tools_str = "\n".join([f"- {t['name']}: {t['description'][:60]}" for t in all_tools])
        
        prompt = FEEDBACK_PROMPT.format(
            current_tools=current_tools_str,
            all_tools=all_tools_str,
            feedback=feedback
        )
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        
        action = result.get("action", "approved")
        tools_to_add = result.get("tools_to_add", [])
        tools_to_remove = result.get("tools_to_remove", [])
        reasoning = result.get("reasoning", "")
        
        print(f"   🤔 Action: {action}")
        if reasoning:
            print(f"   💭 {reasoning}")
        if tools_to_add:
            print(f"   ➕ Adding: {', '.join(tools_to_add)}")
        if tools_to_remove:
            print(f"   ➖ Removing: {', '.join(tools_to_remove)}")
        
        return {
            "feedback_action": action,
            "tools_to_add": tools_to_add,
            "tools_to_remove": tools_to_remove,
            "final_tools": current_tools if action == "approved" else None
        }
        
    except Exception as e:
        print(f"   ⚠️ Failed to parse feedback: {e}")
        print(f"   ✅ Approving by default")
        return {
            "feedback_action": "approved",
            "final_tools": current_tools,
            "tools_to_add": [],
            "tools_to_remove": []
        }
