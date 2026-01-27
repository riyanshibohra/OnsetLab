"""
Process Feedback Node
=====================
Processes user feedback on selected tools (HITL).

No LLM needed - uses simple parsing for UI-provided feedback.
"""

import re
from meta_agent.state import MetaAgentState


def process_feedback(state: MetaAgentState) -> dict:
    """
    Process user feedback on selected tools.
    
    Expects feedback from UI in simple format:
    - "looks good" / "approve" / "yes" → proceed
    - "add tool_name" → add that tool
    - "add tool1, tool2, tool3" → add multiple tools
    - "remove tool_name" → remove that tool
    - "remove tool1, tool2" → remove multiple tools
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with feedback_action and tool changes
    """
    feedback = state.get("user_feedback", "").strip().lower()
    current_tools = state.get("filtered_tools", [])
    all_tools = state.get("all_tools", [])
    
    print(f"\n💬 Processing user feedback: '{feedback}'")
    
    # Check for approval
    approval_phrases = ["looks good", "approve", "yes", "continue", "ok", "good", "done", ""]
    if feedback in approval_phrases:
        print("   ✅ User approved!")
        return {
            "feedback_action": "approved",
            "final_tools": current_tools,
            "tools_to_add": [],
            "tools_to_remove": []
        }
    
    # Check for "add" command
    add_match = re.match(r'^add\s+(.+)$', feedback, re.IGNORECASE)
    if add_match:
        # Parse tool names (comma or space separated)
        tools_str = add_match.group(1)
        tool_names = [t.strip() for t in re.split(r'[,\s]+', tools_str) if t.strip()]
        
        # Validate tools exist in registry
        available_names = {t['name'] for t in all_tools}
        valid_tools = [t for t in tool_names if t in available_names]
        invalid_tools = [t for t in tool_names if t not in available_names]
        
        if invalid_tools:
            print(f"   ⚠️ Unknown tools (ignored): {', '.join(invalid_tools)}")
        
        if valid_tools:
            print(f"   ➕ Adding: {', '.join(valid_tools)}")
            return {
                "feedback_action": "add_tools",
                "tools_to_add": valid_tools,
                "tools_to_remove": [],
                "final_tools": None
            }
        else:
            print("   ⚠️ No valid tools to add, approving")
            return {
                "feedback_action": "approved",
                "final_tools": current_tools,
                "tools_to_add": [],
                "tools_to_remove": []
            }
    
    # Check for "remove" command
    remove_match = re.match(r'^remove\s+(.+)$', feedback, re.IGNORECASE)
    if remove_match:
        # Parse tool names (comma or space separated)
        tools_str = remove_match.group(1)
        tool_names = [t.strip() for t in re.split(r'[,\s]+', tools_str) if t.strip()]
        
        # Validate tools exist in current selection
        current_names = {t['name'] for t in current_tools}
        valid_tools = [t for t in tool_names if t in current_names]
        invalid_tools = [t for t in tool_names if t not in current_names]
        
        if invalid_tools:
            print(f"   ⚠️ Not in selection (ignored): {', '.join(invalid_tools)}")
        
        if valid_tools:
            print(f"   ➖ Removing: {', '.join(valid_tools)}")
            return {
                "feedback_action": "remove_tools",
                "tools_to_add": [],
                "tools_to_remove": valid_tools,
                "final_tools": None
            }
        else:
            print("   ⚠️ No valid tools to remove, approving")
            return {
                "feedback_action": "approved",
                "final_tools": current_tools,
                "tools_to_add": [],
                "tools_to_remove": []
            }
    
    # Unknown feedback - default to approve
    print(f"   ⚠️ Unknown feedback format, approving by default")
    return {
        "feedback_action": "approved",
        "final_tools": current_tools,
        "tools_to_add": [],
        "tools_to_remove": []
    }
