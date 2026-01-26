"""
System Prompt Templates for Single-Tool Architecture
=====================================================

These are optimized for 3B models where:
- Model learns from examples, not from reading long instructions
- Concise prompts work better than detailed ones
- Consistency between training and inference is critical
"""

def generate_prompt_for_3b(
    problem_statement: str,
    tools: list,
    include_clarification_hint: bool = True
) -> str:
    """
    Generate a CONCISE system prompt optimized for 3B models.
    
    This is the recommended prompt for fine-tuning Qwen 3B.
    ~150-200 tokens to leave room for conversation.
    
    Args:
        problem_statement: What the agent does
        tools: List of tool dicts or ToolSchema objects
        include_clarification_hint: Whether to mention asking for missing info
    
    Returns:
        Concise system prompt string
    """
    # Extract tool names and brief descriptions
    tool_lines = []
    for t in tools:
        if isinstance(t, dict):
            name = t.get('name', '')
            desc = t.get('description', '')
        else:
            name = getattr(t, 'name', '')
            desc = getattr(t, 'description', '')
        
        # Keep description short (first sentence only)
        if '. ' in desc:
            desc = desc.split('. ')[0]
        tool_lines.append(f"- {name}: {desc}")
    
    tools_section = "\n".join(tool_lines)
    
    # Build concise prompt
    clarification_line = ""
    if include_clarification_hint:
        clarification_line = "\nIf information is missing, ask before acting."
    
    prompt = f"""You are an assistant for: {problem_statement}

Tools:
{tools_section}

To use a tool:
<tool_call>
{{"tool": "name", "parameters": {{"key": "value"}}}}
</tool_call>{clarification_line}"""
    
    return prompt


def generate_prompt_for_7b_plus(
    problem_statement: str,
    tools: list,
) -> str:
    """
    Generate a DETAILED system prompt for 7B+ models.
    
    Larger models can understand and follow complex instructions.
    Use this for Llama 7B, Mistral 7B, etc.
    
    Args:
        problem_statement: What the agent does
        tools: List of tool dicts or ToolSchema objects
    
    Returns:
        Detailed system prompt string
    """
    # Build detailed tool documentation
    tool_docs = []
    for t in tools:
        if isinstance(t, dict):
            name = t.get('name', '')
            desc = t.get('description', '')
            params = t.get('parameters', {})
            required = t.get('required_params', [])
        else:
            name = getattr(t, 'name', '')
            desc = getattr(t, 'description', '')
            params = getattr(t, 'parameters', {})
            required = getattr(t, 'required_params', [])
        
        # Build parameter list
        param_lines = []
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict):
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = "required" if pname in required else "optional"
                param_lines.append(f"    - {pname} ({ptype}, {req}): {pdesc}")
        
        params_text = "\n".join(param_lines) if param_lines else "    (no parameters)"
        
        tool_doc = f"""**{name}**
  {desc}
  Parameters:
{params_text}"""
        tool_docs.append(tool_doc)
    
    tools_section = "\n\n".join(tool_docs)
    
    prompt = f"""You are an assistant that helps users with: {problem_statement}

## Tools

{tools_section}

## Response Format

For each user message, do ONE of the following:

1. **EXECUTE** - If you have all required info:
<tool_call>
{{"tool": "tool-name", "parameters": {{"param": "value"}}}}
</tool_call>

2. **CLARIFY** - If required info is missing:
Ask specifically what you need.

3. **RESPOND** - If no tool is needed:
Reply naturally (greetings, thanks, out-of-scope).

## Rules
- ONE tool call per message
- Never guess missing information
- Use exact tool names only"""

    return prompt


# Default for the SDK (3B models)
def get_default_prompt(problem_statement: str, tools: list) -> str:
    """
    Get the default system prompt for OnsetLab.
    
    Currently optimized for 3B models (concise).
    """
    return generate_prompt_for_3b(problem_statement, tools)


# Token estimates
PROMPT_TOKEN_ESTIMATES = {
    "3b_concise": 150,   # ~150 tokens
    "7b_detailed": 500,  # ~500 tokens
}
