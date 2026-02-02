"""
System Prompt Templates for Tool Calling
=========================================

Supports multiple formats:
- ToolLLaMA: Pre-trained on ToolBench, uses specific format
- NexusRaven: Uses Python function signatures  
- Qwen: General purpose with custom format
"""

import json

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
    
    # Build concise but effective prompt
    prompt = f"""I help with: {problem_statement}

Tools:
{tools_section}

To use a tool:
<tool_call>
{{"tool": "name", "parameters": {{"key": "value"}}}}
</tool_call>

CRITICAL - When NOT to use tools:
- Greetings (hi, hey, hello): Just respond friendly - "Hey! How can I help?"
- Thanks (thanks, perfect, great): "You're welcome!" - no tool needed
- Questions about me: Explain what you can do - no tool needed
- If you can't do it: Say so nicely - no tool needed
- General chat: Respond normally - no tool needed

ONLY use tools when user explicitly asks to DO something with their data.

Rules:
- Use parameter values exactly as user says them
- If required info is missing: ask, don't guess"""
    
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


def generate_prompt_for_toolllama(
    problem_statement: str,
    tools: list,
) -> str:
    """
    Generate system prompt for ToolLLaMA format.
    
    ToolLLaMA was pre-trained on ToolBench with a specific format.
    Uses JSON tool definitions and expects specific response structure.
    
    Args:
        problem_statement: What the agent does
        tools: List of tool dicts or ToolSchema objects
    
    Returns:
        ToolLLaMA-compatible system prompt
    """
    # Convert tools to ToolBench API format
    tool_definitions = []
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
        
        # Build parameter schema
        param_schema = {}
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict):
                param_schema[pname] = {
                    "type": pinfo.get("type", "string"),
                    "description": pinfo.get("description", ""),
                }
                if "enum" in pinfo:
                    param_schema[pname]["enum"] = pinfo["enum"]
        
        tool_def = {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": param_schema,
                "required": required,
            }
        }
        tool_definitions.append(tool_def)
    
    tools_json = json.dumps(tool_definitions, indent=2)
    
    prompt = f"""You are an assistant for: {problem_statement}

You have access to the following tools:

{tools_json}

When you need to use a tool, respond with:
Action: tool_name
Action Input: {{"param": "value"}}

When you don't need a tool (greetings, questions, thanks), respond normally without Action/Action Input.

Think step by step before acting."""
    
    return prompt


def generate_prompt_for_nexusraven(
    problem_statement: str,
    tools: list,
) -> str:
    """
    Generate system prompt for NexusRaven format.
    
    NexusRaven uses Python function signatures with docstrings.
    
    Args:
        problem_statement: What the agent does
        tools: List of tool dicts or ToolSchema objects
    
    Returns:
        NexusRaven-compatible system prompt with function definitions
    """
    function_defs = []
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
        
        # Build Python function signature
        param_strs = []
        docstring_params = []
        for pname, pinfo in params.items():
            ptype = "str"
            pdesc = ""
            if isinstance(pinfo, dict):
                ptype_raw = pinfo.get("type", "string")
                ptype = {"string": "str", "number": "int", "integer": "int", "boolean": "bool", "array": "list"}.get(ptype_raw, "str")
                pdesc = pinfo.get("description", "")
            
            if pname in required:
                param_strs.append(f"{pname}: {ptype}")
            else:
                param_strs.append(f"{pname}: {ptype} = None")
            docstring_params.append(f"        {pname} ({ptype}): {pdesc}")
        
        params_str = ", ".join(param_strs)
        docstring_params_str = "\n".join(docstring_params)
        
        func_def = f'''def {name}({params_str}):
    """
    {desc}
    
    Args:
{docstring_params_str}
    """
    pass'''
        function_defs.append(func_def)
    
    functions_section = "\n\n".join(function_defs)
    
    prompt = f"""You are an assistant for: {problem_statement}

You have access to the following functions:

{functions_section}

When you need to call a function, use this format:
Call: function_name(param1="value1", param2="value2")

When no function is needed (greetings, thanks, questions about yourself), respond naturally without a Call."""
    
    return prompt


# Default for the SDK
def get_default_prompt(problem_statement: str, tools: list, model_format: str = "toolllama") -> str:
    """
    Get the default system prompt for OnsetLab.
    
    Args:
        problem_statement: What the agent does
        tools: List of tool schemas
        model_format: One of "toolllama", "nexusraven", "qwen", "qwen_detailed"
    
    Returns:
        System prompt string
    """
    if model_format == "toolllama":
        return generate_prompt_for_toolllama(problem_statement, tools)
    elif model_format == "nexusraven":
        return generate_prompt_for_nexusraven(problem_statement, tools)
    elif model_format == "qwen_detailed":
        return generate_prompt_for_7b_plus(problem_statement, tools)
    else:
        return generate_prompt_for_3b(problem_statement, tools)


# Token estimates
PROMPT_TOKEN_ESTIMATES = {
    "3b_concise": 150,   # ~150 tokens
    "7b_detailed": 500,  # ~500 tokens
    "toolllama": 400,    # ~400 tokens (with JSON)
    "nexusraven": 600,   # ~600 tokens (with function defs)
}
