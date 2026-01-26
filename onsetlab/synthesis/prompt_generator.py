"""
OnsetLab Prompt Generator
=========================
Generates system prompts for fine-tuned agents.

The system prompt instructs the model on:
- Its role and purpose
- Available tools and how to use them
- Output format for tool calls
- Behavioral guidelines

Can generate prompts using:
1. LLM-based generation (richer, more tailored prompts)
2. Template-based generation (no API needed, faster)
"""

import json
from typing import Optional
from dataclasses import dataclass

from ..utils.schemas import ToolSchema


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    
    # Style options
    verbose: bool = False           # Include detailed instructions
    include_examples: bool = True   # Include example tool calls
    safety_guidelines: bool = True  # Include safety/refusal instructions
    
    # Tool call format
    tool_call_format: str = "xml"   # "xml" (<tool_call>) or "json" (```json)
    
    # Tool detail level in system prompt
    # "names_only" = Just tool names (shortest, for many tools)
    # "minimal" = Names + one-line description (default, balanced)
    # "full" = Names + description + all params + required markers (verbose)
    tool_detail_level: str = "minimal"
    
    # Model settings (for LLM mode)
    openai_model: str = "gpt-4o-mini"  # Cheaper, still good
    anthropic_model: str = "claude-3-haiku-20240307"  # Cheaper, still good


class PromptGenerator:
    """
    Generates system prompts for fine-tuned agents.
    
    Usage:
        >>> from onsetlab.synthesis import PromptGenerator
        >>> from onsetlab.utils import ToolSchema
        >>> 
        >>> tools = [ToolSchema(name="list-events", ...)]
        >>> generator = PromptGenerator(api_key="sk-...")
        >>> prompt = generator.generate(
        ...     problem_statement="Calendar management assistant",
        ...     tools=tools
        ... )
    """
    
    def __init__(
        self,
        api_key: str,
        api_provider: Optional[str] = None,
        config: Optional[PromptConfig] = None
    ):
        """
        Initialize the prompt generator.
        
        Args:
            api_key: OpenAI or Anthropic API key
            api_provider: "openai" or "anthropic" (auto-detected if not specified)
            config: PromptConfig for customization
        """
        self.api_key = api_key
        self.api_provider = api_provider or self._detect_provider(api_key)
        self.config = config or PromptConfig()
        self.client = None  # Lazy initialization
    
    def _detect_provider(self, api_key: str) -> str:
        """Auto-detect API provider from key format."""
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        elif api_key.startswith("sk-"):
            return "openai"
        else:
            raise ValueError(
                "Could not detect API provider from key. "
                "Specify api_provider='openai' or 'anthropic'"
            )
    
    def _init_client(self):
        """Initialize the LLM client (lazy loading)."""
        if self.client is not None:
            return
        
        if self.api_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        elif self.api_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def _call_llm(self, system: str, user: str, max_tokens: int = 2000) -> str:
        """Call the LLM API (OpenAI or Anthropic)."""
        self._init_client()
        
        if self.api_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:  # anthropic
            response = self.client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text
    
    def generate(
        self,
        problem_statement: str,
        tools: list[ToolSchema],
        use_llm: bool = True
    ) -> str:
        """
        Generate a system prompt.
        
        Args:
            problem_statement: Description of what the agent should do
            tools: List of available tools
            use_llm: If True, use LLM for richer prompt. If False, use template.
            
        Returns:
            Generated system prompt string
        """
        if use_llm:
            return self._generate_with_llm(problem_statement, tools)
        else:
            return self._generate_from_template(problem_statement, tools)
    
    def _generate_with_llm(self, problem_statement: str, tools: list[ToolSchema]) -> str:
        """
        Generate comprehensive system prompt using LLM for small models.
        
        Updated to follow Single-Tool Architecture:
        - One tool call per message
        - Clarify when info is missing
        - No multi-step chaining
        """
        
        # Build detailed tool descriptions
        tool_info = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {},
                "required": tool.required_params
            }
            # Include parameter details with types and enums
            for pname, pinfo in tool.parameters.items():
                if isinstance(pinfo, dict):
                    param_desc = {
                        "type": pinfo.get("type", "string"),
                        "description": pinfo.get("description", ""),
                    }
                    if "enum" in pinfo:
                        param_desc["valid_values"] = pinfo["enum"]
                    tool_dict["parameters"][pname] = param_desc
                else:
                    tool_dict["parameters"][pname] = {"type": "string", "description": str(pinfo)}
            tool_info.append(tool_dict)
        
        llm_system = """You are an expert at writing system prompts for SMALL language models (3B parameters).

Small models need EXPLICIT, DETAILED instructions. The prompt must be comprehensive.

Your task is to create a production-ready system prompt following the SINGLE-TOOL ARCHITECTURE:
- The model calls ONE tool per user message (never chains multiple tools)
- If required info is missing, it asks for clarification instead of guessing
- Multi-step tasks become multi-turn conversations"""

        llm_user = f"""Create a comprehensive system prompt for a 3B parameter LLM with these tools.

PROBLEM STATEMENT:
{problem_statement}

AVAILABLE TOOLS:
{json.dumps(tool_info, indent=2)}

ARCHITECTURE RULES (CRITICAL):
- ONE tool call per user message maximum
- If required parameters are missing, ASK the user (don't guess)
- Never chain tools (e.g., don't call list then create in one turn)
- Guide users to provide complete information

The system prompt MUST include these sections:

1. **ROLE DEFINITION**
   - What the assistant does
   - What it CAN and CANNOT do

2. **TOOL DOCUMENTATION**
   - Each tool with description
   - Required vs optional parameters (clearly marked)
   - Valid values for enum parameters

3. **TOOL CALL FORMAT**
   - Exact JSON syntax: <tool_call>{{"tool": "name", "parameters": {{}}}}</tool_call>

4. **THREE RESPONSE TYPES**
   - EXECUTE: When all required params are available → call the tool
   - CLARIFY: When required params are missing → ask for them
   - RESPOND: When no tool needed → reply naturally (greetings, thanks, out-of-scope)

5. **CLARIFICATION PATTERN**
   - Acknowledge what the user wants
   - List specifically what's needed
   - Give an example format
   - Example: "I can create an issue! Please provide:\\n- **Repository** (e.g., 'owner/repo')\\n- **Title**"

6. **AFTER TOOL RESULTS**
   - Present data clearly with markdown
   - Use actual values from the result
   - Offer next steps if relevant

7. **IMPORTANT RULES**
   - "Call ONE tool per message (never chain)"
   - "NEVER guess parameter values - ask if unsure"
   - "Use ONLY the exact tool names listed"
   - "NEVER make up IDs, names, or values"

Output ONLY the system prompt text. No markdown code blocks, no explanations.
Target length: 600-900 tokens (focused and clear)."""

        return self._call_llm(llm_system, llm_user, max_tokens=1500)
    
    def _generate_from_template(self, problem_statement: str, tools: list[ToolSchema]) -> str:
        """Generate system prompt from template (no LLM needed)."""
        
        # Build tool list
        tool_list = ", ".join([t.name for t in tools])
        
        # Build detailed tool descriptions
        tool_descriptions = []
        for tool in tools:
            params = ", ".join(tool.parameters.keys()) if tool.parameters else "none"
            required = ", ".join(tool.required_params) if tool.required_params else "none"
            tool_descriptions.append(
                f"- **{tool.name}**: {tool.description}\n"
                f"  Parameters: {params}\n"
                f"  Required: {required}"
            )
        
        # Build example tool call
        example_tool = tools[0] if tools else None
        example_call = ""
        if example_tool and self.config.include_examples:
            example_params = {}
            for param_name, param_info in example_tool.parameters.items():
                param_type = param_info.get("type", "string")
                if param_type == "string":
                    example_params[param_name] = f"<{param_name}_value>"
                elif param_type == "number" or param_type == "integer":
                    example_params[param_name] = 0
                elif param_type == "boolean":
                    example_params[param_name] = True
                else:
                    example_params[param_name] = f"<{param_name}>"
            
            example_call = f"""
Example:
<tool_call>
{json.dumps({"tool": example_tool.name, "parameters": example_params}, indent=2)}
</tool_call>
"""
        
        # Safety guidelines
        safety = ""
        if self.config.safety_guidelines:
            safety = """
Safety:
- Never fabricate information - use tools to get real data
- If unsure, ask for clarification
- Refuse harmful or inappropriate requests politely
"""
        
        # Assemble prompt
        prompt = f"""You are an assistant that helps users with: {problem_statement}

Available tools:
{chr(10).join(tool_descriptions)}

When you need to use a tool, respond with EXACTLY this format:
<tool_call>
{{"tool": "tool-name", "parameters": {{"key": "value"}}}}
</tool_call>
{example_call}
Rules:
- Always use the exact tool names listed above
- Use ISO format for dates/times when applicable
- Provide all required parameters
- Be helpful and concise
{safety}"""
        
        return prompt.strip()


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_system_prompt(
    problem_statement: str,
    tools: list[ToolSchema],
    api_key: Optional[str] = None,
    use_llm: bool = False
) -> str:
    """
    Generate a system prompt (convenience function).
    
    Args:
        problem_statement: Description of what the agent should do
        tools: List of available tools
        api_key: LLM API key (required if use_llm=True)
        use_llm: If True, use LLM for richer prompt generation
        
    Returns:
        Generated system prompt string
        
    Example:
        >>> prompt = generate_system_prompt(
        ...     problem_statement="Calendar assistant",
        ...     tools=my_tools,
        ...     use_llm=False  # Fast, no API needed
        ... )
    """
    if use_llm and not api_key:
        raise ValueError("api_key required when use_llm=True")
    
    generator = PromptGenerator(api_key=api_key or "dummy-key")
    return generator.generate(problem_statement, tools, use_llm=use_llm)


def generate_minimal_prompt(
    problem_statement: str, 
    tools: list[ToolSchema],
    detail_level: str = "minimal"
) -> str:
    """
    Generate a system prompt (no LLM, no API key needed).
    
    DEPRECATED: Use generate_single_tool_prompt() instead.
    
    Optimized for SLM fine-tuning where the model learns from examples.
    
    Args:
        problem_statement: Description of what the agent should do
        tools: List of available tools
        detail_level: How much tool detail to include
            - "names_only": Just tool names (shortest)
            - "minimal": Names + one-line description (default)
            - "full": Names + description + params + required markers
        
    Returns:
        System prompt string
    """
    # Build tool descriptions based on detail level
    tool_descriptions = []
    
    if detail_level == "names_only":
        # Just list tool names
        tool_names = [tool.name for tool in tools]
        tool_section = "Available tools: " + ", ".join(tool_names)
        
    elif detail_level == "minimal":
        # Names + one-line description
        for tool in tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        tool_section = "Available tools:\n" + "\n".join(tool_descriptions)
        
    else:  # "full"
        # Full details with params
        for tool in tools:
            param_details = []
            for param_name, param_info in (tool.parameters or {}).items():
                param_type = param_info.get("type", "string") if isinstance(param_info, dict) else "string"
                is_required = param_name in (tool.required_params or [])
                req_marker = " (required)" if is_required else ""
                param_details.append(f"    - {param_name}: {param_type}{req_marker}")
            
            tool_entry = f"- {tool.name}: {tool.description}"
            if param_details:
                tool_entry += "\n" + "\n".join(param_details)
            tool_descriptions.append(tool_entry)
        tool_section = "Available tools:\n" + "\n".join(tool_descriptions)
    
    prompt = f"""You are an assistant that helps users with: {problem_statement}

{tool_section}

When you need to use a tool, respond with:
<tool_call>
{{"tool": "tool-name", "parameters": {{"key": "value"}}}}
</tool_call>

Rules:
- Use the exact tool names listed above
- For casual conversation, respond naturally without tools
- Be helpful and concise"""
    
    return prompt


# ============================================================================
# Single-Tool Architecture (v2.0)
# ============================================================================

def generate_single_tool_prompt(
    problem_statement: str,
    tools: list[ToolSchema],
) -> str:
    """
    Generate a system prompt optimized for single-tool architecture.
    
    This follows the Single-Tool Architecture spec:
    - One tool call per user message
    - Clarify when required parameters are missing
    - Never chain multiple tools
    
    Args:
        problem_statement: Description of what the agent should do
        tools: List of available tools
        
    Returns:
        System prompt string optimized for 3B models
    """
    # Build detailed tool documentation
    tool_docs = []
    for tool in tools:
        # Get required and optional params
        required = tool.required_params or []
        all_params = tool.parameters or {}
        
        # Build parameter list
        param_lines = []
        for pname, pinfo in all_params.items():
            if isinstance(pinfo, dict):
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                # Check for enum values
                if "enum" in pinfo:
                    enum_vals = ", ".join(pinfo["enum"])
                    param_lines.append(f"  - {pname} ({ptype}): {pdesc} [Valid: {enum_vals}]")
                else:
                    param_lines.append(f"  - {pname} ({ptype}): {pdesc}")
            else:
                param_lines.append(f"  - {pname}: {pinfo}")
        
        # Mark required params
        required_str = ", ".join(required) if required else "none"
        
        tool_doc = f"""**{tool.name}**
  {tool.description}
  Required: {required_str}
  Parameters:
{chr(10).join(param_lines) if param_lines else "  (none)"}"""
        
        tool_docs.append(tool_doc)
    
    tools_section = "\n\n".join(tool_docs)
    
    # Build the prompt following Single-Tool Architecture
    prompt = f"""You are an assistant that helps users with: {problem_statement}

## Available Tools

{tools_section}

## How to Respond

For each user message, do exactly ONE of the following:

### 1. EXECUTE (if you have all required parameters)
Call the tool with this exact format:
<tool_call>
{{"tool": "tool-name", "parameters": {{"param": "value"}}}}
</tool_call>

### 2. CLARIFY (if required parameters are missing)
Ask the user for the missing information:
- Acknowledge what they want to do
- List specifically what you need
- Give an example if helpful

Example clarification:
"I can create an issue for you! Please provide:
- **Repository** (e.g., 'owner/repo-name')
- **Title** for the issue"

### 3. RESPOND (if no tool is needed)
Reply naturally for:
- Greetings and thanks
- Questions about your capabilities
- Requests outside your scope

## Important Rules

1. **ONE action per message** - Never chain multiple tool calls
2. **Never guess** - If you don't have required info, ask for it
3. **Use exact tool names** - Only use tools listed above
4. **No made-up data** - Never invent IDs, names, or values
5. **Be helpful** - Guide users on how to provide the right info

## After Tool Results

When you receive tool results:
- Present the data clearly using markdown
- Reference actual values from the result
- Offer to help with next steps if relevant"""

    return prompt


def get_clarification_examples(tools: list[ToolSchema]) -> list[dict]:
    """
    Generate clarification example patterns for each tool.
    
    Returns a list of example clarifications that can be used
    in training data generation.
    
    Args:
        tools: List of available tools
        
    Returns:
        List of {"tool": str, "missing": list, "clarification": str} dicts
    """
    examples = []
    
    for tool in tools:
        required = tool.required_params or []
        
        if not required:
            # No required params = no clarification needed
            continue
        
        # Build clarification message
        param_bullets = []
        for param in required:
            # Get description from tool parameters
            param_info = (tool.parameters or {}).get(param, {})
            if isinstance(param_info, dict):
                pdesc = param_info.get("description", param)
            else:
                pdesc = param
            param_bullets.append(f"- **{param}**: {pdesc}")
        
        clarification = f"I can help with that! Please provide:\n" + "\n".join(param_bullets)
        
        examples.append({
            "tool": tool.name,
            "required_params": required,
            "clarification_template": clarification
        })
    
    return examples
