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
        """Generate comprehensive system prompt using LLM for small models."""
        
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

Your task is to create a production-ready system prompt that will be embedded in a fine-tuned model.
The model will use tools via MCP (Model Context Protocol) to interact with external services."""

        llm_user = f"""Create a comprehensive system prompt for a 3B parameter LLM with these tools.

PROBLEM STATEMENT:
{problem_statement}

AVAILABLE TOOLS:
{json.dumps(tool_info, indent=2)}

The system prompt MUST include ALL of these sections:

1. **ROLE DEFINITION**
   - Clear statement of what the assistant does
   - Scope of capabilities (what it CAN and CANNOT do)

2. **TOOL DOCUMENTATION**
   - List each tool with description
   - Mark required vs optional parameters
   - Include valid values for enum parameters

3. **TOOL CALL FORMAT**
   - Exact JSON syntax inside <tool_call> tags
   - Example: <tool_call>{{"tool": "name", "parameters": {{}}}}</tool_call>

4. **USING TOOL RESULTS**
   - How to extract and present data from tool responses
   - Format responses nicely for the user (use markdown)
   - Reference actual data, don't summarize generically

5. **MULTI-STEP WORKFLOWS**
   - Example: "To post to Slack, first call slack_list_channels to find the channel ID, then call slack_post_message"
   - Show the discover → act → confirm pattern

6. **ERROR HANDLING**
   - "NEVER retry the same tool call with identical parameters"
   - "If a tool fails, explain the error to the user"
   - "Ask user for clarification if parameters are unclear"

7. **DATA INTEGRITY**
   - "Use ONLY actual data from tool results"
   - "NEVER hallucinate or make up IDs, names, or values"
   - "If you don't have data, call the appropriate discovery tool first"

8. **RESPONSE STYLE**
   - Keep Slack messages concise and professional
   - Format issue details with markdown
   - Be helpful but brief

Output ONLY the system prompt text. No markdown code blocks, no explanations.
Target length: 800-1200 tokens for comprehensive coverage."""

        return self._call_llm(llm_system, llm_user, max_tokens=2000)
    
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
