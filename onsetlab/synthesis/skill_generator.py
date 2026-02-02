"""
Skill Generator
================
Generates MCP server-specific skills that guide training data generation.

Part of the OnsetLab SDK - runs in Colab to create high-quality training data.
"""

import json
import os
from typing import Tuple
from openai import OpenAI


SKILL_GENERATION_PROMPT = """Analyze this MCP server and generate a comprehensive skill document.

SERVER: {server_name}
DESCRIPTION: {server_description}

TOOLS:
{tools_json}

Generate a detailed skill document with the following sections:

# {server_name} Skill

## Overview
2-3 sentences about what this server does and when users need it.

## Tools

For EACH tool, provide ALL of the following:

### [tool_name]

**Purpose:** One sentence description

**Triggers:** List 10+ natural language phrases that should invoke this tool
- "phrase 1"
- "phrase 2"
- ... (at least 10)

**Parameters:**

| Parameter | Type | Required | Format | Default | Notes |
|-----------|------|----------|--------|---------|-------|
| param_name | exact_type | Yes/No | EXACT format with example | default_value | common mistakes |

For OBJECT type parameters, show the EXACT nested structure:
```json
"param_name": {{
  "field1": "value",
  "field2": {{
    "nested": "value"
  }}
}}
```

**Correct Examples:** (5 complete examples with realistic values)

User: "realistic user message"
```
<tool_call>{{"tool": "tool_name", "parameters": {{"param1": "actual_value", ...}}}}</tool_call>
```

(repeat for 5 different scenarios)

**Wrong Examples:** (3 examples showing common mistakes)

WRONG: `"param": "string value"` 
WHY: param should be an object, not a string

(repeat for 3 different mistakes)

**Clarification:** When should the model ask for more info?
- If X is missing, ask: "specific question"

## Response Format

ALWAYS respond with:
```
<tool_call>{{"tool": "tool_name", "parameters": {{}}}}</tool_call>
```

## General Rules
- List 5 rules specific to this server
- Include default values to use
- Include format requirements

---

IMPORTANT:
- Be SPECIFIC to this server's tools and parameters
- Show EXACT formats (especially for object/nested parameters)
- Include REALISTIC example values (not placeholders)
- NO generic advice - everything should be server-specific

CRITICAL REQUIREMENTS:
1. Generate COMPLETE documentation for EVERY tool listed. Do NOT skip any.
2. Do NOT write "Continue similarly..." or "Repeat for other tools". Write out each tool fully.
3. For OBJECT type parameters (like "event"), show the EXACT nested JSON structure.
4. This is extremely important: document ALL tools completely.
"""


CONDENSE_PROMPT = """Condense this skill into a minimal system prompt for a fine-tuned model.

FULL SKILL:
{full_skill}

Create a condensed version with ONLY:
1. One-line description of what the assistant does
2. List of tools with one-line descriptions
3. Critical format rules (especially for complex object parameters)
4. Response format instruction

REQUIREMENTS:
- Maximum 300 tokens
- Include the MOST important format rules
- Skip examples (model learned from training)
- Be concise but precise

Output the condensed system prompt directly, no explanation.
"""


class SkillGenerator:
    """
    Generates skills for MCP servers.
    
    Usage:
        generator = SkillGenerator(api_key="sk-...")
        full_skill, condensed_rules = generator.generate(
            server_name="Calendar Agent",
            server_description="Manage calendar events",
            tools=[{...}, ...]
        )
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the skill generator.
        
        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            model: Model to use for skill generation (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, server_name: str, server_description: str, tools: list) -> Tuple[str, str]:
        """
        Generate skill for an MCP server.
        
        Args:
            server_name: Name of the MCP server
            server_description: Description of what the server does
            tools: List of tool schemas
            
        Returns:
            Tuple of (full_skill, condensed_rules)
            - full_skill: Detailed skill document for data generation
            - condensed_rules: Short system prompt for runtime
        """
        # Format tools for the prompt
        tools_json = json.dumps(tools, indent=2)
        
        # Generate full skill
        full_skill = self._generate_full_skill(server_name, server_description, tools_json)
        
        # Generate condensed rules
        condensed_rules = self._generate_condensed_rules(full_skill)
        
        return full_skill, condensed_rules
    
    def _generate_full_skill(self, server_name: str, server_description: str, tools_json: str) -> str:
        """Generate the full skill document."""
        prompt = SKILL_GENERATION_PROMPT.format(
            server_name=server_name,
            server_description=server_description,
            tools_json=tools_json
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a technical writer creating precise documentation for AI tool usage."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8000
        )
        
        return response.choices[0].message.content
    
    def _generate_condensed_rules(self, full_skill: str) -> str:
        """Condense the full skill into minimal rules for system prompt."""
        prompt = CONDENSE_PROMPT.format(full_skill=full_skill)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper model for condensing
            messages=[
                {"role": "system", "content": "You create concise system prompts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
