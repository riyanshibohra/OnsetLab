"""
Skill Generator Node

Generates MCP server-specific skills that guide training data generation.

Input: Tool schemas from registry
Output: full_skill (for data gen) + condensed_rules (for system prompt)
"""

import json
import os
from typing import Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
    """Generates skills for MCP servers."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, server_name: str, server_description: str, tools: list) -> Tuple[str, str]:
        """
        Generate skill for an MCP server.
        
        Args:
            server_name: Name of the MCP server
            server_description: Description of what the server does
            tools: List of tool schemas from registry
            
        Returns:
            Tuple of (full_skill, condensed_rules)
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


def generate_skill_from_registry(registry_path: str) -> Tuple[str, str]:
    """
    Load a registry file and generate skill.
    
    Args:
        registry_path: Path to the registry JSON file
        
    Returns:
        Tuple of (full_skill, condensed_rules)
    """
    with open(registry_path, 'r') as f:
        data = json.load(f)
    
    server_name = data.get("name", "Unknown Server")
    server_description = data.get("description", "")
    tools = data.get("tools", [])
    
    generator = SkillGenerator()
    return generator.generate(server_name, server_description, tools)


# For use as a LangGraph node
def generate_skill(state: dict) -> dict:
    """
    LangGraph node that generates skill for approved tools.
    
    Expects state to have:
        - final_tools: list (user-approved tools)
        - problem_statement: str
        - identified_services: list[str]
        
    Adds to state:
        - full_skill: str (detailed skill for data generation)
        - condensed_rules: str (short rules for system prompt)
    """
    final_tools = state.get("final_tools", [])
    problem_statement = state.get("problem_statement", "")
    services = state.get("identified_services", [])
    
    if not final_tools:
        print("⚠️ No tools found, skipping skill generation")
        return {
            **state,
            "full_skill": "",
            "condensed_rules": ""
        }
    
    # Build server name from services
    server_name = " + ".join(s.replace("_", " ").title() for s in services) + " Agent"
    
    print(f"\n🧠 Generating skill for {len(final_tools)} tools...")
    
    generator = SkillGenerator()
    full_skill, condensed_rules = generator.generate(
        server_name=server_name,
        server_description=problem_statement,
        tools=final_tools
    )
    
    print(f"   ✅ Full skill: {len(full_skill)} chars")
    print(f"   ✅ Condensed rules: {len(condensed_rules)} chars (~{len(condensed_rules)//4} tokens)")
    
    return {
        **state,
        "full_skill": full_skill,
        "condensed_rules": condensed_rules
    }


# Alias for backwards compatibility
skill_generator_node = generate_skill


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        registry_path = sys.argv[1]
    else:
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "registry", "google_calendar.json"
        )
    
    print(f"Generating skill for: {registry_path}")
    print("=" * 60)
    
    full_skill, condensed_rules = generate_skill_from_registry(registry_path)
    
    print("FULL SKILL (first 2000 chars):")
    print("-" * 60)
    print(full_skill[:2000])
    print("\n...")
    
    print("\n" + "=" * 60)
    print("CONDENSED RULES:")
    print("-" * 60)
    print(condensed_rules)
