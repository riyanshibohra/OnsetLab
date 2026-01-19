"""
OnsetLab Data Generator
=======================
Generates high-quality training data for fine-tuning SLMs as tool-calling agents.

Key Learnings from MVP:
1. Single-turn format works best: user query → tool call (no multi-turn reasoning)
2. Consistent <tool_call> tags - never markdown code blocks
3. Real values only - no placeholders like {{date}} or <TODAY>
4. Varied phrasings - 5-10 ways to ask the same thing
5. Minimal system prompt - SLMs learn better from examples than instructions
"""

import json
import os
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

# Use shared schemas from utils
from ..utils.schemas import ToolSchema
from .prompt_generator import generate_minimal_prompt


@dataclass
class GeneratorConfig:
    """Configuration for the data generator."""
    problem_statement: str
    tools: list[ToolSchema]
    api_key: str
    api_provider: str = None  # Auto-detected from API key if not specified
    num_examples: int = 500
    output_path: str = "training_data.jsonl"
    
    # Generation settings
    examples_per_tool: int = 50  # Base examples per tool
    variation_multiplier: int = 3  # Phrasing variations per base example
    
    def __post_init__(self):
        """Auto-detect provider from API key format."""
        if self.api_provider is None:
            if self.api_key.startswith("sk-ant-"):
                self.api_provider = "anthropic"
            elif self.api_key.startswith("sk-"):
                self.api_provider = "openai"
            else:
                raise ValueError(
                    "Could not detect API provider from key. "
                    "Specify api_provider='openai' or 'anthropic'"
                )


class DataGenerator:
    """
    Generates training data for SLM fine-tuning.
    
    Flow:
    1. Generate system prompt from tools
    2. Generate diverse user queries for each tool
    3. Generate correct tool calls for each query
    4. Output as JSONL
    
    Usage:
        >>> from onsetlab.synthesis import DataGenerator, GeneratorConfig
        >>> from onsetlab.utils import ToolSchema
        >>> 
        >>> tools = [ToolSchema.from_mcp(t) for t in mcp_tools]
        >>> config = GeneratorConfig(
        ...     problem_statement="Calendar assistant",
        ...     tools=tools,
        ...     api_key="sk-..."
        ... )
        >>> generator = DataGenerator(config)
        >>> generator.run()
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client = self._init_client()
        self.system_prompt = None
        self.examples = []
    
    def _init_client(self):
        """Initialize the LLM client."""
        if self.config.api_provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        elif self.config.api_provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {self.config.api_provider}")
    
    def _call_llm(self, system: str, user: str, max_tokens: int = 2000) -> str:
        """Call the LLM API."""
        if self.config.api_provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text
    
    def generate_system_prompt(self) -> str:
        """
        Generate a minimal, effective system prompt.
        
        Uses the shared prompt_generator for consistency.
        """
        self.system_prompt = generate_minimal_prompt(
            self.config.problem_statement,
            self.config.tools
        )
        return self.system_prompt
    
    def _generate_date_context(self) -> dict:
        """Generate realistic date context for examples."""
        today = datetime.now()
        return {
            "today": today.strftime("%Y-%m-%d"),
            "tomorrow": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            "next_week": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
            "year": today.year,
            "month": today.strftime("%B"),
            "day_of_week": today.strftime("%A"),
        }
    
    def _generate_queries_for_tool(self, tool: ToolSchema, count: int) -> list[dict]:
        """
        Generate diverse user queries for a specific tool.
        
        Returns list of {"query": str, "tool_call": dict}
        """
        # Batch into smaller chunks to avoid truncation (max 15 per call)
        BATCH_SIZE = 15
        all_examples = []
        
        remaining = count
        while remaining > 0:
            batch_count = min(BATCH_SIZE, remaining)
            batch_examples = self._generate_queries_batch(tool, batch_count)
            all_examples.extend(batch_examples)
            remaining -= batch_count
            
            # Show progress
            if len(all_examples) < count:
                print(f"      Generated {len(all_examples)}/{count}...")
        
        return all_examples
    
    def _generate_queries_batch(self, tool: ToolSchema, count: int) -> list[dict]:
        """Generate a small batch of queries for a tool."""
        date_ctx = self._generate_date_context()
        
        prompt = f"""Generate {count} realistic user queries that would require the "{tool.name}" tool.

Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2)}
Required parameters: {tool.required_params}

Today's date: {date_ctx['today']} ({date_ctx['day_of_week']})

CRITICAL RULES:
1. Each query must be natural and varied (casual, formal, brief, detailed)
2. Use REAL dates based on today being {date_ctx['today']}
3. NEVER use placeholders like {{date}}, <TODAY>, or [NAME]
4. Include realistic names, times, and details
5. Vary the complexity (simple vs detailed requests)

PARAMETER GUIDELINES:
- Study the parameter descriptions and use realistic, sensible default values
- For ID parameters: Use realistic alphanumeric IDs like "abc123", "item_001" - these represent IDs from previous operations
- For dates/times: Always use ISO format. Be precise with the dates based on today being {date_ctx['today']}
- For optional params: Include them in ~50% of examples to show variety
- Match the expected types from the schema (string, number, boolean, etc.)

For each query, also provide the correct tool call.

Output as JSON array:
[
  {{
    "query": "the user's natural language query",
    "tool_call": {{
      "tool": "{tool.name}",
      "parameters": {{...actual parameter values...}}
    }}
  }},
  ...
]

Generate exactly {count} examples. Output ONLY the JSON array, no explanation."""

        response = self._call_llm(
            system="You are a training data generator. Output only valid JSON.",
            user=prompt,
            max_tokens=2000  # Smaller for batched requests
        )
        
        # Parse JSON response
        try:
            examples = self._parse_json_response(response)
            return examples
        except Exception as e:
            print(f"      ⚠️ Failed to parse batch for {tool.name}: {e}")
            return []
    
    def _parse_json_response(self, response: str) -> list:
        """Parse JSON from LLM response, handling common issues."""
        clean = response.strip()
        
        # Remove markdown code blocks
        if clean.startswith("```"):
            # Find the actual JSON start
            lines = clean.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            clean = "\n".join(lines[start_idx:end_idx])
        
        # Try to find JSON array in response
        if "[" in clean:
            start = clean.find("[")
            # Find matching closing bracket
            depth = 0
            end = start
            for i in range(start, len(clean)):
                if clean[i] == "[":
                    depth += 1
                elif clean[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            clean = clean[start:end]
        
        return json.loads(clean)
    
    def _generate_no_tool_examples(self, count: int) -> list[dict]:
        """Generate examples that don't need a tool (greetings, thanks, etc)."""
        examples = [
            {"query": "Hello!", "response": "Hello! How can I help you today?"},
            {"query": "Thanks!", "response": "You're welcome! Let me know if you need anything else."},
            {"query": "Hi there", "response": "Hi! What would you like to do?"},
            {"query": "Thank you so much", "response": "Happy to help! Anything else?"},
            {"query": "Goodbye", "response": "Goodbye! Have a great day!"},
            {"query": "Hey", "response": "Hey! How can I assist you?"},
            {"query": "What can you do?", "response": f"I can help you with {self.config.problem_statement}. Just ask me what you need!"},
            {"query": "Help", "response": f"I'm here to help! I can {self.config.problem_statement}. What would you like to do?"},
        ]
        return examples[:count]
    
    def _generate_clarification_examples(self, count: int) -> list[dict]:
        """Generate examples where the model asks for clarification."""
        prompt = f"""Generate {count} examples where a user gives an incomplete request and the assistant asks for clarification.

Context: {self.config.problem_statement}
Available tools: {', '.join([t.name for t in self.config.tools])}

Output as JSON array:
[
  {{
    "query": "vague user request missing key info",
    "response": "polite request for the missing information"
  }},
  ...
]

Rules:
- Requests should be realistic but missing required info
- Responses should be helpful and specific about what's needed
- No tool calls in these examples

Output ONLY the JSON array."""

        response = self._call_llm(
            system="You are a training data generator. Output only valid JSON.",
            user=prompt,
            max_tokens=2000
        )
        
        try:
            return self._parse_json_response(response)
        except:
            return []
    
    def generate_all(self) -> list[dict]:
        """
        Generate the complete training dataset.
        
        Distribution (based on MVP learnings):
        - 80% tool calls (main focus)
        - 10% no-tool-needed (greetings, thanks)
        - 10% clarification needed
        """
        if not self.system_prompt:
            self.generate_system_prompt()
        
        total = self.config.num_examples
        tool_examples_count = int(total * 0.80)
        no_tool_count = int(total * 0.10)
        clarify_count = int(total * 0.10)
        
        # Calculate examples per tool
        examples_per_tool = tool_examples_count // len(self.config.tools)
        
        all_examples = []
        
        # Generate tool call examples
        print(f"\n🔧 Generating {tool_examples_count} tool call examples...")
        for tool in self.config.tools:
            print(f"   → {tool.name} ({examples_per_tool} examples)")
            tool_examples = self._generate_queries_for_tool(tool, examples_per_tool)
            
            for ex in tool_examples:
                all_examples.append({
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": ex["query"]},
                        {"role": "assistant", "content": f"<tool_call>\n{json.dumps(ex['tool_call'])}\n</tool_call>"}
                    ]
                })
        
        # Generate no-tool examples
        print(f"\n💬 Generating {no_tool_count} no-tool examples...")
        no_tool_examples = self._generate_no_tool_examples(no_tool_count)
        for ex in no_tool_examples:
            all_examples.append({
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ex["query"]},
                    {"role": "assistant", "content": ex["response"]}
                ]
            })
        
        # Generate clarification examples
        print(f"\n❓ Generating {clarify_count} clarification examples...")
        clarify_examples = self._generate_clarification_examples(clarify_count)
        for ex in clarify_examples:
            all_examples.append({
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ex["query"]},
                    {"role": "assistant", "content": ex["response"]}
                ]
            })
        
        # Shuffle to mix example types
        random.shuffle(all_examples)
        
        self.examples = all_examples
        return all_examples
    
    def save(self, path: Optional[str] = None) -> str:
        """Save training data to JSONL file."""
        output_path = path or self.config.output_path
        
        with open(output_path, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example) + "\n")
        
        print(f"\n✅ Saved {len(self.examples)} examples to {output_path}")
        return output_path
    
    def save_system_prompt(self, path: str = "system_prompt.txt") -> str:
        """Save the generated system prompt."""
        with open(path, 'w') as f:
            f.write(self.system_prompt)
        print(f"✅ Saved system prompt to {path}")
        return path
    
    def run(self) -> tuple[str, str]:
        """
        Full generation pipeline.
        
        Returns:
            (training_data_path, system_prompt_path)
        """
        print("=" * 60)
        print("🚀 OnsetLab Data Generator")
        print("=" * 60)
        print(f"\nProblem: {self.config.problem_statement}")
        print(f"Tools: {len(self.config.tools)}")
        print(f"Target examples: {self.config.num_examples}")
        
        # Step 1: Generate system prompt
        print("\n📝 Step 1: Generating system prompt...")
        self.generate_system_prompt()
        print(f"   Generated {len(self.system_prompt)} chars")
        
        # Step 2: Generate training examples
        print("\n📊 Step 2: Generating training examples...")
        self.generate_all()
        
        # Step 3: Save outputs
        print("\n💾 Step 3: Saving outputs...")
        data_path = self.save()
        prompt_path = self.save_system_prompt()
        
        print("\n" + "=" * 60)
        print("✅ Generation complete!")
        print(f"   Training data: {data_path}")
        print(f"   System prompt: {prompt_path}")
        print("=" * 60)
        
        return data_path, prompt_path


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_recommended_examples(num_tools: int) -> int:
    """
    Calculate recommended number of examples based on tool count.
    
    Formula:
    - Base: 150 examples (covers format, greetings, clarifications)
    - Per tool: 50 examples (parameter variations, phrasings)
    - Minimum: 200, Maximum: 600
    """
    base = 150
    per_tool = 50
    
    calculated = base + (num_tools * per_tool)
    
    minimum = 200
    maximum = 600
    
    return max(minimum, min(maximum, calculated))


def generate_training_data(
    problem_statement: str,
    tools: list[ToolSchema],
    api_key: str,
    num_examples: Optional[int] = None,
    output_path: str = "training_data.jsonl"
) -> tuple[str, str]:
    """
    Convenience function to generate training data.
    
    Args:
        problem_statement: What the agent should do
        tools: List of ToolSchema objects
        api_key: OpenAI or Anthropic API key
        num_examples: Number of examples (auto-calculated if not specified)
        output_path: Output file path
        
    Returns:
        (training_data_path, system_prompt_path)
    """
    if num_examples is None:
        num_examples = calculate_recommended_examples(len(tools))
    
    config = GeneratorConfig(
        problem_statement=problem_statement,
        tools=tools,
        api_key=api_key,
        num_examples=num_examples,
        output_path=output_path
    )
    
    generator = DataGenerator(config)
    return generator.run()
