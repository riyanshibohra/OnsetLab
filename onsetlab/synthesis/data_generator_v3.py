"""
Batched Data Generator v3
=========================
Efficient training data generation with batching.

Key features:
- 10x fewer API calls via batching (10 examples per call)
- Flat tool list (no server context needed)
- Single-tool, multi-tool, and edge case coverage
- Train/val/test split
"""

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


@dataclass
class BatchGenConfig:
    """Configuration for batched data generation."""
    
    # Target counts
    total_examples: int = 500         # Total examples to generate
    batch_size: int = 10              # Examples per LLM call
    tools_per_batch: int = 5          # Tools included in each prompt
    
    # Distribution (must sum to 1.0)
    single_tool_ratio: float = 0.80   # 80% single-tool examples
    multi_tool_ratio: float = 0.15    # 15% multi-tool (2-3 tools chained)
    edge_case_ratio: float = 0.05     # 5% edge cases
    
    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # API settings
    llm_model: str = "gpt-4o"
    max_retries: int = 3
    
    def get_targets(self) -> dict:
        """Calculate target counts for each phase."""
        return {
            "single_tool": int(self.total_examples * self.single_tool_ratio),
            "multi_tool": int(self.total_examples * self.multi_tool_ratio),
            "edge_cases": int(self.total_examples * self.edge_case_ratio),
        }


class BatchedDataGenerator:
    """
    Efficient batched data generator.
    
    Uses 1 LLM call to generate 10 examples instead of 10 calls.
    Reduces API usage by ~10x while maintaining quality.
    """
    
    def __init__(
        self,
        tools: list,
        problem_statement: str,
        api_key: str,
        config: BatchGenConfig = None,
        system_prompt: str = None
    ):
        self.tools = tools  # Flat list of all tools
        self.problem_statement = problem_statement
        self.config = config or BatchGenConfig()
        self.system_prompt = system_prompt  # Will be included in training examples
        
        # Build tool name lookup for validation
        self.tool_name_set = set()
        self.tool_name_map = {}  # normalized -> actual
        for t in tools:
            name = t.name if hasattr(t, 'name') else t.get('name', '')
            self.tool_name_set.add(name)
            # Map normalized versions
            normalized = name.lower().replace("-", "_").replace(" ", "_")
            self.tool_name_map[normalized] = name
            self.tool_name_map[name.lower()] = name
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        
        # Stats tracking
        self.stats = {
            "api_calls": 0,
            "examples_generated": 0,
            "failed_batches": 0,
            "fixed_tool_names": 0,
            "skipped_invalid": 0,
        }
    
    def generate(self) -> dict:
        """
        Generate training data in batches.
        
        Returns:
            dict with 'train', 'validation', 'test' lists
        """
        print(f"\n{'='*60}")
        print(f"🚀 Batched Data Generator v3")
        print(f"{'='*60}")
        print(f"Tools: {len(self.tools)}")
        print(f"Target examples: {self.config.total_examples}")
        print(f"Batch size: {self.config.batch_size} examples/call")
        print(f"Expected API calls: ~{self.config.total_examples // self.config.batch_size}")
        print(f"{'='*60}\n")
        
        all_examples = []
        targets = self.config.get_targets()
        
        # Phase 1: Single-tool examples (80%)
        print(f"📝 Phase 1: Single-tool examples ({targets['single_tool']} target)")
        single_examples = self._generate_single_tool_batches(targets['single_tool'])
        all_examples.extend(single_examples)
        print(f"   ✅ Generated {len(single_examples)} single-tool examples\n")
        
        # Phase 2: Multi-tool examples (15%)
        print(f"📝 Phase 2: Multi-tool examples ({targets['multi_tool']} target)")
        multi_examples = self._generate_multi_tool_batches(targets['multi_tool'])
        all_examples.extend(multi_examples)
        print(f"   ✅ Generated {len(multi_examples)} multi-tool examples\n")
        
        # Phase 3: Edge cases (5%)
        print(f"📝 Phase 3: Edge cases ({targets['edge_cases']} target)")
        edge_examples = self._generate_edge_cases(targets['edge_cases'])
        all_examples.extend(edge_examples)
        print(f"   ✅ Generated {len(edge_examples)} edge case examples\n")
        
        # Shuffle and split
        random.shuffle(all_examples)
        
        # Convert to chat format for training
        all_examples = [self._to_chat_format(ex) for ex in all_examples]
        
        datasets = self._split_data(all_examples)
        
        # Print summary
        print(f"{'='*60}")
        print(f"📊 Generation Complete!")
        print(f"{'='*60}")
        print(f"Total examples: {len(all_examples)}")
        print(f"  - Train: {len(datasets['train'])}")
        print(f"  - Validation: {len(datasets['validation'])}")
        print(f"  - Test: {len(datasets['test'])}")
        print(f"API calls made: {self.stats['api_calls']}")
        print(f"Failed batches: {self.stats['failed_batches']}")
        if self.stats.get('fixed_tool_names', 0) > 0:
            print(f"Fixed tool names: {self.stats['fixed_tool_names']}")
        if self.stats.get('skipped_invalid', 0) > 0:
            print(f"Skipped invalid: {self.stats['skipped_invalid']}")
        print(f"{'='*60}\n")
        
        return datasets
    
    def _generate_single_tool_batches(self, target: int) -> list:
        """Generate single-tool examples in batches."""
        examples = []
        tools_list = list(self.tools)  # Copy for shuffling
        
        while len(examples) < target:
            # Shuffle and pick tools for this batch
            random.shuffle(tools_list)
            batch_tools = tools_list[:min(self.config.tools_per_batch, len(tools_list))]
            
            # Generate batch
            batch_examples = self._call_llm_single_tool_batch(batch_tools)
            examples.extend(batch_examples)
            
            # Progress
            if len(examples) % 50 == 0:
                print(f"   ... {len(examples)}/{target} examples")
        
        return examples[:target]
    
    def _call_llm_single_tool_batch(self, tools: list) -> list:
        """Single LLM call → multiple single-tool examples."""
        tools_desc = self._format_tools(tools)
        tool_names = [t.name if hasattr(t, 'name') else t.get('name', '') for t in tools]
        
        # Build dynamic example using first tool
        first_tool = tools[0]
        first_name = first_tool.name if hasattr(first_tool, 'name') else first_tool.get('name', '')
        first_params = first_tool.parameters if hasattr(first_tool, 'parameters') else first_tool.get('parameters', {})
        example_params = {k: f"example_{k}_value" for k in list(first_params.keys())[:2]}
        
        prompt = f"""Generate {self.config.batch_size} training examples for an AI assistant.

Context: {self.problem_statement}

EXACT TOOL NAMES YOU MUST USE (copy these exactly, no variations):
{tool_names}

Tool details:
{tools_desc}

CRITICAL RULES:
1. Tool name MUST be EXACTLY one from the list above - copy it character-by-character!
2. Use REAL values, NOT placeholders like {{{{date}}}}, <NAME>, [USER], etc.
3. Good parameter values: "2026-01-20", "john@example.com", "Project Meeting", 42, true
4. Bad parameter values: "{{{{date}}}}", "<email>", "[title]", "YOUR_VALUE_HERE"

Output format (JSON array):
[
  {{"query": "natural user request", "tool": "{first_name}", "parameters": {json.dumps(example_params)}}},
  ...
]

Generate {self.config.batch_size} diverse examples. Use realistic values for ALL parameters:"""

        return self._call_llm(prompt)
    
    def _generate_multi_tool_batches(self, target: int) -> list:
        """Generate multi-tool chain examples."""
        examples = []
        
        while len(examples) < target:
            # Pick 2-3 random tools for chaining
            num_tools = random.choice([2, 3])
            chain_tools = random.sample(self.tools, min(num_tools, len(self.tools)))
            
            batch_examples = self._call_llm_multi_tool_batch(chain_tools)
            examples.extend(batch_examples)
        
        return examples[:target]
    
    def _call_llm_multi_tool_batch(self, tools: list) -> list:
        """Generate multi-step examples that chain multiple tools."""
        tools_desc = self._format_tools(tools)
        tool_names = [t.name if hasattr(t, 'name') else t.get('name', '') for t in tools]
        
        # Build dynamic example using actual tools
        example_tools = tool_names[:2] if len(tool_names) >= 2 else tool_names
        
        prompt = f"""Generate {min(5, self.config.batch_size)} multi-step examples requiring MULTIPLE tools.

Context: {self.problem_statement}

EXACT TOOL NAMES (copy these exactly):
{tool_names}

Tool details:
{tools_desc}

CRITICAL RULES:
1. Tool names MUST be EXACTLY from the list above - copy character-by-character!
2. Use REAL values: "2026-01-20", "john@company.com", "Team Standup", 42
3. NO placeholders like {{{{date}}}}, <name>, [value], YOUR_VALUE_HERE

Output format:
[
  {{
    "query": "user request requiring multiple steps",
    "tool_calls": [
      {{"tool": "{example_tools[0]}", "parameters": {{}}}},
      {{"tool": "{example_tools[-1]}", "parameters": {{}}}}
    ]
  }}
]

Generate {min(5, self.config.batch_size)} multi-step examples with realistic parameter values:"""

        return self._call_llm(prompt)
    
    def _generate_edge_cases(self, target: int) -> list:
        """Generate edge case examples."""
        examples = []
        edge_types = ["out_of_scope", "ambiguous", "missing_params"]
        
        for edge_type in edge_types:
            count = target // len(edge_types)
            batch_examples = self._call_llm_edge_cases(edge_type, count)
            examples.extend(batch_examples)
        
        return examples[:target]
    
    def _call_llm_edge_cases(self, edge_type: str, count: int) -> list:
        """Generate edge case examples of a specific type."""
        sample_tools = random.sample(self.tools, min(5, len(self.tools)))
        tool_names = [t.name if hasattr(t, 'name') else t.get('name', '') for t in sample_tools]
        
        # Get a sample tool for context
        sample_tool = sample_tools[0]
        sample_name = sample_tool.name if hasattr(sample_tool, 'name') else sample_tool.get('name', '')
        sample_desc = sample_tool.description if hasattr(sample_tool, 'description') else sample_tool.get('description', '')
        
        base_context = f"""Context: {self.problem_statement}
Available tools: {tool_names}

IMPORTANT: Use realistic text, NO placeholders like {{{{x}}}}, <value>, [name]."""
        
        if edge_type == "out_of_scope":
            prompt = f"""{base_context}

Generate {count} examples where user asks for something the assistant CANNOT do.
The assistant should politely decline and mention what it CAN help with (using the available tools).

Output format:
[{{"query": "request for unsupported feature", "response": "polite decline mentioning available capabilities", "tool": null}}]

Generate {count} out-of-scope examples:"""
        
        elif edge_type == "ambiguous":
            prompt = f"""{base_context}

Generate {count} examples where the user's request is VAGUE and needs clarification.
The assistant should ask a clarifying question related to the available tools.

Output format:
[{{"query": "vague request", "response": "clarifying question", "tool": null}}]

Generate {count} ambiguous examples:"""
        
        else:  # missing_params
            prompt = f"""{base_context}

Generate {count} examples where user wants to use a tool but is missing required info.
Example tool: {sample_name} - {sample_desc}

Output format:
[{{"query": "incomplete request for {sample_name}", "response": "question asking for missing details", "tool": null}}]

Generate {count} missing-info examples:"""
        
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> list:
        """Make LLM call with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                self.stats["api_calls"] += 1
                
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=3000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON from response
                examples = self._parse_json_response(content)
                
                # Validate and fix tool names
                valid_examples = []
                for ex in examples:
                    fixed_ex = self._fix_tool_name(ex)
                    if fixed_ex:
                        valid_examples.append(fixed_ex)
                
                self.stats["examples_generated"] += len(valid_examples)
                return valid_examples
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    self.stats["failed_batches"] += 1
                    print(f"   ⚠️ Batch failed after {self.config.max_retries} attempts: {e}")
                    return []
        
        return []
    
    def _parse_json_response(self, content: str) -> list:
        """Extract JSON array from LLM response."""
        # Try direct parse
        try:
            return json.loads(content)
        except:
            pass
        
        # Try to find JSON array in response
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        # Try removing markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        try:
            return json.loads(content)
        except:
            return []
    
    def _has_placeholder(self, text: str) -> bool:
        """Check if text contains placeholder patterns."""
        if not isinstance(text, str):
            return False
        patterns = [
            r'\{\{[^{}]+\}\}',        # {{date}}, {{time}}
            r'<[A-Z][A-Z_]{2,}>',     # <TODAY>, <USER_NAME>
            r'\[[A-Z][A-Z_]*\]',      # [NAME], [DATE]
            r'<[a-z_]+_placeholder>', # <any_placeholder>
            r'INSERT_\w+_HERE',       # INSERT_VALUE_HERE
            r'YOUR_\w+_HERE',         # YOUR_EMAIL_HERE
            r'PLACEHOLDER',           # literal PLACEHOLDER
            r'<[a-z_]+>',             # <email>, <date>
            r'\{[a-z_]+\}',           # {date}, {time} - single braces
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _example_has_placeholder(self, example: dict) -> bool:
        """Check if any part of the example has placeholders."""
        # Check query
        if self._has_placeholder(example.get("query", "")):
            return True
        # Check response
        if self._has_placeholder(example.get("response", "")):
            return True
        # Check parameters
        params = example.get("parameters", {})
        for value in params.values():
            if self._has_placeholder(str(value)):
                return True
        # Check tool_calls
        for call in example.get("tool_calls", []):
            for value in call.get("parameters", {}).values():
                if self._has_placeholder(str(value)):
                    return True
        return False
    
    def _fix_tool_name(self, example: dict) -> Optional[dict]:
        """
        Validate and fix tool name in example.
        Returns None if tool is invalid and can't be fixed.
        """
        # Filter out examples with placeholders
        if self._example_has_placeholder(example):
            self.stats["skipped_invalid"] = self.stats.get("skipped_invalid", 0) + 1
            return None
        
        # Edge cases with no tool are valid
        if example.get("tool") is None:
            return example
        
        tool_name = example.get("tool")
        
        # Check if tool exists directly
        if tool_name in self.tool_name_set:
            return example
        
        # Try normalized matching (case-insensitive, underscore/hyphen agnostic)
        normalized = tool_name.lower().replace("-", "_").replace(" ", "_")
        matched = self.tool_name_map.get(normalized)
        
        if matched:
            # Fix the tool name
            example["tool"] = matched
            self.stats["fixed_tool_names"] = self.stats.get("fixed_tool_names", 0) + 1
            return example
        
        # Multi-tool examples
        if "tool_calls" in example:
            fixed_calls = []
            for call in example["tool_calls"]:
                call_name = call.get("tool")
                if call_name in self.tool_name_set:
                    fixed_calls.append(call)
                else:
                    normalized = call_name.lower().replace("-", "_").replace(" ", "_")
                    matched = self.tool_name_map.get(normalized)
                    if matched:
                        call["tool"] = matched
                        self.stats["fixed_tool_names"] = self.stats.get("fixed_tool_names", 0) + 1
                        fixed_calls.append(call)
                    # Skip invalid tool calls
            
            if fixed_calls:
                example["tool_calls"] = fixed_calls
                return example
            
            self.stats["skipped_invalid"] = self.stats.get("skipped_invalid", 0) + 1
            return None
        
        # Tool name doesn't match anything - skip this example
        self.stats["skipped_invalid"] = self.stats.get("skipped_invalid", 0) + 1
        return None
    
    def _format_tools(self, tools: list) -> str:
        """Format tools list for prompt."""
        lines = []
        for t in tools:
            params = t.parameters if hasattr(t, 'parameters') else {}
            required = t.required_params if hasattr(t, 'required_params') else []
            lines.append(f"• {t.name}: {t.description}")
            lines.append(f"  Parameters: {json.dumps(params)}")
            if required:
                lines.append(f"  Required: {required}")
        return "\n".join(lines)
    
    def _split_data(self, examples: list) -> dict:
        """Split examples into train/val/test."""
        n = len(examples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        return {
            "train": examples[:train_end],
            "validation": examples[train_end:val_end],
            "test": examples[val_end:]
        }
    
    def _to_chat_format(self, example: dict) -> dict:
        """
        Convert simple format to chat format for training.
        
        Input format (simple):
            {"query": "...", "tool": "...", "parameters": {...}}
            or
            {"query": "...", "tool_calls": [...]}
            or
            {"query": "...", "response": "...", "tool": null}  # edge cases
        
        Output format (chat):
            {"messages": [
                {"role": "system", "content": "..."},  # <-- CRITICAL: Include system prompt!
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "<tool_call>...</tool_call>"}
            ]}
        """
        messages = []
        
        # System prompt (CRITICAL for training!)
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        else:
            # Generate minimal system prompt if not provided
            messages.append({"role": "system", "content": self._generate_minimal_system_prompt()})
        
        # User message
        query = example.get("query", "")
        messages.append({"role": "user", "content": query})
        
        # Assistant response
        if example.get("tool") is None and example.get("response"):
            # Edge case: no tool call, just a response
            messages.append({"role": "assistant", "content": example["response"]})
        elif "tool_calls" in example:
            # Multi-tool: chain of tool calls
            tool_calls = example["tool_calls"]
            # For simplicity, output just the first tool call in training
            # (multi-turn is handled separately)
            if tool_calls:
                first_call = tool_calls[0]
                tool_json = json.dumps({
                    "tool": first_call.get("tool"),
                    "parameters": first_call.get("parameters", {})
                })
                messages.append({
                    "role": "assistant",
                    "content": f"<tool_call>\n{tool_json}\n</tool_call>"
                })
        elif "tool" in example:
            # Single tool call
            tool_json = json.dumps({
                "tool": example.get("tool"),
                "parameters": example.get("parameters", {})
            })
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{tool_json}\n</tool_call>"
            })
        else:
            # Fallback: just echo the query (shouldn't happen)
            messages.append({"role": "assistant", "content": "I can help with that."})
        
        return {"messages": messages}
    
    def _generate_minimal_system_prompt(self) -> str:
        """Generate a minimal system prompt from tools (fallback)."""
        tool_descriptions = []
        for tool in self.tools:
            name = tool.name if hasattr(tool, 'name') else tool.get('name', '')
            desc = tool.description if hasattr(tool, 'description') else tool.get('description', '')
            params = tool.parameters if hasattr(tool, 'parameters') else tool.get('parameters', {})
            
            param_list = ", ".join(params.keys()) if params else "none"
            tool_descriptions.append(f"- {name}: {desc} (params: {param_list})")
        
        tools_text = "\n".join(tool_descriptions)
        
        return f"""You are an AI assistant that helps users with: {self.problem_statement}

Available tools:
{tools_text}

When the user requests an action, respond with a tool call in this format:
<tool_call>
{{"tool": "tool-name", "parameters": {{"key": "value"}}}}
</tool_call>

Be helpful and precise. Use the exact tool names provided."""

    def save(self, output_dir: str) -> dict:
        """Generate and save to files."""
        datasets = self.generate()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for split_name, examples in datasets.items():
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            paths[split_name] = str(file_path)
            print(f"💾 Saved {len(examples)} examples to {file_path}")
        
        return paths


# =============================================================================
# Convenience function
# =============================================================================

def generate_training_data_batched(
    tools: list,
    problem_statement: str,
    api_key: str,
    total_examples: int = 500,
    output_dir: str = None,
) -> dict:
    """
    Generate training data with efficient batching.
    
    Args:
        tools: List of ToolSchema objects (flat list from all servers)
        problem_statement: What the agent should do
        api_key: OpenAI API key
        total_examples: Target number of examples
        output_dir: Where to save (optional)
    
    Returns:
        dict with 'train', 'validation', 'test' lists
    """
    config = BatchGenConfig(total_examples=total_examples)
    generator = BatchedDataGenerator(tools, problem_statement, api_key, config)
    
    if output_dir:
        return generator.save(output_dir)
    else:
        return generator.generate()
