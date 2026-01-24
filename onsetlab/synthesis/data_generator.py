"""
Batched Data Generator
======================
Efficient training data generation with smart batching.

Key features:
- 10x fewer API calls via batching (10 examples per call)
- Flat tool list (no server context needed)
- Single-tool, multi-tool, and edge case coverage
- Stratified train/val/test split (ensures balanced categories)
- Supports both OpenAI and Anthropic APIs
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
    total_examples: int = 500         # Total examples to generate (or use calculate_optimal_examples)
    batch_size: int = 10              # Examples per LLM call
    tools_per_batch: int = 5          # Tools included in each prompt
    
    # Distribution (must sum to 1.0)
    # - More multi-tool examples for better chaining ability
    # - More edge cases for robust refusal/casual handling
    single_tool_ratio: float = 0.60   # 60% single-tool examples
    multi_tool_ratio: float = 0.25    # 25% multi-tool (2-3 tools chained)
    edge_case_ratio: float = 0.15     # 15% edge cases (refusals, casual, ambiguous)
    
    # Split ratios - more training data, less test (test is optional)
    train_ratio: float = 0.85         # 85% for training
    val_ratio: float = 0.10           # 10% for validation (early stopping)
    test_ratio: float = 0.05          # 5% for final evaluation (optional)
    
    # API settings - use best models for quality data generation
    openai_model: str = "gpt-4o"                    # Best OpenAI model
    anthropic_model: str = "claude-sonnet-4-20250514"  # Best Anthropic model
    max_retries: int = 3
    
    # Per-tool settings
    examples_per_tool: int = 30       # Recommended: 30-50 per tool
    min_examples_per_tool: int = 10   # Minimum examples any tool should have
    
    def get_targets(self) -> dict:
        """Calculate target counts for each phase."""
        return {
            "single_tool": int(self.total_examples * self.single_tool_ratio),
            "multi_tool": int(self.total_examples * self.multi_tool_ratio),
            "edge_cases": int(self.total_examples * self.edge_case_ratio),
        }
    
    @staticmethod
    def calculate_optimal_examples(
        num_tools: int, 
        examples_per_tool: int = 30,
        max_total: int = 1500,
        min_per_tool: int = 10
    ) -> int:
        """
        Calculate optimal total examples based on number of tools.
        
        Uses smart scaling:
        - Few tools (≤10): Full 30 examples per tool
        - Many tools (>10): Scales down per-tool to stay under max
        - Always respects minimum per-tool coverage
        
        Args:
            num_tools: Number of tools the agent will use
            examples_per_tool: Target examples per tool (default: 30)
            max_total: Maximum total examples (default: 1500, good for LoRA)
            min_per_tool: Minimum examples per tool (default: 10)
        
        Returns:
            Recommended total_examples value (capped at max_total)
        
        Examples:
            - 5 tools:  ~250 examples (30/tool)
            - 10 tools: ~500 examples (30/tool)
            - 15 tools: ~750 examples (30/tool)
            - 20 tools: ~1000 examples (30/tool)
            - 30 tools: ~1200 examples (24/tool, scaled down)
            - 50 tools: ~1500 examples (18/tool, capped)
        """
        # Calculate what we'd ideally want
        ideal_single_tool = num_tools * examples_per_tool
        
        # Multi-tool + edge cases add ~65% more (25% + 15% relative to 60%)
        ideal_total = int(ideal_single_tool * 1.65)
        
        # If under cap, use ideal
        if ideal_total <= max_total:
            # Round to nearest 50
            return max(200, ((ideal_total + 25) // 50) * 50)
        
        # Otherwise, scale down per-tool to fit under cap
        # Working backwards: total = single * 1.65, single = tools * per_tool
        # So: per_tool = total / (tools * 1.65)
        scaled_per_tool = max_total / (num_tools * 1.65)
        
        # Ensure at least minimum per tool
        actual_per_tool = max(min_per_tool, int(scaled_per_tool))
        
        # Recalculate with scaled value
        single_tool = num_tools * actual_per_tool
        total = int(single_tool * 1.65)
        
        # Cap at max_total
        total = min(total, max_total)
        
        # Warn if per-tool is below minimum (edge case with many tools)
        final_per_tool = int(total * 0.6 / num_tools)
        if final_per_tool < min_per_tool:
            print(f"   ⚠️ {num_tools} tools with cap {max_total} gives ~{final_per_tool}/tool.")
            print(f"   💡 Consider increasing max_total or reducing tools for better coverage.")
        
        # Round to nearest 50
        return max(200, ((total + 25) // 50) * 50)


class BatchedDataGenerator:
    """
    Efficient batched data generator for training data.
    
    Features:
    - Uses 1 LLM call to generate 10 examples (10x cheaper than individual calls)
    - Stratified splitting ensures balanced train/val/test sets
    - Auto-detects OpenAI vs Anthropic from API key format
    - Validates and auto-fixes tool names
    - Filters out placeholder values for quality data
    """
    
    def __init__(
        self,
        tools: list,
        problem_statement: str,
        api_key: str,
        config: BatchGenConfig = None,
        system_prompt: str = None,
        api_provider: str = None  # "openai" or "anthropic", auto-detected if None
    ):
        self.tools = tools  # Flat list of all tools
        self.problem_statement = problem_statement
        self.config = config or BatchGenConfig()
        self.system_prompt = system_prompt  # Will be included in training examples
        self.api_key = api_key
        
        # Auto-detect API provider from key format
        self.api_provider = api_provider or self._detect_provider(api_key)
        
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
        
        # Initialize LLM client (lazy - created on first call)
        self.client = None
        
        # Stats tracking
        self.stats = {
            "api_calls": 0,
            "examples_generated": 0,
            "failed_batches": 0,
            "fixed_tool_names": 0,
            "skipped_invalid": 0,
        }
    
    def _detect_provider(self, api_key: str) -> str:
        """Auto-detect API provider from key format."""
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        elif api_key.startswith("sk-"):
            return "openai"
        else:
            # Default to OpenAI if can't detect
            print("   ⚠️ Could not detect API provider from key format, defaulting to OpenAI")
            return "openai"
    
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
        else:  # anthropic
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(self) -> dict:
        """
        Generate training data in batches.
        
        Returns:
            dict with 'train', 'validation', 'test' lists
        """
        model = self.config.openai_model if self.api_provider == "openai" else self.config.anthropic_model
        
        print(f"\n{'='*60}")
        print(f"🚀 Batched Data Generator")
        print(f"{'='*60}")
        print(f"Provider: {self.api_provider.upper()} ({model})")
        print(f"Tools: {len(self.tools)}")
        print(f"Target examples: {self.config.total_examples}")
        print(f"  - Single-tool: {int(self.config.total_examples * self.config.single_tool_ratio)} ({int(self.config.single_tool_ratio * 100)}%)")
        print(f"  - Multi-tool: {int(self.config.total_examples * self.config.multi_tool_ratio)} ({int(self.config.multi_tool_ratio * 100)}%)")
        print(f"  - Edge cases: {int(self.config.total_examples * self.config.edge_case_ratio)} ({int(self.config.edge_case_ratio * 100)}%)")
        print(f"Batch size: {self.config.batch_size} examples/call")
        print(f"Expected API calls: ~{self.config.total_examples // self.config.batch_size}")
        print(f"{'='*60}\n")
        
        targets = self.config.get_targets()
        
        # Phase 1: Single-tool examples (60%)
        print(f"📝 Phase 1: Single-tool examples ({targets['single_tool']} target)")
        single_examples = self._generate_single_tool_batches(targets['single_tool'])
        print(f"   ✅ Generated {len(single_examples)} single-tool examples\n")
        
        # Phase 2: Multi-tool examples (25%)
        print(f"📝 Phase 2: Multi-tool examples ({targets['multi_tool']} target)")
        multi_examples = self._generate_multi_tool_batches(targets['multi_tool'])
        print(f"   ✅ Generated {len(multi_examples)} multi-tool examples\n")
        
        # Phase 3: Edge cases (15%) - includes refusals, clarifications, AND casual chat
        print(f"📝 Phase 3: Edge cases ({targets['edge_cases']} target)")
        edge_examples = self._generate_edge_cases(targets['edge_cases'])
        print(f"   ✅ Generated {len(edge_examples)} edge case examples\n")
        
        # Convert each category to chat format
        single_chat = [self._to_chat_format(ex) for ex in single_examples]
        multi_chat = [self._to_chat_format(ex) for ex in multi_examples]
        edge_chat = [self._to_chat_format(ex) for ex in edge_examples]
        
        # STRATIFIED SPLIT: Split each category proportionally, then combine
        # This ensures each split has examples from ALL categories
        datasets = self._stratified_split(single_chat, multi_chat, edge_chat)
        
        # Calculate totals
        total_examples = len(datasets['train']) + len(datasets['validation']) + len(datasets['test'])
        
        # Print summary
        print(f"{'='*60}")
        print(f"📊 Generation Complete!")
        print(f"{'='*60}")
        print(f"Total examples: {total_examples}")
        print(f"  By category:")
        print(f"    - Single-tool: {len(single_examples)}")
        print(f"    - Multi-tool: {len(multi_examples)}")
        print(f"    - Edge cases: {len(edge_examples)}")
        print(f"  By split (stratified):")
        print(f"    - Train: {len(datasets['train'])} ({int(self.config.train_ratio * 100)}%)")
        print(f"    - Validation: {len(datasets['validation'])} ({int(self.config.val_ratio * 100)}%)")
        print(f"    - Test: {len(datasets['test'])} ({int(self.config.test_ratio * 100)}%)")
        print(f"API calls made: {self.stats['api_calls']}")
        print(f"Failed batches: {self.stats['failed_batches']}")
        if self.stats.get('fixed_tool_names', 0) > 0:
            print(f"Fixed tool names: {self.stats['fixed_tool_names']}")
        if self.stats.get('skipped_invalid', 0) > 0:
            print(f"Skipped invalid: {self.stats['skipped_invalid']}")
        print(f"{'='*60}\n")
        
        return datasets
    
    def _generate_single_tool_batches(self, target: int) -> list:
        """
        Generate single-tool examples with BALANCED per-tool coverage.
        
        Uses round-robin to ensure each tool gets roughly equal examples.
        """
        examples = []
        num_tools = len(self.tools)
        
        # Calculate minimum examples per tool
        min_per_tool = max(self.config.min_examples_per_tool, target // num_tools)
        
        # Track how many examples each tool has
        tool_counts = {
            (t.name if hasattr(t, 'name') else t.get('name', '')): 0 
            for t in self.tools
        }
        
        # Round-robin through tools to ensure coverage
        tool_index = 0
        
        while len(examples) < target:
            # Pick tools that need more examples (round-robin style)
            # Start from current index and pick tools_per_batch tools
            batch_tools = []
            indices_tried = 0
            
            while len(batch_tools) < self.config.tools_per_batch and indices_tried < num_tools:
                tool = self.tools[(tool_index + indices_tried) % num_tools]
                tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', '')
                
                # Include tool if it needs more examples
                if tool_counts[tool_name] < min_per_tool or len(examples) >= target - self.config.batch_size:
                    batch_tools.append(tool)
                indices_tried += 1
            
            # If we couldn't find enough tools needing examples, just pick any
            if len(batch_tools) < self.config.tools_per_batch:
                remaining = [t for t in self.tools if t not in batch_tools]
                random.shuffle(remaining)
                batch_tools.extend(remaining[:self.config.tools_per_batch - len(batch_tools)])
            
            # Advance the round-robin index
            tool_index = (tool_index + self.config.tools_per_batch) % num_tools
            
            # Generate batch
            batch_examples = self._call_llm_single_tool_batch(batch_tools)
            
            # Update counts
            for ex in batch_examples:
                tool_name = ex.get('tool')
                if tool_name in tool_counts:
                    tool_counts[tool_name] += 1
            
            examples.extend(batch_examples)
            
            # Progress
            if len(examples) % 50 == 0:
                print(f"   ... {len(examples)}/{target} examples")
        
        # Log distribution
        counts = list(tool_counts.values())
        if counts:
            print(f"   📊 Per-tool distribution: min={min(counts)}, max={max(counts)}, avg={sum(counts)//len(counts)}")
        
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
5. Generate diverse, realistic examples that match what real users would ask
6. Include short direct queries AND longer natural language requests

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
        """
        Generate edge case examples for robust agent behavior.
        
        Categories:
        - out_of_scope: User asks for something the agent can't do
        - ambiguous: User request is vague, needs clarification
        - missing_params: User wants a tool but missing required info
        - casual_conversation: Friendly chat without needing tools
        """
        examples = []
        edge_types = ["out_of_scope", "ambiguous", "missing_params", "casual_conversation"]
        
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
        
        elif edge_type == "casual_conversation":
            prompt = f"""{base_context}

Generate {count} examples of CASUAL/FRIENDLY conversation that don't require any tool.
These are social interactions like greetings, thanks, reactions, small talk.
The assistant should respond naturally and warmly, optionally offering to help.

Examples of casual messages:
- "Thanks!", "Awesome!", "Perfect!", "Great job!"
- "Hey", "Hello", "Hi there"
- "That's exactly what I needed", "You're the best"
- "lol", "haha nice", "cool"
- "Got it, thanks!", "Appreciate it"

Output format:
[{{"query": "casual message", "response": "friendly response, optionally offering help", "tool": null}}]

Generate {count} casual conversation examples (be natural and warm, not robotic):"""
        
        else:  # missing_params
            prompt = f"""{base_context}

Generate {count} examples where user wants to use a tool but is missing required info.
Example tool: {sample_name} - {sample_desc}

Output format:
[{{"query": "incomplete request for {sample_name}", "response": "question asking for missing details", "tool": null}}]

Generate {count} missing-info examples:"""
        
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> list:
        """Make LLM call with retry logic. Supports OpenAI and Anthropic."""
        self._init_client()
        
        for attempt in range(self.config.max_retries):
            try:
                self.stats["api_calls"] += 1
                
                if self.api_provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.9,
                        max_tokens=3000
                    )
                    content = response.choices[0].message.content.strip()
                else:  # anthropic
                    response = self.client.messages.create(
                        model=self.config.anthropic_model,
                        max_tokens=3000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text.strip()
                
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
        """Split examples into train/val/test (simple, non-stratified)."""
        n = len(examples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        return {
            "train": examples[:train_end],
            "validation": examples[train_end:val_end],
            "test": examples[val_end:]
        }
    
    def _stratified_split(self, single: list, multi: list, edge: list) -> dict:
        """
        Stratified split: ensures each split has proportional examples from ALL categories.
        
        This prevents scenarios where all edge cases end up in test set,
        leaving the model untrained on refusals/casual chat.
        
        Args:
            single: Single-tool examples (already in chat format)
            multi: Multi-tool examples (already in chat format)
            edge: Edge case examples (already in chat format)
        
        Returns:
            {"train": [...], "validation": [...], "test": [...]}
        """
        def split_category(examples: list) -> tuple:
            """Split a single category by train/val/test ratios."""
            random.shuffle(examples)
            n = len(examples)
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)
            return (
                examples[:train_end],
                examples[train_end:val_end],
                examples[val_end:]
            )
        
        # Split each category proportionally
        single_train, single_val, single_test = split_category(single)
        multi_train, multi_val, multi_test = split_category(multi)
        edge_train, edge_val, edge_test = split_category(edge)
        
        # Combine and shuffle within each split
        train = single_train + multi_train + edge_train
        val = single_val + multi_val + edge_val
        test = single_test + multi_test + edge_test
        
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return {
            "train": train,
            "validation": val,
            "test": test
        }
    
    def _simulate_tool_result(self, tool_name: str, parameters: dict) -> str:
        """
        Generate a realistic simulated tool result for training.
        
        This teaches the model what tool results look like so it can
        continue appropriately after receiving them at runtime.
        """
        # Common patterns based on tool name keywords
        tool_lower = tool_name.lower()
        
        if "list_issues" in tool_lower or "get_issue" in tool_lower:
            repo = parameters.get("repo", "my-repo")
            return json.dumps({
                "issues": [
                    {"number": 42, "title": "Bug in authentication flow", "state": "open"},
                    {"number": 38, "title": "Performance issue on dashboard", "state": "open"},
                    {"number": 35, "title": "Update dependencies", "state": "closed"}
                ],
                "total_count": 3
            })
        
        elif "create_issue" in tool_lower:
            return json.dumps({
                "number": random.randint(100, 999),
                "title": parameters.get("title", "New issue"),
                "url": f"https://github.com/{parameters.get('owner', 'org')}/{parameters.get('repo', 'repo')}/issues/{random.randint(100, 999)}",
                "state": "open"
            })
        
        elif "search_repo" in tool_lower:
            query = parameters.get("query", "search")
            return json.dumps({
                "items": [
                    {"full_name": "facebook/react", "description": "A JavaScript library for building user interfaces", "stars": 220000},
                    {"full_name": "vuejs/vue", "description": "Progressive JavaScript Framework", "stars": 205000}
                ],
                "total_count": 2
            })
        
        elif "send_message" in tool_lower or "post_message" in tool_lower:
            return json.dumps({
                "ok": True,
                "message_id": f"msg_{random.randint(1000, 9999)}",
                "channel": parameters.get("channel", "#general"),
                "timestamp": "2026-01-23T10:30:00Z"
            })
        
        elif "list_channel" in tool_lower:
            return json.dumps({
                "channels": [
                    {"id": "C001", "name": "general"},
                    {"id": "C002", "name": "dev-team"},
                    {"id": "C003", "name": "alerts"}
                ]
            })
        
        elif "get_file" in tool_lower or "read_file" in tool_lower:
            return json.dumps({
                "content": "# README\n\nThis is the project documentation...",
                "path": parameters.get("path", "README.md"),
                "encoding": "utf-8"
            })
        
        elif "search_user" in tool_lower:
            return json.dumps({
                "users": [
                    {"login": "octocat", "name": "The Octocat", "type": "User"},
                    {"login": "hubot", "name": "Hubot", "type": "Bot"}
                ]
            })
        
        elif "get_pull" in tool_lower or "list_pull" in tool_lower:
            return json.dumps({
                "pull_requests": [
                    {"number": 101, "title": "Add new feature", "state": "open", "user": "developer1"},
                    {"number": 99, "title": "Fix critical bug", "state": "merged", "user": "developer2"}
                ]
            })
        
        else:
            # Generic success response
            return json.dumps({
                "success": True,
                "message": f"Operation completed successfully",
                "data": parameters
            })
    
    def _generate_final_summary(self, query: str, tool_calls: list, results: list) -> str:
        """
        Generate a natural final response summarizing what was done.
        """
        # Build a simple summary based on tools used
        tool_names = [tc.get("tool", "") for tc in tool_calls]
        
        summaries = []
        for i, (tool, result) in enumerate(zip(tool_names, results)):
            tool_lower = tool.lower()
            try:
                result_data = json.loads(result)
            except:
                result_data = {}
            
            if "list_issues" in tool_lower:
                count = result_data.get("total_count", len(result_data.get("issues", [])))
                summaries.append(f"found {count} issues")
            elif "create_issue" in tool_lower:
                num = result_data.get("number", "")
                summaries.append(f"created issue #{num}")
            elif "search_repo" in tool_lower:
                count = result_data.get("total_count", len(result_data.get("items", [])))
                summaries.append(f"found {count} repositories")
            elif "send_message" in tool_lower:
                channel = result_data.get("channel", "the channel")
                summaries.append(f"posted the update to {channel}")
            elif "list_channel" in tool_lower:
                count = len(result_data.get("channels", []))
                summaries.append(f"found {count} channels")
            else:
                summaries.append(f"completed {tool}")
        
        if len(summaries) == 1:
            return f"Done! I {summaries[0]}."
        elif len(summaries) == 2:
            return f"Done! I {summaries[0]} and {summaries[1]}."
        else:
            return f"Done! I {', '.join(summaries[:-1])}, and {summaries[-1]}."
    
    def _to_chat_format(self, example: dict) -> dict:
        """
        Convert simple format to chat format for training.
        
        Input format (simple):
            {"query": "...", "tool": "...", "parameters": {...}}
            or
            {"query": "...", "tool_calls": [...]}  # Multi-tool with full chain!
            or
            {"query": "...", "response": "...", "tool": null}  # edge cases
        
        Output format (chat) - SINGLE TOOL:
            {"messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "<tool_call>...</tool_call>"}
            ]}
        
        Output format (chat) - MULTI-TOOL (NEW!):
            {"messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "<tool_call>tool1</tool_call>"},
                {"role": "tool", "content": "{result1}"},
                {"role": "assistant", "content": "<tool_call>tool2</tool_call>"},
                {"role": "tool", "content": "{result2}"},
                {"role": "assistant", "content": "Done! I completed both tasks."}
            ]}
        """
        messages = []
        
        # System prompt (CRITICAL for training!)
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        else:
            messages.append({"role": "system", "content": self._generate_minimal_system_prompt()})
        
        # User message
        query = example.get("query", "")
        messages.append({"role": "user", "content": query})
        
        # Assistant response
        if example.get("tool") is None and example.get("response"):
            # Edge case: no tool call, just a response
            messages.append({"role": "assistant", "content": example["response"]})
        
        elif "tool_calls" in example:
            # MULTI-TOOL: Full conversation chain with simulated results!
            tool_calls = example["tool_calls"]
            results = []
            
            for i, call in enumerate(tool_calls):
                tool_name = call.get("tool", "")
                params = call.get("parameters", {})
                
                # Assistant makes tool call
                tool_json = json.dumps({"tool": tool_name, "parameters": params})
                messages.append({
                    "role": "assistant",
                    "content": f"<tool_call>\n{tool_json}\n</tool_call>"
                })
                
                # Simulate tool result
                result = self._simulate_tool_result(tool_name, params)
                results.append(result)
                messages.append({
                    "role": "tool",
                    "content": result
                })
            
            # Final summary from assistant
            final_summary = self._generate_final_summary(query, tool_calls, results)
            messages.append({
                "role": "assistant",
                "content": final_summary
            })
        
        elif "tool" in example:
            # Single tool call (unchanged)
            tool_json = json.dumps({
                "tool": example.get("tool"),
                "parameters": example.get("parameters", {})
            })
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{tool_json}\n</tool_call>"
            })
        
        else:
            # Fallback
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
    
    def get_quality_metrics(self, datasets: dict) -> dict:
        """
        Analyze generated data quality.
        
        Returns metrics useful for understanding the data quality:
        - Tool distribution (is it balanced?)
        - Edge case coverage
        - Example diversity
        
        Args:
            datasets: Output from generate()
        
        Returns:
            dict with quality metrics
        """
        all_examples = datasets['train'] + datasets['validation'] + datasets['test']
        
        # Count tools used
        tool_counts = {}
        edge_case_count = 0
        
        for ex in all_examples:
            messages = ex.get('messages', [])
            assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
            
            if assistant_msg:
                content = assistant_msg['content']
                if '<tool_call>' in content:
                    # Extract tool name
                    match = re.search(r'"tool":\s*"([^"]+)"', content)
                    if match:
                        tool_name = match.group(1)
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                else:
                    edge_case_count += 1
        
        # Calculate balance score (0-1, 1 = perfectly balanced)
        if tool_counts:
            counts = list(tool_counts.values())
            avg = sum(counts) / len(counts)
            variance = sum((c - avg) ** 2 for c in counts) / len(counts)
            std_dev = variance ** 0.5
            # Balance score: lower std_dev relative to avg = better balance
            balance_score = max(0, 1 - (std_dev / avg)) if avg > 0 else 0
        else:
            balance_score = 0
        
        return {
            "total_examples": len(all_examples),
            "tool_distribution": tool_counts,
            "edge_cases": edge_case_count,
            "balance_score": round(balance_score, 2),  # 0-1, higher is better
            "api_calls": self.stats.get("api_calls", 0),
            "failed_batches": self.stats.get("failed_batches", 0),
            "fixed_tool_names": self.stats.get("fixed_tool_names", 0),
            "skipped_invalid": self.stats.get("skipped_invalid", 0),
        }


# =============================================================================
# Convenience functions
# =============================================================================

def generate_training_data_batched(
    tools: list,
    problem_statement: str,
    api_key: str,
    total_examples: int = None,  # Auto-calculated if None
    output_dir: str = None,
    system_prompt: str = None,
) -> dict:
    """
    Generate training data with efficient batching.
    
    Args:
        tools: List of ToolSchema objects (flat list from all servers)
        problem_statement: What the agent should do
        api_key: OpenAI or Anthropic API key (auto-detected)
        total_examples: Target number of examples (auto-calculated from tool count if None)
        output_dir: Where to save (optional)
        system_prompt: System prompt to include in training examples
    
    Returns:
        dict with 'train', 'validation', 'test' lists
    """
    # Auto-calculate optimal examples based on tool count
    if total_examples is None:
        total_examples = BatchGenConfig.calculate_optimal_examples(len(tools))
        print(f"📊 Auto-calculated {total_examples} examples for {len(tools)} tools")
    
    config = BatchGenConfig(total_examples=total_examples)
    generator = BatchedDataGenerator(
        tools=tools,
        problem_statement=problem_statement,
        api_key=api_key,
        config=config,
        system_prompt=system_prompt
    )
    
    if output_dir:
        return generator.save(output_dir)
    else:
        return generator.generate()


def recommend_examples(num_tools: int) -> dict:
    """
    Get recommended dataset configuration for a given number of tools.
    
    Args:
        num_tools: Number of tools the agent will use
    
    Returns:
        dict with recommendations
    """
    total = BatchGenConfig.calculate_optimal_examples(num_tools)
    config = BatchGenConfig(total_examples=total)
    targets = config.get_targets()
    
    return {
        "num_tools": num_tools,
        "recommended_total": total,
        "breakdown": {
            "single_tool": targets["single_tool"],
            "multi_tool": targets["multi_tool"],
            "edge_cases": targets["edge_cases"],
        },
        "splits": {
            "train": int(total * config.train_ratio),
            "validation": int(total * config.val_ratio),
            "test": int(total * config.test_ratio),
        },
        "estimated_api_calls": total // config.batch_size,
    }
