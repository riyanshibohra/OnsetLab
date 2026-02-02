"""
Single-Tool Data Generator
===========================
Training data generation for single-tool-per-turn architecture.
Orchestrates: prompt building (generators), API calls, validation (validators), format conversion.

Key features:
- Single tool calls only (no multi-tool chains)
- Strong clarification handling (users often give incomplete info)
- Batched generation (10 examples per LLM call)
- Tool-aware sample sizing
- Format + semantic validation (validators.py)
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from . import generators as gen
from . import validators as val


@dataclass
class DataGenConfig:
    """
    Configuration for data generation.
    
    SAMPLE SIZE RULES (per tool):
    - MINIMUM: 10 examples/tool → Risk of underfitting
    - SUGGESTED: 25 examples/tool → Sweet spot for 3B model
    - MAXIMUM: 40 examples/tool → Cap to avoid overfitting
    
    EXAMPLES BY TOOL COUNT (at 50/tool suggested):
    | Tools | Minimum | Suggested | Maximum |
    |-------|---------|-----------|---------|
    | 10    | 200     | 667       | 1000    |
    | 11    | 220     | 733       | 1100    |
    | 15    | 300     | 1000      | 1500    |
    | 20    | 400     | 1333      | 2000    |
    
    Note: These are TOTAL examples (75% single-tool + 25% edge cases)
    """
    
    # Per-tool limits (ENFORCED)
    min_per_tool: int = 15            # Minimum - below this = underfitting
    examples_per_tool: int = 50       # Suggested - 50/tool for robust learning
    max_per_tool: int = 75            # Maximum - 3B can handle more with LoRA
    
    # Distribution (must sum to 1.0)
    single_tool_ratio: float = 0.65   # 65% successful tool calls
    edge_case_ratio: float = 0.35     # 35% edge cases (increased for better "no tool" learning)
    
    # Edge case breakdown (within the 35%)
    # CRITICAL: 3B models default to calling tools - need lots of "no tool" examples
    clarification_ratio: float = 0.15  # 15% of edge cases = 5.25% total
    follow_up_ratio: float = 0.10      # 10% of edge cases = 3.5% total
    refusal_ratio: float = 0.25        # 25% of edge cases = 8.75% total (can't do this)
    casual_ratio: float = 0.50         # 50% of edge cases = 17.5% total (greetings, thanks)
    
    # Split ratios
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    
    # Batch settings
    batch_size: int = 20              # Examples per LLM call (20 = more per call, fewer total calls)
    max_retries: int = 3
    
    # API settings (model used to GENERATE training data; provider = OpenAI if key not sk-ant-*, else Anthropic)
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    
    # Tool call format (matches model type)
    # - "qwen": <tool_call>{"tool": "...", "parameters": {...}}</tool_call>
    # - "toolllama": Action: tool_name\nAction Input: {...}
    # - "nexusraven": Call: function_name(param="value")
    tool_format: str = "toolllama"    # Default to ToolLLaMA format
    
    def calculate_total(self, num_tools: int, level: str = "suggested") -> int:
        """
        Calculate total examples based on number of tools.
        
        Args:
            num_tools: Number of tools
            level: "min", "suggested", or "max"
        
        Returns:
            Total examples (includes 75% single-tool + 25% edge cases)
        """
        if level == "min":
            per_tool = self.min_per_tool
        elif level == "max":
            per_tool = self.max_per_tool
        else:
            per_tool = self.examples_per_tool
        
        # Single-tool examples = num_tools * per_tool
        single_tool_count = num_tools * per_tool
        
        # Total = single_tool / 0.75 (because single-tool is 75% of total)
        total = int(single_tool_count / self.single_tool_ratio)
        
        return total
    
    def get_limits(self, num_tools: int) -> dict:
        """
        Get min/suggested/max limits for a given number of tools.
        
        Args:
            num_tools: Number of tools
        
        Returns:
            dict with min, suggested, max totals
        """
        return {
            "num_tools": num_tools,
            "minimum": {
                "total": self.calculate_total(num_tools, "min"),
                "per_tool": self.min_per_tool,
                "risk": "Underfitting - model may not learn well"
            },
            "suggested": {
                "total": self.calculate_total(num_tools, "suggested"),
                "per_tool": self.examples_per_tool,
                "risk": "Sweet spot for 3B + LoRA"
            },
            "maximum": {
                "total": self.calculate_total(num_tools, "max"),
                "per_tool": self.max_per_tool,
                "risk": "Overfitting risk - model memorizes instead of learns"
            }
        }
    
    def get_targets(self, num_tools: int) -> dict:
        """Get target counts for each category (uses suggested level)."""
        total = self.calculate_total(num_tools, "suggested")
        edge_total = int(total * self.edge_case_ratio)
        
        return {
            "single_tool": int(total * self.single_tool_ratio),
            "clarification": int(edge_total * self.clarification_ratio),
            "follow_up": int(edge_total * self.follow_up_ratio),
            "refusal": int(edge_total * self.refusal_ratio),
            "casual": int(edge_total * self.casual_ratio),
            "total": total,
            "per_tool": self.examples_per_tool
        }


class DataGenerator:
    """
    Generates training data for single-tool architecture.
    
    Focus: One tool call per turn, strong clarification handling.
    """
    
    def __init__(
        self,
        tools: list,
        problem_statement: str,
        api_key: str,
        config: DataGenConfig = None,
        system_prompt: str = None,
        skill: str = None,
    ):
        self.tools = tools
        self.problem_statement = problem_statement
        self.api_key = api_key
        self.config = config or DataGenConfig()
        self.system_prompt = system_prompt
        self.skill = skill  # Skill document for guided generation
        
        # Auto-detect API provider
        self.api_provider = "anthropic" if api_key.startswith("sk-ant-") else "openai"
        
        # Build tool lookup
        self.tool_names = set()
        self.tool_map = {}  # normalized -> actual name
        for t in tools:
            name = t.get('name') if isinstance(t, dict) else getattr(t, 'name', '')
            self.tool_names.add(name)
            self.tool_map[name.lower().replace("-", "_")] = name
        
        # Stats
        self.stats = {"api_calls": 0, "examples_generated": 0, "failed": 0}
        self.client = None
    
    def generate(self) -> dict:
        """Generate training data."""
        targets = self.config.get_targets(len(self.tools))
        
        print(f"\n{'='*50}")
        print(f"🚀 Single-Tool Data Generator")
        print(f"{'='*50}")
        print(f"Tools: {len(self.tools)}")
        print(f"Total examples: {targets['total']}")
        print(f"  - Single-tool: {targets['single_tool']} (65%)")
        print(f"  - Clarification: {targets['clarification']} (5%)")
        print(f"  - Follow-up: {targets['follow_up']} (3.5%)")
        print(f"  - Refusal: {targets['refusal']} (9%)")
        print(f"  - Casual: {targets['casual']} (17.5%)")
        print(f"{'='*50}\n")
        
        # Generate each category
        print("📝 Generating single-tool examples...")
        single_examples = self._generate_single_tool(targets['single_tool'])
        print(f"   ✅ {len(single_examples)} examples\n")
        
        print("🤔 Generating clarification examples...")
        clarification_examples = self._generate_clarification(targets['clarification'])
        print(f"   ✅ {len(clarification_examples)} examples\n")
        
        print("🔄 Generating follow-up examples...")
        follow_up_examples = self._generate_follow_up(targets['follow_up'])
        print(f"   ✅ {len(follow_up_examples)} examples\n")
        
        print("🚫 Generating refusal examples...")
        refusal_examples = self._generate_refusal(targets['refusal'])
        print(f"   ✅ {len(refusal_examples)} examples\n")
        
        print("💬 Generating casual examples...")
        casual_examples = self._generate_casual(targets['casual'])
        print(f"   ✅ {len(casual_examples)} examples\n")
        
        # Convert to chat format
        all_examples = []
        for ex in single_examples:
            all_examples.append(self._to_chat_format(ex, "tool_call"))
        for ex in clarification_examples:
            all_examples.append(self._to_chat_format(ex, "clarification"))
        for ex in follow_up_examples:
            all_examples.append(self._to_chat_format(ex, "follow_up"))
        for ex in refusal_examples:
            all_examples.append(self._to_chat_format(ex, "refusal"))
        for ex in casual_examples:
            all_examples.append(self._to_chat_format(ex, "casual"))
        
        # Shuffle and split
        random.shuffle(all_examples)
        datasets = self._split(all_examples)
        
        # Update stats with total generated
        self.stats["examples_generated"] = len(all_examples)
        
        print(f"{'='*50}")
        print(f"📊 Generation Complete!")
        print(f"{'='*50}")
        print(f"Total: {len(all_examples)}")
        print(f"  Train: {len(datasets['train'])}")
        print(f"  Validation: {len(datasets['validation'])}")
        print(f"  Test: {len(datasets['test'])}")
        print(f"API calls: {self.stats['api_calls']}")
        print(f"{'='*50}\n")
        
        return datasets
    
    # =========================================================================
    # Single-Tool Generation
    # =========================================================================
    
    def _generate_single_tool(self, target: int) -> list:
        """Generate successful single-tool examples with balanced coverage using parallel API calls."""
        examples = []
        tool_counts = {self._get_name(t): 0 for t in self.tools}
        parallel_workers = 4  # Make 4 API calls at once
        
        while len(examples) < target:
            # Prepare multiple batches of tools (for parallel calls)
            batch_list = []
            for _ in range(parallel_workers):
                batch_tools = sorted(
                    self.tools, 
                    key=lambda t: tool_counts[self._get_name(t)]
                )[:5]
                batch_list.append(batch_tools)
            
            # Make parallel API calls
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(self._call_llm_single_tool, bt) for bt in batch_list]
                for future in as_completed(futures):
                    try:
                        batch = future.result()
                        for ex in batch:
                            tool = ex.get('tool')
                            if tool in tool_counts:
                                tool_counts[tool] += 1
                                examples.append(ex)
                    except Exception as e:
                        print(f"   ⚠ Batch failed: {e}")
            
            # Progress after parallel batch
            print(f"   ✓ {len(examples)}/{target} examples generated")
        
        return examples[:target]
    
    def _call_llm_single_tool(self, tools: list) -> list:
        """Generate batch of single-tool examples."""
        tools_desc = self._format_tools(tools)
        prompt = gen.build_single_tool_prompt(
            self.problem_statement, tools_desc, self.config.batch_size,
            skill=self.skill  # Pass skill for guided generation
        )
        return self._call_llm(prompt, example_type="tool_call")
    
    # =========================================================================
    # Clarification Generation (most important edge case!)
    # =========================================================================
    
    def _generate_clarification(self, target: int) -> list:
        """Generate examples where user gives incomplete info using parallel API calls."""
        examples = []
        parallel_workers = 3
        
        while len(examples) < target:
            batch_list = [random.sample(self.tools, min(5, len(self.tools))) for _ in range(parallel_workers)]
            
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(self._call_llm_clarification, bt) for bt in batch_list]
                for future in as_completed(futures):
                    try:
                        examples.extend(future.result())
                    except Exception as e:
                        print(f"   ⚠ Batch failed: {e}")
            
            print(f"   ✓ {len(examples)}/{target} clarification examples")
        
        return examples[:target]
    
    def _call_llm_clarification(self, tools: list) -> list:
        """Generate clarification examples - user missing required params."""
        tools_desc = self._format_tools_with_required(tools)
        prompt = gen.build_clarification_prompt(
            self.problem_statement, tools_desc, self.config.batch_size,
            skill=self.skill  # Pass skill for clarification patterns
        )
        return self._call_llm(prompt, example_type="clarification")
    
    # =========================================================================
    # Follow-up Generation (context-aware queries)
    # =========================================================================
    
    def _generate_follow_up(self, target: int) -> list:
        """
        Generate examples where user asks a follow-up question with conversation context.
        Uses parallel API calls for speed.
        """
        examples = []
        parallel_workers = 3
        
        while len(examples) < target:
            batch_list = [random.sample(self.tools, min(5, len(self.tools))) for _ in range(parallel_workers)]
            
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(self._call_llm_follow_up, bt) for bt in batch_list]
                for future in as_completed(futures):
                    try:
                        examples.extend(future.result())
                    except Exception as e:
                        print(f"   ⚠ Batch failed: {e}")
            
            print(f"   ✓ {len(examples)}/{target} follow-up examples")
        
        return examples[:target]
    
    def _call_llm_follow_up(self, tools: list) -> list:
        """Generate follow-up examples with conversation context."""
        tools_desc = self._format_tools_with_required(tools)
        prompt = gen.build_follow_up_prompt(tools_desc, self.config.batch_size)
        return self._call_llm(prompt, example_type="follow_up")
    
    # =========================================================================
    # Refusal Generation
    # =========================================================================
    
    def _generate_refusal(self, target: int) -> list:
        """Generate examples where request is out of scope using parallel API calls."""
        examples = []
        parallel_workers = 3
        
        while len(examples) < target:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(self._call_llm_refusal) for _ in range(parallel_workers)]
                for future in as_completed(futures):
                    try:
                        examples.extend(future.result())
                    except Exception as e:
                        print(f"   ⚠ Batch failed: {e}")
            
            print(f"   ✓ {len(examples)}/{target} refusal examples")
        
        return examples[:target]
    
    def _call_llm_refusal(self) -> list:
        """Generate refusal examples - can't do what user asks."""
        tool_names = [self._get_name(t) for t in self.tools[:10]]
        prompt = gen.build_refusal_prompt(
            self.problem_statement, tool_names, self.config.batch_size
        )
        return self._call_llm(prompt, example_type="refusal")
    
    # =========================================================================
    # Casual Conversation Generation
    # =========================================================================
    
    def _generate_casual(self, target: int) -> list:
        """Generate casual conversation examples using parallel API calls."""
        examples = []
        parallel_workers = 3
        
        while len(examples) < target:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(self._call_llm_casual) for _ in range(parallel_workers)]
                for future in as_completed(futures):
                    try:
                        examples.extend(future.result())
                    except Exception as e:
                        print(f"   ⚠ Batch failed: {e}")
            
            print(f"   ✓ {len(examples)}/{target} casual examples")
        
        return examples[:target]
    
    def _call_llm_casual(self) -> list:
        """Generate casual conversation - greetings, capability questions, thanks, etc."""
        tool_names = [self._get_name(t) for t in self.tools]
        tool_list = ", ".join(tool_names[:5]) + ("..." if len(tool_names) > 5 else "")
        prompt = gen.build_casual_prompt(tool_list, self.config.batch_size)
        return self._call_llm(prompt, example_type="casual")
    
    # =========================================================================
    # LLM Calling
    # =========================================================================
    
    def _call_llm(self, prompt: str, example_type: str = "tool_call") -> list:
        """Make LLM API call; validate responses with validators (format + semantic)."""
        if self.client is None:
            self._init_client()
        # Semantic validation only for tool-call examples (Shaw / Microsoft: filter wrong-tool-for-query)
        do_semantic = example_type in ("tool_call", "follow_up")

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
                else:
                    response = self.client.messages.create(
                        model=self.config.anthropic_model,
                        max_tokens=3000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text.strip()

                examples = self._parse_json(content)
                valid = [
                    ex for ex in examples
                    if val.validate_example(
                        ex, self.tool_names, self.tool_map, self.tools, semantic=do_semantic
                    )
                ]
                return valid
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    self.stats["failed"] += 1
                    print(f"   ⚠️ Batch failed: {e}")
                    return []
        return []
    
    def _init_client(self):
        """Initialize API client."""
        if self.api_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        else:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _parse_json(self, content: str) -> list:
        """Extract JSON array from response."""
        try:
            return json.loads(content)
        except:
            pass
        
        # Find JSON array
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        # Remove markdown
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        try:
            return json.loads(content)
        except:
            return []
    
    # =========================================================================
    # Format Conversion
    # =========================================================================
    
    def _to_chat_format(self, example: dict, example_type: str) -> dict:
        """Convert to chat format for training."""
        messages = []
        
        # System prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        else:
            messages.append({"role": "system", "content": self._default_system_prompt()})
        
        # User message
        messages.append({"role": "user", "content": example.get("query", "")})
        
        # Get tool format from config
        tool_format = self.config.tool_format if self.config else "toolllama"
        
        # Assistant response
        if example_type in ("tool_call", "follow_up"):
            # Tool call + result + final response
            # follow_up is the same as tool_call but with context in the query
            tool = example.get("tool")
            params = example.get("parameters", example.get("params", {}))
            
            # Format tool call based on model type
            if tool_format == "toolllama":
                # ToolLLaMA format: Action: tool_name\nAction Input: {...}
                params_json = json.dumps(params)
                tool_call_str = f"Action: {tool}\nAction Input: {params_json}"
            elif tool_format == "nexusraven":
                # NexusRaven format: Call: func(param="value")
                param_strs = [f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in params.items()]
                tool_call_str = f"Call: {tool}({', '.join(param_strs)})"
            else:
                # Qwen/default format: <tool_call>{"tool": ..., "parameters": ...}</tool_call>
                tool_json = json.dumps({"tool": tool, "parameters": params})
                tool_call_str = f"<tool_call>\n{tool_json}\n</tool_call>"
            
            messages.append({
                "role": "assistant",
                "content": tool_call_str
            })
            
            # Simulated result
            result = self._simulate_result(tool, params)
            
            # Format result based on model type
            if tool_format == "toolllama":
                messages.append({"role": "user", "content": f"Observation: {result}"})
            else:
                messages.append({"role": "tool", "content": result})
            
            # Final response
            response = self._generate_response(tool, params, result)
            messages.append({"role": "assistant", "content": response})
        
        else:
            # Clarification, refusal, or casual - just text response
            messages.append({
                "role": "assistant",
                "content": example.get("response", "I can help with that.")
            })
        
        return {"messages": messages}
    
    def _simulate_result(self, tool_name: str, params: dict) -> str:
        """Generate realistic tool result (no LLM call)."""
        name = tool_name.lower()
        item_id = random.randint(100, 9999)
        
        # List/search operations
        if any(x in name for x in ['list', 'search', 'find', 'query']):
            items = [
                {"id": f"item_{item_id + i}", "name": f"Result {i+1}", "created_at": "2026-01-25T10:00:00Z"}
                for i in range(random.randint(2, 5))
            ]
            return json.dumps({"items": items, "total": len(items)})
        
        # Create operations
        if any(x in name for x in ['create', 'add', 'new', 'write']):
            return json.dumps({
                "id": item_id,
                "created": True,
                "url": f"https://example.com/item/{item_id}"
            })
        
        # Get/read operations
        if any(x in name for x in ['get', 'read', 'fetch', 'retrieve']):
            return json.dumps({
                "id": item_id,
                "name": params.get("name", "Item"),
                "content": "Sample content here"
            })
        
        # Update operations
        if any(x in name for x in ['update', 'edit', 'modify']):
            return json.dumps({"id": item_id, "updated": True})
        
        # Delete operations
        if any(x in name for x in ['delete', 'remove']):
            return json.dumps({"deleted": True})
        
        # Send/post operations
        if any(x in name for x in ['send', 'post', 'message']):
            return json.dumps({"sent": True, "message_id": f"msg_{item_id}"})
        
        # Default
        return json.dumps({"success": True})
    
    def _generate_response(self, tool_name: str, params: dict, result: str) -> str:
        """Generate human-friendly response from tool result."""
        try:
            data = json.loads(result)
        except:
            return "Done!"
        
        name = tool_name.lower()
        
        # List results
        if 'items' in data:
            items = data['items']
            if not items:
                return "No results found."
            lines = [f"Found {len(items)} results:"]
            for i, item in enumerate(items[:5], 1):
                n = item.get('name', item.get('title', item.get('id', f'Item {i}')))
                lines.append(f"{i}. {n}")
            return "\n".join(lines)
        
        # Created
        if data.get('created'):
            url = data.get('url', '')
            if url:
                return f"✓ Created successfully!\nURL: {url}"
            return "✓ Created successfully!"
        
        # Updated
        if data.get('updated'):
            return "✓ Updated successfully!"
        
        # Deleted
        if data.get('deleted'):
            return "✓ Deleted successfully!"
        
        # Sent
        if data.get('sent'):
            return "✓ Sent successfully!"
        
        # Default
        return "Done!"
    
    def _default_system_prompt(self) -> str:
        """Generate concise system prompt optimized for 3B models."""
        from .prompts import generate_prompt_for_3b
        return generate_prompt_for_3b(self.problem_statement, self.tools)
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _get_name(self, tool) -> str:
        """Get tool name from dict or object."""
        if isinstance(tool, dict):
            return tool.get('name', '')
        return getattr(tool, 'name', '')
    
    def _get_params(self, tool) -> dict:
        """Get tool parameters."""
        if isinstance(tool, dict):
            return tool.get('parameters', {})
        return getattr(tool, 'parameters', {})
    
    def _get_required(self, tool) -> list:
        """Get required parameters."""
        if isinstance(tool, dict):
            return tool.get('required_params', [])
        return getattr(tool, 'required_params', [])
    
    def _format_tools(self, tools: list) -> str:
        """Format tools for prompt with parameter types."""
        lines = []
        for t in tools:
            name = self._get_name(t)
            desc = t.get('description', '') if isinstance(t, dict) else getattr(t, 'description', '')
            params = self._get_params(t)
            
            # Include type info so LLM generates correct types
            if params:
                param_parts = []
                for pname, pinfo in params.items():
                    if isinstance(pinfo, dict):
                        ptype = pinfo.get('type', 'string')
                        enum_values = pinfo.get('enum')
                        
                        # If enum is defined, show allowed values (CRITICAL for APIs)
                        if enum_values:
                            enum_str = "|".join(str(v) for v in enum_values[:4])  # Show up to 4 values
                            param_parts.append(f"{pname}:enum({enum_str})")
                        # Make types explicit for common mistakes
                        elif ptype == 'array':
                            param_parts.append(f"{pname}:array(use [...])")
                        elif ptype in ('number', 'integer'):
                            param_parts.append(f"{pname}:number(no quotes)")
                        elif ptype == 'boolean':
                            param_parts.append(f"{pname}:boolean(true/false)")
                        elif ptype == 'object':
                            # Handle nested object types - show properties if available
                            props = pinfo.get('properties', {})
                            if props:
                                # Show nested property names with types
                                nested_parts = []
                                for prop_name, prop_info in props.items():
                                    prop_type = prop_info.get('type', 'string') if isinstance(prop_info, dict) else 'string'
                                    nested_parts.append(f"{prop_name}:{prop_type}")
                                nested_str = ", ".join(nested_parts[:5])  # Show up to 5 nested props
                                param_parts.append(f"{pname}:object({{{nested_str}}})")
                            else:
                                param_parts.append(f"{pname}:object({{...}})")
                        else:
                            param_parts.append(f"{pname}:string")
                    else:
                        param_parts.append(pname)
                param_str = ", ".join(param_parts)
            else:
                param_str = "none"
            
            lines.append(f"• {name}: {desc}\n  Params: {param_str}")
        return "\n".join(lines)
    
    def _format_tools_with_required(self, tools: list) -> str:
        """Format tools with required params highlighted."""
        lines = []
        for t in tools:
            name = self._get_name(t)
            desc = t.get('description', '') if isinstance(t, dict) else getattr(t, 'description', '')
            required = self._get_required(t)
            if required:
                lines.append(f"• {name}: {desc}\n  REQUIRED: {', '.join(required)}")
            else:
                params = self._get_params(t)
                lines.append(f"• {name}: {desc}\n  Params: {', '.join(params.keys()) if params else 'none'}")
        return "\n".join(lines)
    
    def _split(self, examples: list) -> dict:
        """Split into train/val/test."""
        n = len(examples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        return {
            "train": examples[:train_end],
            "validation": examples[train_end:val_end],
            "test": examples[val_end:]
        }
    
    def save(self, output_dir: str) -> dict:
        """Generate and save to files."""
        datasets = self.generate()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for name, examples in datasets.items():
            file_path = output_path / f"{name}.jsonl"
            with open(file_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            paths[name] = str(file_path)
            print(f"💾 Saved {len(examples)} to {file_path}")
        
        return paths


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_training_data(
    tools: list,
    problem_statement: str,
    api_key: str,
    output_dir: str = None,
    system_prompt: str = None,
    examples_per_tool: int = 25,
    skill: str = None,
) -> dict:
    """
    Generate training data for single-tool architecture.
    
    Args:
        tools: List of tool dicts with 'name', 'description', 'parameters', 'required_params'
        problem_statement: What the agent should do
        api_key: OpenAI or Anthropic API key
        output_dir: Where to save (optional)
        system_prompt: System prompt for training examples
        examples_per_tool: Target examples per tool (default 25)
        skill: Skill document for guided generation (optional but recommended)
    
    Returns:
        dict with 'train', 'validation', 'test' lists
    """
    config = DataGenConfig(examples_per_tool=examples_per_tool)
    
    generator = DataGenerator(
        tools=tools,
        problem_statement=problem_statement,
        api_key=api_key,
        config=config,
        system_prompt=system_prompt,
        skill=skill
    )
    
    if output_dir:
        return generator.save(output_dir)
    return generator.generate()


def recommend_dataset_size(num_tools: int) -> dict:
    """
    Get recommended dataset configuration with min/suggested/max limits.
    
    SAMPLE SIZE RULES:
    - MINIMUM: 10 examples per tool → Risk of underfitting
    - SUGGESTED: 25 examples per tool → Sweet spot for 3B model
    - MAXIMUM: 40 examples per tool → Cap to avoid overfitting
    
    Args:
        num_tools: Number of tools the agent will use
    
    Returns:
        dict with limits and breakdown
    
    Example:
        >>> recommend_dataset_size(15)
        {
            'num_tools': 15,
            'limits': {
                'minimum': {'total': 200, 'per_tool': 10},
                'suggested': {'total': 500, 'per_tool': 25},
                'maximum': {'total': 800, 'per_tool': 40}
            },
            ...
        }
    """
    config = DataGenConfig()
    limits = config.get_limits(num_tools)
    targets = config.get_targets(num_tools)
    
    return {
        "num_tools": num_tools,
        "limits": {
            "minimum": {
                "total": limits['minimum']['total'],
                "per_tool": limits['minimum']['per_tool'],
                "note": "Below this = underfitting"
            },
            "suggested": {
                "total": limits['suggested']['total'],
                "per_tool": limits['suggested']['per_tool'],
                "note": "Sweet spot for 3B + LoRA"
            },
            "maximum": {
                "total": limits['maximum']['total'],
                "per_tool": limits['maximum']['per_tool'],
                "note": "Above this = overfitting risk"
            }
        },
        "using_suggested": {
            "total": targets['total'],
            "breakdown": {
                "single_tool (65%)": targets['single_tool'],
                "clarification (5%)": targets['clarification'],
                "follow_up (3.5%)": targets['follow_up'],
                "refusal (9%)": targets['refusal'],
                "casual (17.5%)": targets['casual'],
            },
            "splits": {
                "train (85%)": int(targets['total'] * config.train_ratio),
                "validation (10%)": int(targets['total'] * config.val_ratio),
                "test (5%)": int(targets['total'] * config.test_ratio),
            },
        },
        "estimated_api_calls": targets['total'] // config.batch_size,
    }


def print_dataset_recommendation(num_tools: int):
    """
    Print a nice summary of dataset recommendations.
    
    Args:
        num_tools: Number of tools
    """
    config = DataGenConfig()
    limits = config.get_limits(num_tools)
    targets = config.get_targets(num_tools)
    
    print(f"\n{'='*50}")
    print(f"📊 Dataset Recommendations for {num_tools} Tools")
    print(f"{'='*50}")
    print()
    print("SAMPLE SIZE LIMITS:")
    print(f"  ❌ Minimum:   {limits['minimum']['total']:>5} total ({limits['minimum']['per_tool']}/tool) - underfitting risk")
    print(f"  ✅ Suggested: {limits['suggested']['total']:>5} total ({limits['suggested']['per_tool']}/tool) - sweet spot")
    print(f"  ⚠️  Maximum:   {limits['maximum']['total']:>5} total ({limits['maximum']['per_tool']}/tool) - overfitting risk")
    print()
    print("USING SUGGESTED:")
    print(f"  Total: {targets['total']}")
    print(f"    - Single-tool (65%):   {targets['single_tool']}")
    print(f"    - Clarification (5%):  {targets['clarification']}")
    print(f"    - Follow-up (3.5%):    {targets['follow_up']}")
    print(f"    - Refusal (9%):        {targets['refusal']}")
    print(f"    - Casual (17.5%):      {targets['casual']}")
    print()
    print("SPLITS:")
    print(f"    - Train (85%):      {int(targets['total'] * config.train_ratio)}")
    print(f"    - Validation (10%): {int(targets['total'] * config.val_ratio)}")
    print(f"    - Test (5%):        {int(targets['total'] * config.test_ratio)}")
    print()
    print(f"Estimated API calls: ~{targets['total'] // config.batch_size}")
    print(f"{'='*50}\n")
