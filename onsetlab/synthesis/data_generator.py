"""
Single-Tool Data Generator
===========================
Training data generation for single-tool-per-turn architecture.

Key features:
- Single tool calls only (no multi-tool chains)
- Strong clarification handling (users often give incomplete info)
- Batched generation (10 examples per LLM call)
- Tool-aware sample sizing
"""

import json
import random
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DataGenConfig:
    """
    Configuration for data generation.
    
    SAMPLE SIZE RULES (per tool):
    - MINIMUM: 10 examples/tool → Risk of underfitting
    - SUGGESTED: 25 examples/tool → Sweet spot for 3B model
    - MAXIMUM: 40 examples/tool → Cap to avoid overfitting
    
    EXAMPLES BY TOOL COUNT:
    | Tools | Minimum | Suggested | Maximum |
    |-------|---------|-----------|---------|
    | 10    | 133     | 333       | 533     |
    | 15    | 200     | 500       | 800     |
    | 20    | 267     | 667       | 1067    |
    | 50    | 667     | 1667      | 2667    |
    
    Note: These are TOTAL examples (75% single-tool + 25% edge cases)
    """
    
    # Per-tool limits (ENFORCED)
    min_per_tool: int = 15            # Minimum - below this = underfitting
    examples_per_tool: int = 35       # Suggested - higher for better generalization
    max_per_tool: int = 50            # Maximum - 3B can handle more with LoRA
    
    # Distribution (must sum to 1.0)
    single_tool_ratio: float = 0.75   # 75% successful tool calls
    edge_case_ratio: float = 0.25     # 25% edge cases
    
    # Edge case breakdown (within the 25%)
    # IMPORTANT: casual needs higher % because small models default to tool calls
    clarification_ratio: float = 0.25  # 25% of edge cases = 6.25% total
    follow_up_ratio: float = 0.15      # 15% of edge cases = 3.75% total
    refusal_ratio: float = 0.20        # 20% of edge cases = 5% total (invalid tool, out of scope)
    casual_ratio: float = 0.40         # 40% of edge cases = 10% total (greetings, thanks, questions)
    
    # Split ratios
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    
    # Batch settings
    batch_size: int = 25              # Examples per LLM call (higher = faster generation)
    max_retries: int = 3
    
    # API settings
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    
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
    ):
        self.tools = tools
        self.problem_statement = problem_statement
        self.api_key = api_key
        self.config = config or DataGenConfig()
        self.system_prompt = system_prompt
        
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
        print(f"  - Single-tool: {targets['single_tool']} (75%)")
        print(f"  - Clarification: {targets['clarification']} (10%)")
        print(f"  - Follow-up: {targets['follow_up']} (5%)")
        print(f"  - Refusal: {targets['refusal']} (5%)")
        print(f"  - Casual: {targets['casual']} (5%)")
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
        """Generate successful single-tool examples with balanced coverage."""
        examples = []
        tool_counts = {self._get_name(t): 0 for t in self.tools}
        min_per_tool = max(5, target // len(self.tools))
        
        while len(examples) < target:
            # Pick tools that need more examples
            batch_tools = sorted(
                self.tools, 
                key=lambda t: tool_counts[self._get_name(t)]
            )[:5]
            
            batch = self._call_llm_single_tool(batch_tools)
            
            for ex in batch:
                tool = ex.get('tool')
                if tool in tool_counts:
                    tool_counts[tool] += 1
                    examples.append(ex)
            
            if len(examples) % 50 == 0:
                print(f"   ... {len(examples)}/{target}")
        
        return examples[:target]
    
    def _call_llm_single_tool(self, tools: list) -> list:
        """Generate batch of single-tool examples."""
        tools_desc = self._format_tools(tools)
        tool_names = [self._get_name(t) for t in tools]
        
        prompt = f"""Generate {self.config.batch_size} training examples for an AI assistant.

Context: {self.problem_statement}

TOOLS (use EXACT names and correct TYPES):
{tools_desc}

CRITICAL - ENUM VALUES ARE CASE-SENSITIVE:
- If a param shows enum(OPEN|CLOSED), you MUST use "OPEN" or "CLOSED" exactly (uppercase)
- Do NOT use lowercase like "open" or "closed" - the API will reject it
- Always copy enum values EXACTLY as shown in the param list above

CRITICAL RULES:
1. Tool name must be EXACTLY from the list above
2. The USER QUERY must CONTAIN all required parameter values explicitly
3. DO NOT make up/hallucinate values that the user didn't provide!
4. TYPES ARE CRITICAL:
   - array params MUST use [...] syntax: "labels": ["bug", "urgent"]
   - number params MUST NOT be quoted: "limit": 10 (NOT "limit": "10")
   - boolean params use true/false: "active": true

GOOD examples (user provides all info):
- "Create issue in facebook/react titled 'Bug in hooks'" → owner=facebook, repo=react, title=Bug in hooks ✓
- "List open issues in kubernetes/kubernetes with bug label" → all params from query ✓
- "Send 'Deploy complete' to channel C123456" → channel and message from query ✓

BAD examples (DO NOT generate these - info is missing):
- "Create an issue about the bug" → NO owner/repo specified! ✗
- "Comment on issue 73" → NO owner/repo specified! ✗
- "Send a message to the team" → NO channel specified! ✗

Output JSON array:
[
  {{"query": "user request with ALL required info", "tool": "exact_tool_name", "parameters": {{"param": "value_from_query"}}}}
]

Generate {self.config.batch_size} diverse examples where the user query contains ALL required information:"""

        return self._call_llm(prompt)
    
    # =========================================================================
    # Clarification Generation (most important edge case!)
    # =========================================================================
    
    def _generate_clarification(self, target: int) -> list:
        """Generate examples where user gives incomplete info."""
        examples = []
        
        while len(examples) < target:
            # Pick random tools
            batch_tools = random.sample(self.tools, min(5, len(self.tools)))
            batch = self._call_llm_clarification(batch_tools)
            examples.extend(batch)
        
        return examples[:target]
    
    def _call_llm_clarification(self, tools: list) -> list:
        """Generate clarification examples - user missing required params."""
        tools_desc = self._format_tools_with_required(tools)
        
        prompt = f"""Generate {self.config.batch_size} examples where user wants to use a tool but is MISSING required information.

Context: {self.problem_statement}

TOOLS AND THEIR REQUIRED PARAMS:
{tools_desc}

PATTERN:
1. User makes a request but leaves out 1-2 required parameters
2. Assistant asks a SPECIFIC question to get the missing info
3. Do NOT guess or use placeholders - ask!

EXAMPLES:
- User: "create an issue" → Missing: repo, title → Ask: "Which repository should I create the issue in, and what should the title be?"
- User: "send a message to the team" → Missing: channel, message → Ask: "Which Slack channel should I send to, and what's the message?"
- User: "schedule a meeting tomorrow" → Missing: time, title → Ask: "What time tomorrow, and what should I call the meeting?"

Output JSON array:
[
  {{"query": "incomplete user request", "response": "friendly question asking for missing info", "missing_params": ["param1", "param2"]}}
]

Generate {self.config.batch_size} diverse clarification examples:"""

        return self._call_llm(prompt)
    
    # =========================================================================
    # Follow-up Generation (context-aware queries)
    # =========================================================================
    
    def _generate_follow_up(self, target: int) -> list:
        """
        Generate examples where user asks a follow-up question with conversation context.
        
        These teach the model to:
        1. Parse injected conversation context
        2. Carry forward relevant parameters (owner, repo, etc.)
        3. Apply new filters based on the follow-up query
        """
        examples = []
        
        while len(examples) < target:
            # Pick random tools that have modifiable params
            batch_tools = random.sample(self.tools, min(5, len(self.tools)))
            batch = self._call_llm_follow_up(batch_tools)
            examples.extend(batch)
        
        return examples[:target]
    
    def _call_llm_follow_up(self, tools: list) -> list:
        """Generate follow-up examples with conversation context."""
        tools_desc = self._format_tools_with_required(tools)
        
        prompt = f"""Generate follow-up query training examples for a conversational agent.

TOOLS AVAILABLE:
{tools_desc}

SCENARIO: The user previously asked a question, got a response, and now asks a SHORT follow-up
that references the previous context. The follow-up query should be BRIEF (like real users).

FORMAT - The query MUST include conversation context in this EXACT format:
[Conversation Context]
User asked: <previous query>
Agent called <tool_name> -> <brief result summary>
[Current Query]
<short follow-up question>

EXAMPLES OF GOOD FOLLOW-UP PATTERNS:
1. "what about the open ones" (filter change)
2. "show me just 5" (limit change)  
3. "same but for the other repo" (param swap)
4. "any from last week?" (time filter)
5. "what's the most recent?" (sort/limit)
6. "now search in the body too" (param addition)

OUTPUT FORMAT (JSON array):
[
  {{"query": "[Conversation Context]\\nUser asked: find issues in owner/repo\\nAgent called list_issues -> Found 5 issues: #1 Bug, #2 Feature...\\n[Current Query]\\nwhat about the open ones", "tool": "list_issues", "params": {{"owner": "owner", "repo": "repo", "state": "OPEN"}}}}
]

RULES:
1. The follow-up query must be SHORT and natural (3-8 words)
2. Carry forward ALL relevant params from the previous query
3. Apply the modification from the follow-up (filter, limit, sort, etc.)
4. Use EXACT tool names and valid param values from the list above
5. ENUM VALUES ARE CASE-SENSITIVE: Use "OPEN"/"CLOSED" exactly as shown, NOT lowercase

Generate {self.config.batch_size} diverse follow-up examples:"""

        return self._call_llm(prompt)
    
    # =========================================================================
    # Refusal Generation
    # =========================================================================
    
    def _generate_refusal(self, target: int) -> list:
        """Generate examples where request is out of scope."""
        examples = []
        
        while len(examples) < target:
            batch = self._call_llm_refusal()
            examples.extend(batch)
        
        return examples[:target]
    
    def _call_llm_refusal(self) -> list:
        """Generate refusal examples - can't do what user asks."""
        tool_names = [self._get_name(t) for t in self.tools[:10]]
        
        prompt = f"""Generate {self.config.batch_size} examples where user asks for something the assistant CANNOT do.

Context: {self.problem_statement}
Available tools: {tool_names}

PATTERN:
1. User asks for something outside the available tools
2. Assistant politely declines and mentions what they CAN help with

EXAMPLES:
- User: "Can you book a flight for me?" → "I can't book flights, but I can help you with GitHub, Slack, or scheduling meetings."
- User: "Send an email to john@example.com" → "I don't have email capabilities, but I can send Slack messages if that helps."
- User: "What's the weather?" → "I can't check weather, but I can search the web or help with your calendar."

Output JSON array:
[
  {{"query": "out of scope request", "response": "polite decline mentioning available capabilities"}}
]

Generate {self.config.batch_size} diverse refusal examples:"""

        return self._call_llm(prompt)
    
    # =========================================================================
    # Casual Conversation Generation
    # =========================================================================
    
    def _generate_casual(self, target: int) -> list:
        """Generate casual conversation examples."""
        examples = []
        
        while len(examples) < target:
            batch = self._call_llm_casual()
            examples.extend(batch)
        
        return examples[:target]
    
    def _call_llm_casual(self) -> list:
        """Generate casual conversation - greetings, capability questions, thanks, etc."""
        
        # Get tool names for capability descriptions
        tool_names = [t.name for t in self.tools]
        tool_list = ", ".join(tool_names[:5]) + ("..." if len(tool_names) > 5 else "")
        
        prompt = f"""Generate {self.config.batch_size} casual conversation examples that DON'T need any tool call.

The agent has these tools: {tool_list}

CRITICAL: These are messages where the user is NOT asking to DO something.
The assistant should respond with friendly text, NO tool calls.

CATEGORIES TO COVER (mix all types):

1. GREETINGS:
   - "Hey!" → "Hey there! How can I help you today?"
   - "Good morning" → "Good morning! What can I help you with?"
   - "hi" → "Hi! What can I do for you?"

2. THANKS:
   - "Thanks!" → "You're welcome! Let me know if you need anything else."
   - "That was helpful" → "Glad I could help! Anything else?"
   - "perfect" → "Great! Let me know if you need anything else."

3. CAPABILITY QUESTIONS:
   - "What can you do?" → "I can help with GitHub issues and Slack. I can list issues, create them, add comments, and send Slack messages. What would you like me to do?"
   - "What are you?" → "I'm an assistant that helps manage GitHub issues and Slack messages."
   - "help" → "I'm here to help! I can work with GitHub issues and Slack. What do you need?"

4. ACKNOWLEDGMENTS:
   - "ok" → "Got it! Let me know when you need something."
   - "I see" → "Yep! Anything else?"
   - "cool" → "Glad that works! What's next?"

5. CONFUSION:
   - "I don't understand" → "No problem! What would you like me to explain?"
   - "wait what" → "Sorry if I was unclear. How can I help?"

Output JSON array:
[
  {{"query": "casual message", "response": "friendly response"}}
]

Generate {self.config.batch_size} DIVERSE examples from ALL categories:"""

        return self._call_llm(prompt)
    
    # =========================================================================
    # LLM Calling
    # =========================================================================
    
    def _call_llm(self, prompt: str) -> list:
        """Make LLM API call."""
        if self.client is None:
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
                else:
                    response = self.client.messages.create(
                        model=self.config.anthropic_model,
                        max_tokens=3000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text.strip()
                
                # Parse JSON
                examples = self._parse_json(content)
                
                # Validate tool names
                valid = []
                for ex in examples:
                    if self._validate_example(ex):
                        valid.append(ex)
                
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
    
    def _validate_example(self, ex: dict) -> bool:
        """Validate and fix example, including type coercion."""
        # Skip placeholders
        if self._has_placeholder(str(ex)):
            return False
        
        # Edge cases without tool are valid
        if ex.get('tool') is None and ex.get('response'):
            return True
        
        # Fix tool name if needed
        tool = ex.get('tool')
        if tool:
            actual_tool = None
            if tool in self.tool_names:
                actual_tool = tool
            else:
                # Try normalized
                normalized = tool.lower().replace("-", "_")
                if normalized in self.tool_map:
                    ex['tool'] = self.tool_map[normalized]
                    actual_tool = self.tool_map[normalized]
            
            if not actual_tool:
                return False
            
            # Fix parameter types based on schema
            params = ex.get('parameters', {})
            if params:
                self._fix_param_types(actual_tool, params)
            
            return True
        
        return True
    
    def _fix_param_types(self, tool_name: str, params: dict):
        """
        Coerce parameter values to correct types based on tool schema.
        
        Common LLM mistakes:
        - "10" instead of 10 (number)
        - "bug" instead of ["bug"] (array)
        - "true" instead of true (boolean)
        - null for optional params (remove them)
        """
        # Find the tool schema
        tool_schema = None
        for t in self.tools:
            name = self._get_name(t)
            if name == tool_name:
                tool_schema = t
                break
        
        if not tool_schema:
            return
        
        schema_params = self._get_params(tool_schema)
        required_params = set(self._get_required(tool_schema))
        
        # First pass: remove None values for optional params
        params_to_remove = []
        for param_name, value in params.items():
            if value is None and param_name not in required_params:
                params_to_remove.append(param_name)
        for param_name in params_to_remove:
            del params[param_name]
        
        # Second pass: fix types
        for param_name, value in list(params.items()):
            if param_name not in schema_params:
                continue
            
            param_info = schema_params[param_name]
            if not isinstance(param_info, dict):
                continue
            
            expected_type = param_info.get('type', 'string')
            
            # Handle None for required params - use default or empty value
            if value is None:
                if expected_type == 'array':
                    params[param_name] = []
                elif expected_type in ('number', 'integer'):
                    params[param_name] = 0
                elif expected_type == 'boolean':
                    params[param_name] = False
                elif expected_type == 'string':
                    params[param_name] = ""
                continue
            
            # Fix array: "bug" -> ["bug"], "bug,feature" -> ["bug", "feature"]
            if expected_type == 'array':
                if isinstance(value, str):
                    if ',' in value:
                        params[param_name] = [v.strip() for v in value.split(',')]
                    else:
                        params[param_name] = [value] if value else []
                elif not isinstance(value, list):
                    # Convert any other type to single-element array
                    params[param_name] = [value] if value else []
            
            # Fix number/integer: "10" -> 10
            elif expected_type in ('number', 'integer'):
                if isinstance(value, str):
                    try:
                        if '.' in value:
                            params[param_name] = float(value)
                        else:
                            params[param_name] = int(value)
                    except ValueError:
                        params[param_name] = 0
                elif isinstance(value, bool):
                    # bool is subclass of int, convert explicitly
                    params[param_name] = 1 if value else 0
            
            # Fix boolean: "true" -> True, "false" -> False
            elif expected_type == 'boolean':
                if isinstance(value, str):
                    params[param_name] = value.lower() in ('true', 'yes', '1')
                elif isinstance(value, (int, float)):
                    params[param_name] = bool(value)
            
            # Fix string: convert non-strings to strings
            elif expected_type == 'string' and not isinstance(value, str):
                if value is not None:
                    params[param_name] = str(value)
            
            # Fix enum values: case-sensitive matching
            # If schema has enum, find the correct case version
            enum_values = param_info.get('enum')
            if enum_values and isinstance(value, str):
                # Build case-insensitive lookup
                enum_map = {str(v).lower(): v for v in enum_values}
                value_lower = value.lower()
                if value_lower in enum_map:
                    params[param_name] = enum_map[value_lower]
    
    def _has_placeholder(self, text: str) -> bool:
        """Check for placeholder patterns."""
        patterns = [
            r'\{\{[^{}]+\}\}',        # {{date}}
            r'<[A-Z][A-Z_]+>',        # <NAME>
            r'\[[A-Z][A-Z_]*\]',      # [DATE]
            r'PLACEHOLDER',
            r'YOUR_\w+_HERE',
        ]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False
    
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
        
        # Assistant response
        if example_type in ("tool_call", "follow_up"):
            # Tool call + result + final response
            # follow_up is the same as tool_call but with context in the query
            tool = example.get("tool")
            params = example.get("parameters", example.get("params", {}))
            
            tool_json = json.dumps({"tool": tool, "parameters": params})
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{tool_json}\n</tool_call>"
            })
            
            # Simulated result
            result = self._simulate_result(tool, params)
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
        
        # Memory operations (check FIRST before generic patterns)
        if 'memory' in name:
            if 'set' in name:
                return json.dumps({"stored": True, "key": params.get("key")})
            if 'get' in name:
                return json.dumps({"key": params.get("key"), "value": "stored_value"})
            if 'list' in name:
                return json.dumps({"keys": ["user_name", "project", "preference"]})
            if 'delete' in name:
                return json.dumps({"deleted": True, "key": params.get("key")})
        
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
        
        # Memory
        if data.get('stored'):
            return f"✓ Saved '{params.get('key')}' to memory."
        if 'value' in data and 'key' in data:
            return f"The value for '{data['key']}' is: {data['value']}"
        if 'keys' in data:
            keys = data['keys']
            return f"I have {len(keys)} items in memory: {', '.join(keys)}"
        
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
    
    Returns:
        dict with 'train', 'validation', 'test' lists
    """
    config = DataGenConfig(examples_per_tool=examples_per_tool)
    
    generator = DataGenerator(
        tools=tools,
        problem_statement=problem_statement,
        api_key=api_key,
        config=config,
        system_prompt=system_prompt
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
                "single_tool (75%)": targets['single_tool'],
                "clarification (10%)": targets['clarification'],
                "follow_up (5%)": targets['follow_up'],
                "refusal (5%)": targets['refusal'],
                "casual (5%)": targets['casual'],
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
    print(f"    - Single-tool (75%):   {targets['single_tool']}")
    print(f"    - Clarification (10%): {targets['clarification']}")
    print(f"    - Follow-up (5%):      {targets['follow_up']}")
    print(f"    - Refusal (5%):        {targets['refusal']}")
    print(f"    - Casual (5%):         {targets['casual']}")
    print()
    print("SPLITS:")
    print(f"    - Train (85%):      {int(targets['total'] * config.train_ratio)}")
    print(f"    - Validation (10%): {int(targets['total'] * config.val_ratio)}")
    print(f"    - Test (5%):        {int(targets['total'] * config.test_ratio)}")
    print()
    print(f"Estimated API calls: ~{targets['total'] // config.batch_size}")
    print(f"{'='*50}\n")
