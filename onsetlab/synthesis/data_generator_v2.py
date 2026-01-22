"""
High-Quality Data Generator v2
==============================
Production-grade training data generation with:
- Systematic coverage (query categories)
- Multi-step examples
- Tool result simulation
- Quality validation
- Diversity checking
- Train/Val/Test split
"""

import json
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


@dataclass
class DataGenConfig:
    """Configuration for data generation."""
    # Target counts
    examples_per_tool: int = 50  # ~50 per tool
    multi_step_ratio: float = 0.15  # 15% multi-step
    
    # Category weights (must sum to 1.0)
    category_weights: dict = field(default_factory=lambda: {
        "direct": 0.30,       # Clear, direct requests
        "natural": 0.25,      # Casual, conversational
        "complex": 0.15,      # Multi-step tasks
        "ambiguous": 0.10,    # Need clarification
        "edge_cases": 0.10,   # Unusual but valid
        "out_of_scope": 0.05, # Should decline
        "error_handling": 0.05,  # Invalid inputs
    })
    
    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Quality thresholds
    min_query_length: int = 10
    max_query_length: int = 200
    diversity_threshold: float = 0.75  # Max similarity allowed
    
    # API settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"  # Use GPT-4 for high-quality data


class ToolResultSimulator:
    """Generates realistic fake tool results for multi-step training."""
    
    def __init__(self):
        self.fake_events = [
            {"id": "evt_001", "title": "Team Standup", "time": "09:00", "duration": 30},
            {"id": "evt_002", "title": "Project Review", "time": "14:00", "duration": 60},
            {"id": "evt_003", "title": "1:1 with Manager", "time": "11:00", "duration": 30},
            {"id": "evt_004", "title": "Lunch with Sarah", "time": "12:30", "duration": 60},
            {"id": "evt_005", "title": "Client Call", "time": "16:00", "duration": 45},
        ]
    
    def simulate(self, tool_name: str, params: dict) -> dict:
        """Generate realistic tool result."""
        
        if "list" in tool_name.lower() or "get" in tool_name.lower():
            # Return 0-3 items
            num_items = random.choice([0, 1, 2, 3])
            return {
                "status": "success",
                "events": random.sample(self.fake_events, min(num_items, len(self.fake_events)))
            }
        
        elif "add" in tool_name.lower() or "create" in tool_name.lower():
            return {
                "status": "created",
                "id": f"evt_{uuid.uuid4().hex[:8]}",
                "title": params.get("title", "New Event")
            }
        
        elif "delete" in tool_name.lower() or "remove" in tool_name.lower():
            # 10% chance of error
            if random.random() < 0.1:
                return {"status": "error", "message": "Event not found"}
            return {"status": "deleted", "id": params.get("event_id", "unknown")}
        
        elif "update" in tool_name.lower() or "modify" in tool_name.lower():
            return {
                "status": "updated",
                "id": params.get("event_id", "unknown"),
                "changes": params
            }
        
        else:
            return {"status": "success", "result": "Operation completed"}


class QualityValidator:
    """Validates training examples for quality."""
    
    def __init__(self, tools: list, config: DataGenConfig):
        self.tools = {t.name: t for t in tools}
        self.config = config
        self.seen_queries = []
    
    def validate(self, example: dict) -> tuple[bool, list[str]]:
        """Validate a single example. Returns (is_valid, errors)."""
        errors = []
        
        query = example.get("query", "")
        
        # Query checks
        if len(query) < self.config.min_query_length:
            errors.append(f"Query too short: {len(query)} chars")
        if len(query) > self.config.max_query_length:
            errors.append(f"Query too long: {len(query)} chars")
        if query.isupper():
            errors.append("Query is all caps")
        
        # Tool call checks
        tool_call = example.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("tool") or tool_call.get("name")
            params = tool_call.get("parameters") or tool_call.get("arguments", {})
            
            # Valid tool name
            if tool_name not in self.tools:
                errors.append(f"Invalid tool: {tool_name}")
            else:
                # Check required params
                tool = self.tools[tool_name]
                for req_param in tool.required_params:
                    if req_param not in params:
                        errors.append(f"Missing required param: {req_param}")
            
            # Check for placeholder values
            placeholder_patterns = [
                r'\bINSERT\b', r'\bPLACEHOLDER\b', r'\bXXX\b', 
                r'\bTODO\b', r'\.\.\.', r'\bexample\b',
                r'\[.*\]',  # [something]
                r'<.*>',    # <something>
            ]
            for key, value in params.items():
                for pattern in placeholder_patterns:
                    if re.search(pattern, str(value), re.IGNORECASE):
                        errors.append(f"Placeholder in {key}: {value}")
                        break
        
        # Diversity check
        if self._is_too_similar(query):
            errors.append("Too similar to existing example")
        else:
            self.seen_queries.append(query)
        
        return len(errors) == 0, errors
    
    def _is_too_similar(self, query: str) -> bool:
        """Check if query is too similar to existing ones."""
        query_words = set(query.lower().split())
        
        for existing in self.seen_queries[-100:]:  # Check last 100
            existing_words = set(existing.lower().split())
            
            if not query_words or not existing_words:
                continue
            
            intersection = len(query_words & existing_words)
            union = len(query_words | existing_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > self.config.diversity_threshold:
                return True
        
        return False


class HighQualityDataGenerator:
    """
    Production-grade training data generator.
    
    Usage:
        generator = HighQualityDataGenerator(
            tools=tools,
            problem_statement="Calendar assistant",
            api_key="sk-..."
        )
        datasets = generator.generate()
        # Returns {"train": [...], "validation": [...], "test": [...]}
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
        self.system_prompt = system_prompt or ""
        
        self.validator = QualityValidator(tools, self.config)
        self.result_simulator = ToolResultSimulator()
        
        # Setup LLM client
        self._setup_client()
    
    def _setup_client(self):
        """Setup LLM client."""
        if self.config.llm_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.llm_provider}")
    
    def generate(self) -> dict:
        """
        Generate high-quality training data.
        
        Returns:
            dict with "train", "validation", "test" splits
        """
        print(f"\n{'='*60}")
        print("🔬 High-Quality Data Generator v2")
        print(f"{'='*60}")
        print(f"Tools: {len(self.tools)}")
        print(f"Target per tool: {self.config.examples_per_tool}")
        print(f"Multi-step ratio: {self.config.multi_step_ratio:.0%}")
        
        all_examples = []
        
        # Calculate targets per category
        total_target = len(self.tools) * self.config.examples_per_tool
        
        # Generate by category
        for category, weight in self.config.category_weights.items():
            category_target = int(total_target * weight)
            print(f"\n📝 Generating {category} examples ({category_target})...")
            
            if category == "complex":
                # Multi-step examples
                examples = self._generate_multi_step(category_target)
            elif category == "out_of_scope":
                examples = self._generate_out_of_scope(category_target)
            elif category == "ambiguous":
                examples = self._generate_ambiguous(category_target)
            elif category == "error_handling":
                examples = self._generate_error_handling(category_target)
            else:
                examples = self._generate_single_step(category, category_target)
            
            all_examples.extend(examples)
            print(f"   ✅ Generated {len(examples)} valid examples")
        
        # Shuffle
        random.shuffle(all_examples)
        
        # Split
        datasets = self._split_data(all_examples)
        
        # Summary
        self._print_summary(datasets)
        
        return datasets
    
    def _generate_single_step(self, category: str, target: int) -> list:
        """Generate single-step tool call examples."""
        examples = []
        attempts = 0
        max_attempts = target * 3
        
        category_prompts = {
            "direct": "direct and clear request",
            "natural": "casual, conversational request like a real user would say",
            "edge_cases": "unusual but valid edge case request",
        }
        
        style = category_prompts.get(category, "natural request")
        
        while len(examples) < target and attempts < max_attempts:
            attempts += 1
            
            # Pick a random tool
            tool = random.choice(self.tools)
            
            try:
                example = self._generate_example_for_tool(tool, style)
                is_valid, errors = self.validator.validate(example)
                
                if is_valid:
                    examples.append(example)
                
            except Exception as e:
                continue
        
        return examples
    
    def _generate_example_for_tool(self, tool, style: str) -> dict:
        """Generate a single example for a tool."""
        tool_desc = f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.parameters)}"
        
        prompt = f"""Generate a {style} that would require using this tool:

{tool_desc}

Today's date is {datetime.now().strftime('%Y-%m-%d')}.

Respond with JSON only:
{{
    "query": "user's natural language request",
    "tool_call": {{
        "tool": "{tool.name}",
        "parameters": {{...actual values, not placeholders...}}
    }}
}}

Use realistic values. No placeholders like [DATE] or INSERT_HERE."""

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        # Extract JSON
        if "```" in content:
            content = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            content = content.group(1) if content else "{}"
        
        return json.loads(content)
    
    def _generate_multi_step(self, target: int) -> list:
        """Generate multi-step examples requiring 2-3 tool calls."""
        examples = []
        
        # Multi-step scenarios
        scenarios = self._get_multi_step_scenarios()
        
        for i in range(target):
            scenario = random.choice(scenarios)
            
            try:
                example = self._generate_multi_step_example(scenario)
                examples.append(example)
            except Exception as e:
                continue
        
        return examples
    
    def _get_multi_step_scenarios(self) -> list:
        """Get multi-step scenario templates."""
        tool_names = [t.name for t in self.tools]
        scenarios = []
        
        # Find tool combinations
        for t1 in self.tools:
            for t2 in self.tools:
                if t1.name != t2.name:
                    scenarios.append({
                        "tools": [t1.name, t2.name],
                        "description": f"First use {t1.name}, then use {t2.name}"
                    })
        
        return scenarios[:10]  # Limit to 10 scenarios
    
    def _generate_multi_step_example(self, scenario: dict) -> dict:
        """Generate a multi-step conversation example."""
        tools_info = []
        for tool_name in scenario["tools"]:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                tools_info.append(f"- {tool.name}: {tool.description}")
        
        prompt = f"""Generate a multi-step user request that requires using these tools in sequence:

{chr(10).join(tools_info)}

The user should make ONE request that requires BOTH tools.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

Respond with JSON:
{{
    "query": "user's complex request",
    "steps": [
        {{
            "tool_call": {{"tool": "...", "parameters": {{...}}}},
            "result": {{...simulated result...}},
            "reasoning": "why this tool was called"
        }},
        {{
            "tool_call": {{"tool": "...", "parameters": {{...}}}},
            "result": {{...simulated result...}},
            "reasoning": "why this tool was called next"
        }}
    ],
    "final_response": "what the assistant says after completing all steps"
}}"""

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            content = content.group(1) if content else "{}"
        
        return json.loads(content)
    
    def _generate_out_of_scope(self, target: int) -> list:
        """Generate out-of-scope examples (should decline)."""
        examples = []
        
        out_of_scope_queries = [
            "Send an email to John",
            "Order me a pizza",
            "What's the weather like?",
            "Play some music",
            "Set a timer for 5 minutes",
            "Translate this to Spanish",
            "Search the web for cats",
            "Take a screenshot",
        ]
        
        for i in range(min(target, len(out_of_scope_queries))):
            examples.append({
                "query": out_of_scope_queries[i],
                "response": "I'm sorry, but I can only help with calendar-related tasks. I can list events, add new events, or delete events. Is there something calendar-related I can help you with?",
                "tool_call": None  # No tool should be called
            })
        
        return examples
    
    def _generate_ambiguous(self, target: int) -> list:
        """Generate ambiguous examples that need clarification."""
        examples = []
        
        ambiguous_queries = [
            ("Add a meeting", "I'd be happy to add a meeting. Could you please tell me:\n1. What's the title/subject?\n2. What date and time?\n3. How long should it be?"),
            ("Delete that event", "I'd be happy to delete an event. Which event would you like me to delete? Please provide the event name or ID."),
            ("Change my meeting", "I can help modify a meeting. Which meeting would you like to change, and what would you like to update (time, date, title)?"),
            ("Schedule something", "I'd be happy to schedule something. What would you like to add to your calendar? Please provide details like title, date, and time."),
        ]
        
        for query, response in ambiguous_queries[:target]:
            examples.append({
                "query": query,
                "response": response,
                "tool_call": None  # Need clarification first
            })
        
        return examples
    
    def _generate_error_handling(self, target: int) -> list:
        """Generate error handling examples."""
        examples = []
        
        error_scenarios = [
            {
                "query": "Delete event nonexistent123",
                "tool_call": {"tool": "delete-event", "parameters": {"event_id": "nonexistent123"}},
                "tool_result": {"status": "error", "message": "Event not found"},
                "response": "I couldn't find an event with that ID. Could you please check the event ID or tell me the event name so I can help you find it?"
            },
        ]
        
        for scenario in error_scenarios[:target]:
            examples.append(scenario)
        
        return examples
    
    def _split_data(self, examples: list) -> dict:
        """Split data into train/val/test."""
        total = len(examples)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)
        
        return {
            "train": examples[:train_end],
            "validation": examples[train_end:val_end],
            "test": examples[val_end:],
        }
    
    def _print_summary(self, datasets: dict):
        """Print generation summary."""
        print(f"\n{'='*60}")
        print("📊 Generation Summary")
        print(f"{'='*60}")
        print(f"Train: {len(datasets['train'])} examples")
        print(f"Validation: {len(datasets['validation'])} examples")
        print(f"Test: {len(datasets['test'])} examples")
        print(f"Total: {sum(len(d) for d in datasets.values())} examples")
    
    def save(self, output_dir: str, datasets: dict):
        """Save datasets to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, examples in datasets.items():
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"💾 Saved {file_path}")
        
        return output_path


# Convenience function
def generate_high_quality_data(
    tools: list,
    problem_statement: str,
    api_key: str,
    output_dir: str = "./training_data",
    examples_per_tool: int = 50,
) -> dict:
    """
    Generate high-quality training data.
    
    Args:
        tools: List of ToolSchema objects
        problem_statement: Description of the agent
        api_key: OpenAI API key
        output_dir: Where to save the data
        examples_per_tool: Target examples per tool
    
    Returns:
        dict with train/validation/test splits
    """
    config = DataGenConfig(examples_per_tool=examples_per_tool)
    
    generator = HighQualityDataGenerator(
        tools=tools,
        problem_statement=problem_statement,
        api_key=api_key,
        config=config,
    )
    
    datasets = generator.generate()
    generator.save(output_dir, datasets)
    
    return datasets
