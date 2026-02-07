"""
REWOO Planner — generates an execution plan upfront.

Uses a THINK → PLAN prompt structure so the model reasons about which
tool to use before writing the call.  Tool rules are auto-generated
from tool schemas (via ``onsetlab.skills.generate_tool_rules``).
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple

from ..model.base import BaseModel
from ..tools.base import BaseTool
from ..skills import generate_tool_rules, generate_examples


# ---------------------------------------------------------------------------
# Prompt — THINK → PLAN structure
# ---------------------------------------------------------------------------

PLANNER_PROMPT = '''You are a tool-calling assistant. Select the right tool and provide correct parameters.

TOOLS:
{tool_rules}

EXAMPLES:
{examples}

FORMAT:
THINK: [1 sentence — which tool to use and why]
PLAN:
#E1 = tool_name(param="value")

RULES:
- Use EXACT tool names from the TOOLS list
- Include ALL required parameters (marked [ALL REQUIRED])
- Copy values exactly from the task (names, numbers, IDs)
- 1 step usually. Use 2-3 ONLY if the task explicitly requires chaining.
- To reference a previous result use #E1, #E2, etc.
{context_section}
Task: {task}

THINK:'''


class PlanResult:
    """Result from the Planner, including reasoning."""

    __slots__ = ("steps", "think")

    def __init__(self, steps: List[Dict[str, Any]], think: str = ""):
        self.steps = steps
        self.think = think


class Planner:
    """
    REWOO Planner — creates an execution plan upfront.

    The plan is a list of ``{id, tool, params, depends_on}`` dicts.
    Tool rules are auto-generated from schemas; no hardcoded skill
    definitions needed.
    """

    def __init__(self, model: BaseModel, tools: List[BaseTool], debug: bool = False):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self._tools_list = tools
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, task: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate an execution plan for *task*.

        Returns a list of step dicts (same shape as before).
        The THINK text is available via ``plan_with_reasoning()``.
        """
        result = self.plan_with_reasoning(task, context)
        return result.steps

    def plan_with_reasoning(
        self, task: str, context: Optional[str] = None,
    ) -> PlanResult:
        """
        Generate a plan **and** return the model's reasoning.

        Returns:
            PlanResult with ``.steps`` and ``.think``.
        """
        # Build prompt sections
        tool_rules = generate_tool_rules(self._tools_list)
        examples = generate_examples(self._tools_list, max_examples=2)

        context_section = ""
        if context:
            context_section = (
                f"\nCONTEXT (conversation history):\n{context}\n"
            )

        prompt = PLANNER_PROMPT.format(
            tool_rules=tool_rules,
            examples=examples,
            context_section=context_section,
            task=task,
        )

        if self.debug:
            print(f"\n[DEBUG] Planner prompt ({len(prompt)} chars):\n{prompt}\n")

        response = self.model.generate(
            prompt,
            temperature=0.0,
            max_tokens=250,
            stop_sequences=["\n\n\n", "Task:", "---"],
        )

        if self.debug:
            print(f"[DEBUG] Planner raw response:\n{response}\n")

        # Extract THINK and PLAN sections
        think, plan_text = self._split_think_plan(response)

        if self.debug:
            print(f"[DEBUG] THINK: {think}")
            print(f"[DEBUG] PLAN text: {plan_text}")

        steps = self._parse_plan(plan_text)
        steps = self._deduplicate_steps(steps)
        steps = self._validate_steps(steps)

        if self.debug:
            print(f"[DEBUG] Parsed steps: {steps}\n")

        return PlanResult(steps=steps, think=think)

    # ------------------------------------------------------------------
    # THINK / PLAN splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_think_plan(response: str) -> Tuple[str, str]:
        """
        Split the model's response into THINK and PLAN sections.

        Handles cases where the model:
        - Outputs "THINK: ... PLAN: #E1 = ..."
        - Outputs "THINK: ... #E1 = ..." (no explicit PLAN: label)
        - Outputs just "#E1 = ..." or "tool_name(...)" directly
        """
        response = response.strip()

        # Try to split on PLAN: label
        plan_match = re.search(r'\bPLAN\s*:', response, re.IGNORECASE)
        if plan_match:
            think = response[:plan_match.start()].strip()
            plan_text = response[plan_match.end():].strip()
            # Clean THINK: prefix if present
            think = re.sub(r'^THINK\s*:\s*', '', think, flags=re.IGNORECASE).strip()
            return think, plan_text

        # Try to split on first #E1 or tool call
        step_match = re.search(r'(#E\d+\s*=|[\w-]+\s*\()', response)
        if step_match:
            think = response[:step_match.start()].strip()
            plan_text = response[step_match.start():].strip()
            think = re.sub(r'^THINK\s*:\s*', '', think, flags=re.IGNORECASE).strip()
            return think, plan_text

        # Fallback: treat everything as plan text
        return "", response

    # ------------------------------------------------------------------
    # Plan parsing (robust, handles many SLM output formats)
    # ------------------------------------------------------------------

    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse plan from model response."""
        steps = []
        step_counter = 1

        # Clean up common model mistakes
        response = response.replace('expression*=', 'expression=')
        response = response.replace('*=', '=')
        # Handle colon syntax: param: "value" -> param="value"
        response = re.sub(r'(\w+):\s*"', r'\1="', response)
        response = re.sub(r'(\w+):\s*(\d)', r'\1=\2', response)

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that don't look like tool calls
            if not ('(' in line and ')' in line):
                continue

            # Remove numbered list prefixes: "1. ", "2. ", etc.
            line = re.sub(r'^\d+\.\s*', '', line)

            # Handle: #E1 = #E1 = ToolName(...) — remove duplicate
            line = re.sub(r'^(#E\d+)\s*=\s*#E\d+\s*=', r'\1 =', line)

            # Try pattern with #E prefix
            match = re.match(r'#E(\d+)\s*=\s*([\w-]+)\s*\((.*)\)', line)

            if match:
                step_num = int(match.group(1))
                tool_name = match.group(2)
                params_str = match.group(3)
            else:
                # Try pattern without #E prefix: ToolName(params)
                match = re.match(r'([\w-]+)\s*\((.*)\)', line)
                if not match:
                    continue

                tool_name = match.group(1)
                params_str = match.group(2)

                # Skip non-tool keywords
                if tool_name.lower() in [
                    'result', 'note', 'explanation', 'answer', 'think', 'plan',
                ]:
                    continue

                step_num = step_counter

            params = self._parse_params(params_str)
            depends_on = re.findall(r'#E\d+', params_str)

            steps.append({
                "id": f"#E{step_num}",
                "tool": tool_name,
                "params": params,
                "depends_on": depends_on,
            })

            step_counter = max(step_counter, step_num) + 1

        return steps

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dict. Handles arrays, objects, strings, numbers."""
        params: Dict[str, Any] = {}
        params_str = params_str.strip()

        if not params_str:
            return params

        i = 0
        while i < len(params_str):
            # Skip whitespace and commas
            while i < len(params_str) and params_str[i] in ' ,\t\n':
                i += 1
            if i >= len(params_str):
                break

            # Find parameter name
            name_match = re.match(r'(\w+)\s*=\s*', params_str[i:])
            if not name_match:
                break

            param_name = name_match.group(1)
            i += name_match.end()

            if i >= len(params_str):
                break

            char = params_str[i]

            if char == '"':
                end = params_str.find('"', i + 1)
                if end == -1:
                    end = len(params_str)
                value = params_str[i + 1:end]
                i = end + 1
                if value.lower() not in ('none', 'null', 'undefined'):
                    params[param_name] = value

            elif char == "'":
                end = params_str.find("'", i + 1)
                if end == -1:
                    end = len(params_str)
                value = params_str[i + 1:end]
                i = end + 1
                if value.lower() not in ('none', 'null', 'undefined'):
                    params[param_name] = value

            elif char == '[':
                value, end_pos = self._extract_bracket(params_str, i, '[', ']')
                i = end_pos
                try:
                    params[param_name] = json.loads(value)
                except json.JSONDecodeError:
                    params[param_name] = value

            elif char == '{':
                value, end_pos = self._extract_bracket(params_str, i, '{', '}')
                i = end_pos
                try:
                    params[param_name] = json.loads(value)
                except json.JSONDecodeError:
                    params[param_name] = value

            else:
                match = re.match(r'([^,\)\s]+)', params_str[i:])
                if match:
                    value = match.group(1).strip()
                    i += match.end()

                    if value.lower() in ('none', 'null', 'undefined'):
                        continue
                    if value.startswith('#E'):
                        params[param_name] = value
                    elif value.lower() == 'true':
                        params[param_name] = True
                    elif value.lower() == 'false':
                        params[param_name] = False
                    elif re.match(r'^-?\d+$', value):
                        params[param_name] = int(value)
                    elif re.match(r'^-?\d+\.\d+$', value):
                        params[param_name] = float(value)
                    else:
                        params[param_name] = value
                else:
                    i += 1

        # Handle positional arguments
        if not params:
            positional_matches = re.findall(r'"([^"]*)"', params_str)
            if positional_matches:
                for idx, val in enumerate(positional_matches):
                    params[f"_positional_{idx}"] = val
            elif '=' not in params_str:
                params["_positional_0"] = params_str.strip('"\'')

        return params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_bracket(s: str, start: int, open_b: str, close_b: str) -> tuple:
        """Extract content within balanced brackets."""
        depth = 0
        i = start
        while i < len(s):
            if s[i] == open_b:
                depth += 1
            elif s[i] == close_b:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1], i + 1
            elif s[i] == '"':
                i += 1
                while i < len(s) and s[i] != '"':
                    if s[i] == '\\':
                        i += 1
                    i += 1
            i += 1
        return s[start:], len(s)

    @staticmethod
    def _deduplicate_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate steps."""
        seen: set = set()
        unique: List[Dict[str, Any]] = []

        for step in steps:
            param_key = tuple(sorted((k, str(v)) for k, v in step["params"].items()))
            key = (step["tool"], param_key)
            if key not in seen:
                seen.add(key)
                unique.append(step)

        for i, step in enumerate(unique):
            step["id"] = f"#E{i + 1}"

        return unique

    def _normalize_tool_name(self, name: str) -> Optional[str]:
        """Strict tool name matching (exact or case-insensitive)."""
        if name in self.tools:
            return name

        name_lower = name.lower()
        for tool_name in self.tools:
            if tool_name.lower() == name_lower:
                return tool_name

        return None

    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate steps — check tool exists and has required params.

        Invalid steps get an ``error`` field so the Solver can explain.
        """
        validated: List[Dict[str, Any]] = []

        for step in steps:
            tool_name = step["tool"]

            normalized = self._normalize_tool_name(tool_name)

            if not normalized:
                available = list(self.tools.keys())
                suggestions = self._find_similar_tools(tool_name, available)

                error_msg = f"Tool '{tool_name}' not found."
                if suggestions:
                    error_msg += f" Did you mean: {', '.join(suggestions)}?"
                else:
                    error_msg += f" Available tools: {', '.join(available[:5])}"
                    if len(available) > 5:
                        error_msg += f"... ({len(available)} total)"

                if self.debug:
                    print(f"[DEBUG] {error_msg}")

                step["error"] = error_msg
                validated.append(step)
                continue

            step["tool"] = normalized

            tool = self.tools[normalized]
            tool_params = tool.parameters

            # Map positional args BEFORE validation
            step["params"] = self._map_positional_params(tool_params, step["params"])

            # Get required params
            if "required" in tool_params and isinstance(tool_params.get("required"), list):
                required = tool_params["required"]
            else:
                required = [
                    p for p, details in tool_params.items()
                    if isinstance(details, dict) and details.get("required")
                ]

            missing = [p for p in required if p not in step["params"]]
            if missing:
                error_msg = f"Tool '{normalized}' missing required params: {missing}"
                if self.debug:
                    print(f"[DEBUG] {error_msg}")
                step["error"] = error_msg

            validated.append(step)

        return validated

    @staticmethod
    def _map_positional_params(
        tool_params: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map positional arguments to parameter names in order."""
        positional_args = []
        i = 0
        while f"_positional_{i}" in params:
            positional_args.append(params.pop(f"_positional_{i}"))
            i += 1

        if not positional_args:
            return params

        required_params = []
        optional_params = []

        for param_name, param_info in tool_params.items():
            if isinstance(param_info, dict) and param_info.get("required"):
                required_params.append(param_name)
            else:
                optional_params.append(param_name)

        ordered_params = required_params + optional_params

        for idx, positional_value in enumerate(positional_args):
            if idx < len(ordered_params):
                params[ordered_params[idx]] = positional_value

        return params

    @staticmethod
    def _find_similar_tools(
        name: str, available: List[str], max_suggestions: int = 3,
    ) -> List[str]:
        """Find similar tool names for helpful suggestions."""
        name_lower = name.lower()
        name_words = set(name_lower.replace('_', ' ').replace('-', ' ').split())

        scored = []
        for tool_name in available:
            tool_lower = tool_name.lower()
            tool_words = set(tool_lower.replace('_', ' ').replace('-', ' ').split())
            overlap = len(name_words & tool_words)
            if overlap > 0:
                scored.append((tool_name, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in scored[:max_suggestions]]
