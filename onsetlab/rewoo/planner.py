"""REWOO Planner - generates execution plan upfront."""

import re
import json
from typing import List, Dict, Any, Optional

from ..model.base import BaseModel
from ..tools.base import BaseTool


# Planner prompt - optimized for small models
PLANNER_PROMPT = '''Call ONE tool to complete the task.

Tools:
{tools_description}
{examples_section}
Rules:
- Copy exact values from task
- Use format: tool(param="value")
{context_section}
Task: {task}
#E1 ='''


class Planner:
    """REWOO Planner - creates execution plan upfront."""
    
    def __init__(self, model: BaseModel, tools: List[BaseTool], debug: bool = False):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.debug = debug
    
    def plan(self, task: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate execution plan for a task."""
        tools_desc = self._format_tools_description()
        examples = self._generate_examples()
        
        context_section = ""
        if context:
            context_section = f"\nCONTEXT (previous results you can reference):\n{context}\n"
        
        examples_section = ""
        if examples:
            examples_section = f"\nEXAMPLES:\n{examples}\n"
        
        prompt = PLANNER_PROMPT.format(
            tools_description=tools_desc,
            examples_section=examples_section,
            context_section=context_section,
            task=task
        )
        
        response = self.model.generate(
            prompt,
            temperature=0.0,
            max_tokens=150,
            stop_sequences=["\n\n", "\n#E2", "Note:", "Explanation:", "---", "Task:"],
        )
        
        # Clean response and ensure #E1 prefix
        response = response.strip()
        # Remove duplicate #E1 if model added it
        if response.startswith('#E1'):
            full_response = response
        else:
            full_response = "#E1 = " + response
        
        if self.debug:
            print(f"\n[DEBUG] Planner raw response:\n{full_response}\n")
        
        steps = self._parse_plan(full_response)
        steps = self._deduplicate_steps(steps)
        steps = self._validate_steps(steps)
        
        if self.debug:
            print(f"[DEBUG] Parsed steps: {steps}\n")
        
        return steps
    
    def _generate_examples(self, max_examples: int = 2) -> str:
        """Generate minimal concrete examples for small models."""
        examples = []
        
        for name, tool in list(self.tools.items())[:max_examples]:
            tool_params = tool.parameters
            
            if "properties" in tool_params:
                params = tool_params.get("properties", {})
                required = tool_params.get("required", [])
            else:
                params = tool_params
                required = [p for p, d in params.items() 
                           if isinstance(d, dict) and d.get("required")]
            
            # Build example with required params only
            example_params = []
            for param_name, param_info in params.items():
                if param_name in required or (isinstance(param_info, dict) and param_info.get("required")):
                    if not isinstance(param_info, dict):
                        continue
                    param_type = param_info.get("type", "string")
                    
                    # Use concrete realistic values
                    if "enum" in param_info:
                        example_params.append(f'{param_name}="{param_info["enum"][0]}"')
                    elif param_type == "integer":
                        example_params.append(f'{param_name}=1')
                    elif param_type == "number":
                        example_params.append(f'{param_name}=1.0')
                    elif param_type == "boolean":
                        example_params.append(f'{param_name}=true')
                    elif param_type == "array":
                        example_params.append(f'{param_name}=["item"]')
                    else:
                        example_params.append(f'{param_name}="example"')
            
            params_str = ", ".join(example_params) if example_params else ""
            examples.append(f"{name}({params_str})")
        
        return "Examples: " + ", ".join(examples) if examples else ""
    
    def _format_tools_description(self) -> str:
        """Format tools for SLM-optimized prompts - minimal, clear, no confusing syntax."""
        lines = []
        
        for name, tool in self.tools.items():
            tool_params = tool.parameters
            
            # Handle both formats
            if "properties" in tool_params:
                params = tool_params.get("properties", {})
                required = tool_params.get("required", [])
            else:
                params = tool_params
                required = [p for p, details in params.items() 
                           if isinstance(details, dict) and details.get("required")]
            
            # Only show required params to reduce confusion
            req_parts = []
            opt_parts = []
            
            for p, details in params.items():
                if not isinstance(details, dict):
                    continue
                
                p_type = details.get("type", "string")
                is_req = p in required or details.get("required")
                
                # Format value hint - SLM friendly (no confusing {a|b} syntax)
                if "enum" in details:
                    # Show first enum value as example, mention others exist
                    enum_vals = details["enum"]
                    if len(enum_vals) <= 3:
                        hint = " or ".join(f'"{v}"' for v in enum_vals)
                    else:
                        hint = f'"{enum_vals[0]}" (or: {", ".join(enum_vals[1:3])}...)'
                    part = f'{p}={hint}'
                elif p_type == "array":
                    part = f'{p}=[...]'
                elif p_type == "boolean":
                    part = f'{p}=true'
                elif p_type in ("integer", "number"):
                    part = f'{p}=N'
                else:
                    part = f'{p}="..."'
                
                if is_req:
                    req_parts.append(part)
                else:
                    opt_parts.append(part)
            
            # Build compact signature - required params only for most tools
            if req_parts:
                sig = f"{name}({', '.join(req_parts)})"
            else:
                sig = f"{name}()"
            
            # Truncate description to 80 chars
            desc = tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
            lines.append(f"{sig} - {desc}")
        
        return "\n".join(lines)
    
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
            
            # Handle: #E1 = #E1 = ToolName(...) or #E1 =#E1 = by removing duplicate
            line = re.sub(r'^(#E\d+)\s*=\s*#E\d+\s*=', r'\1 =', line)
            
            # Try pattern with #E prefix: #E1 = ToolName(params) or #E1 = ToolName()
            # Use [\w-]+ to match tool names with hyphens (e.g., API-get-self)
            match = re.match(r'#E(\d+)\s*=\s*([\w-]+)\s*\((.*)\)', line)
            
            if match:
                step_num = int(match.group(1))
                tool_name = match.group(2)
                params_str = match.group(3)
            else:
                # Try pattern without #E prefix: ToolName(params) or ToolName()
                match = re.match(r'([\w-]+)\s*\((.*)\)', line)
                if not match:
                    continue
                
                tool_name = match.group(1)
                params_str = match.group(2)
                
                # Skip if tool name looks like a non-tool keyword
                if tool_name.lower() in ['result', 'note', 'explanation', 'answer']:
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
    
    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dict. Handles arrays, objects, strings, numbers."""
        params = {}
        params_str = params_str.strip()
        
        if not params_str:
            return params
        
        # Try to extract param=value pairs, handling complex types
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
                # No more named params, check for positional
                break
            
            param_name = name_match.group(1)
            i += name_match.end()
            
            if i >= len(params_str):
                break
            
            # Parse the value based on what character we see
            char = params_str[i]
            
            if char == '"':
                # Quoted string
                end = params_str.find('"', i + 1)
                if end == -1:
                    end = len(params_str)
                value = params_str[i+1:end]
                i = end + 1
                # Skip None/null string values
                if value.lower() not in ('none', 'null', 'undefined'):
                    params[param_name] = value
            
            elif char == "'":
                # Single-quoted string
                end = params_str.find("'", i + 1)
                if end == -1:
                    end = len(params_str)
                value = params_str[i+1:end]
                i = end + 1
                if value.lower() not in ('none', 'null', 'undefined'):
                    params[param_name] = value
            
            elif char == '[':
                # Array - find matching ]
                value, end_pos = self._extract_bracket(params_str, i, '[', ']')
                i = end_pos
                try:
                    params[param_name] = json.loads(value)
                except json.JSONDecodeError:
                    params[param_name] = value
            
            elif char == '{':
                # Object - find matching }
                value, end_pos = self._extract_bracket(params_str, i, '{', '}')
                i = end_pos
                try:
                    params[param_name] = json.loads(value)
                except json.JSONDecodeError:
                    params[param_name] = value
            
            else:
                # Unquoted value (number, boolean, #E ref, or plain string)
                match = re.match(r'([^,\)\s]+)', params_str[i:])
                if match:
                    value = match.group(1).strip()
                    i += match.end()
                    
                    # Skip None/null
                    if value.lower() in ('none', 'null', 'undefined'):
                        continue
                    
                    # Parse value type
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
        
        # Handle positional arguments if no named params found
        if not params:
            positional_matches = re.findall(r'"([^"]*)"', params_str)
            if positional_matches:
                for idx, val in enumerate(positional_matches):
                    params[f"_positional_{idx}"] = val
            elif '=' not in params_str:
                params["_positional_0"] = params_str.strip('"\'')
        
        return params
    
    def _extract_bracket(self, s: str, start: int, open_b: str, close_b: str) -> tuple:
        """Extract content within balanced brackets."""
        depth = 0
        i = start
        while i < len(s):
            if s[i] == open_b:
                depth += 1
            elif s[i] == close_b:
                depth -= 1
                if depth == 0:
                    return s[start:i+1], i + 1
            elif s[i] == '"':
                # Skip quoted strings
                i += 1
                while i < len(s) and s[i] != '"':
                    if s[i] == '\\':
                        i += 1
                    i += 1
            i += 1
        return s[start:], len(s)
    
    def _deduplicate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate steps."""
        seen = set()
        unique = []
        
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
        """
        Validate tool name exists - strict matching like effGen.
        
        Only allows:
        1. Exact match
        2. Case-insensitive match
        
        No fuzzy matching - if tool doesn't exist, return None
        and let the error propagate clearly.
        """
        # Exact match
        if name in self.tools:
            return name
        
        # Case-insensitive match
        name_lower = name.lower()
        for tool_name in self.tools:
            if tool_name.lower() == name_lower:
                return tool_name
        
        # No match found - don't guess
        return None
    
    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate steps - check tool exists and has required params.
        
        Returns validated steps. Invalid steps are converted to error steps
        so the Solver can explain what went wrong.
        """
        validated = []
        
        for step in steps:
            tool_name = step["tool"]
            
            # Try to find the tool (strict matching)
            normalized = self._normalize_tool_name(tool_name)
            
            if not normalized:
                # Tool not found - create error step with helpful message
                available = list(self.tools.keys())
                
                # Find similar tool names for suggestion
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
                
                # Add as error step so Solver sees it
                step["error"] = error_msg
                validated.append(step)
                continue
            
            # Update step with normalized name
            step["tool"] = normalized
            
            tool = self.tools[normalized]
            tool_params = tool.parameters
            
            # Map positional args to param names BEFORE validation
            step["params"] = self._map_positional_params(tool_params, step["params"])
            
            # Get required params - handle both formats
            if "required" in tool_params and isinstance(tool_params.get("required"), list):
                required = tool_params["required"]
            else:
                required = [p for p, details in tool_params.items()
                           if isinstance(details, dict) and details.get("required")]
            
            missing = [p for p in required if p not in step["params"]]
            if missing:
                error_msg = f"Tool '{normalized}' missing required params: {missing}"
                if self.debug:
                    print(f"[DEBUG] {error_msg}")
                step["error"] = error_msg
            
            validated.append(step)
        
        return validated
    
    def _map_positional_params(
        self,
        tool_params: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map positional arguments to parameter names during validation.
        
        This allows SLM to use write_file("path", "content") format.
        """
        # Collect all positional args
        positional_args = []
        i = 0
        while f"_positional_{i}" in params:
            positional_args.append(params.pop(f"_positional_{i}"))
            i += 1
        
        if not positional_args:
            return params
        
        # Build ordered list of params: required first, then optional
        required_params = []
        optional_params = []
        
        for param_name, param_info in tool_params.items():
            if isinstance(param_info, dict) and param_info.get("required"):
                required_params.append(param_name)
            else:
                optional_params.append(param_name)
        
        # Combine: required first, then optional
        ordered_params = required_params + optional_params
        
        # Map positional args to params in order
        for i, positional_value in enumerate(positional_args):
            if i < len(ordered_params):
                param_name = ordered_params[i]
                params[param_name] = positional_value
        
        return params
    
    def _find_similar_tools(self, name: str, available: List[str], max_suggestions: int = 3) -> List[str]:
        """Find similar tool names for helpful suggestions."""
        name_lower = name.lower()
        name_words = set(name_lower.replace('_', ' ').replace('-', ' ').split())
        
        scored = []
        for tool_name in available:
            tool_lower = tool_name.lower()
            tool_words = set(tool_lower.replace('_', ' ').replace('-', ' ').split())
            
            # Score by word overlap
            overlap = len(name_words & tool_words)
            if overlap > 0:
                scored.append((tool_name, overlap))
        
        # Sort by overlap descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in scored[:max_suggestions]]
