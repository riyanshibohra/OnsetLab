"""REWOO Planner - generates execution plan upfront."""

import re
from typing import List, Dict, Any, Optional

from ..model.base import BaseModel
from ..tools.base import BaseTool


# Simple planner prompt
PLANNER_PROMPT = '''TOOLS:
{tools_description}

Write ONE tool call to solve the task. Use EXACT tool names above.
{context_section}
Task: {task}
Answer: #E1 ='''


class Planner:
    """REWOO Planner - creates execution plan upfront."""
    
    def __init__(self, model: BaseModel, tools: List[BaseTool], debug: bool = False):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.debug = debug
    
    def plan(self, task: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate execution plan for a task."""
        tools_desc = self._format_tools_description()
        
        context_section = ""
        if context:
            context_section = f"\nCONTEXT (previous results you can reference):\n{context}\n"
        
        prompt = PLANNER_PROMPT.format(
            tools_description=tools_desc,
            context_section=context_section,
            task=task
        )
        
        response = self.model.generate(
            prompt,
            temperature=0.0,
            max_tokens=100,
            stop_sequences=["\n\n", "\n#E2", "Note:", "Explanation:", "---", "Task:"],
        )
        
        # Prepend "#E1 =" since prompt ends with it
        full_response = "#E1 =" + response
        
        if self.debug:
            print(f"\n[DEBUG] Planner raw response:\n{full_response}\n")
        
        steps = self._parse_plan(full_response)
        steps = self._deduplicate_steps(steps)
        steps = self._validate_steps(steps)
        
        if self.debug:
            print(f"[DEBUG] Parsed steps: {steps}\n")
        
        return steps
    
    def _format_tools_description(self) -> str:
        """Format tools for the prompt with clear parameter details."""
        lines = []
        for name, tool in self.tools.items():
            params = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            
            param_parts = []
            for p, details in params.items():
                p_type = details.get("type", "string")
                req = " [required]" if p in required else ""
                
                # Show enum values if available
                if "enum" in details:
                    enum_values = "|".join(details["enum"])
                    param_parts.append(f'{p}="{{{enum_values}}}"{req}')
                else:
                    param_parts.append(f'{p}: {p_type}{req}')
            
            params_str = ", ".join(param_parts)
            lines.append(f"{name}({params_str})")
            lines.append(f"  - {tool.description}")
        
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
            
            # Handle: #E1 = #E1 = ToolName(...) by removing duplicate
            line = re.sub(r'^(#E\d+)\s*=\s*\1\s*=', r'\1 =', line)
            
            # Try pattern with #E prefix: #E1 = ToolName(params)
            match = re.match(r'#E(\d+)\s*=\s*(\w+)\s*\((.+)\)', line)
            
            if match:
                step_num = int(match.group(1))
                tool_name = match.group(2)
                params_str = match.group(3)
            else:
                # Try pattern without #E prefix: ToolName(params)
                match = re.match(r'(\w+)\s*\((.+)\)', line)
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
        """Parse parameter string into dict."""
        params = {}
        
        # First, extract quoted values: param="value"
        quoted_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        for match in re.finditer(quoted_pattern, params_str):
            params[match.group(1)] = match.group(2)
        
        # Remove matched quoted params
        remaining = re.sub(quoted_pattern, '', params_str)
        
        # Extract unquoted values: param=#E1 or param=123 or param=word
        unquoted_pattern = r'(\w+)\s*=\s*([^,\s\)]+)'
        for match in re.finditer(unquoted_pattern, remaining):
            key = match.group(1)
            if key in params:
                continue
            value = match.group(2).strip()
            
            # Handle #E references
            if value.startswith('#E'):
                params[key] = value
            # Handle numbers
            elif re.match(r'^[\d.\-]+$', value):
                try:
                    params[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    params[key] = value
            # Handle plain string values (like operation=difference)
            else:
                params[key] = value
        
        return params
    
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
        """Try to match tool name to actual tool, handling model mistakes."""
        # Exact match
        if name in self.tools:
            return name
        
        # Case-insensitive match
        name_lower = name.lower()
        for tool_name in self.tools:
            if tool_name.lower() == name_lower:
                return tool_name
        
        # Partial/substring match (e.g., "Calc" -> "Calculator", "ToolCalculator" -> "Calculator")
        # Works for any tool name without hardcoding
        for tool_name in self.tools:
            tool_lower = tool_name.lower()
            # Check if tool name is contained in the given name or vice versa
            if tool_lower in name_lower or name_lower in tool_lower:
                return tool_name
        
        return None
    
    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate steps - check tool exists and has required params."""
        valid = []
        
        for step in steps:
            tool_name = step["tool"]
            
            # Try to normalize the tool name
            normalized = self._normalize_tool_name(tool_name)
            if not normalized:
                if self.debug:
                    print(f"[DEBUG] Unknown tool: {tool_name}")
                continue
            
            # Update step with normalized name
            step["tool"] = normalized
            
            tool = self.tools[normalized]
            required = tool.parameters.get("required", [])
            
            missing = [p for p in required if p not in step["params"]]
            if missing:
                if self.debug:
                    print(f"[DEBUG] Missing params for {normalized}: {missing}")
                continue
            
            valid.append(step)
        
        return valid
