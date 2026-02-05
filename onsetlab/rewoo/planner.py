"""REWOO Planner - generates execution plan upfront."""

import re
from typing import List, Dict, Any, Optional

from ..model.base import BaseModel
from ..tools.base import BaseTool


# Few-shot examples showing EXACT format expected
PLANNER_PROMPT = '''You are a planner. Write tool calls in EXACT format shown below.

AVAILABLE TOOLS:
{tools_description}

EXAMPLE FORMAT (follow exactly):
Task: Do X, then use result for Y
#E1 = ToolA(param="value")
#E2 = ToolB(input=#E1)

Task: Calculate something complex
#E1 = ToolA(x="first")
#E2 = ToolA(x="second") 
#E3 = ToolB(expr="#E1 + #E2")

RULES:
- Output ONLY tool calls, nothing else
- Use #E1, #E2 to chain results (NO quotes around #E1)
- One tool per line
- ALL params in format: param="value" or param=#E1
{context_section}
Task: {task}
'''


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
            temperature=0.1,
            max_tokens=256,
            stop_sequences=["\n\n", "Note:", "Explanation:", "---"],
        )
        
        if self.debug:
            print(f"\n[DEBUG] Planner raw response:\n{response}\n")
        
        steps = self._parse_plan(response)
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
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that don't look like tool calls
            if not ('(' in line and ')' in line):
                continue
            
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
            
            # Skip if not a known tool (but allow parsing to continue for validation later)
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
        
        # Split by comma, but be careful with nested quotes
        # Handle: param="value", param2=#E1, param3="complex #E1 + 5"
        
        # First, extract quoted values
        quoted_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        for match in re.finditer(quoted_pattern, params_str):
            params[match.group(1)] = match.group(2)
        
        # Remove matched quoted params
        remaining = re.sub(quoted_pattern, '', params_str)
        
        # Then extract unquoted values (#E refs, numbers)
        unquoted_pattern = r'(\w+)\s*=\s*(#E\d+|[\d.\-]+)'
        for match in re.finditer(unquoted_pattern, remaining):
            key = match.group(1)
            if key in params:
                continue
            value = match.group(2)
            
            if value.startswith('#E'):
                params[key] = value
            else:
                try:
                    params[key] = float(value) if '.' in value else int(value)
                except ValueError:
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
    
    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate steps - check tool exists and has required params."""
        valid = []
        
        for step in steps:
            tool_name = step["tool"]
            
            if tool_name not in self.tools:
                if self.debug:
                    print(f"[DEBUG] Unknown tool: {tool_name}")
                continue
            
            tool = self.tools[tool_name]
            required = tool.parameters.get("required", [])
            
            missing = [p for p in required if p not in step["params"]]
            if missing:
                if self.debug:
                    print(f"[DEBUG] Missing params for {tool_name}: {missing}")
                continue
            
            valid.append(step)
        
        return valid
