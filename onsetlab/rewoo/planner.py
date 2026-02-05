"""REWOO Planner - generates execution plan upfront."""

import re
from typing import List, Dict, Any, Optional

from ..model.base import BaseModel
from ..tools.base import BaseTool


PLANNER_PROMPT = '''You are a planning assistant. Given a task, create a step-by-step plan using the available tools.

Available tools:
{tools_description}

Instructions:
1. Break down the task into steps
2. Each step should use ONE tool
3. Use #E1, #E2, etc. to reference results from previous steps
4. Output ONLY the plan in the exact format shown below

Format:
Plan:
#E1 = ToolName(param1="value1", param2="value2")
#E2 = ToolName(param1="value1", param2=#E1)
...

Example:
Task: What is 15% of 84.50?
Plan:
#E1 = Calculator(expression="84.50 * 0.15")

Example:
Task: What day of the week was January 1, 2000?
Plan:
#E1 = DateTime(operation="day_of_week", date="2000-01-01")

Task: {task}
Plan:'''


class Planner:
    """
    REWOO Planner - creates execution plan upfront.
    
    The plan contains all tool calls needed to complete the task,
    with dependencies explicitly marked using #E1, #E2, etc.
    """
    
    def __init__(self, model: BaseModel, tools: List[BaseTool]):
        """
        Initialize planner.
        
        Args:
            model: The SLM backend.
            tools: Available tools.
        """
        self.model = model
        self.tools = {t.name: t for t in tools}
    
    def plan(self, task: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate execution plan for a task.
        
        Args:
            task: The user's task/question.
            context: Optional conversation context.
            
        Returns:
            List of plan steps, each with:
            - id: Step ID (e.g., "#E1")
            - tool: Tool name
            - params: Tool parameters
            - depends_on: List of step IDs this depends on
        """
        # Build tools description
        tools_desc = self._format_tools_description()
        
        # Build prompt
        full_task = task
        if context:
            full_task = f"Context:\n{context}\n\nTask: {task}"
        
        prompt = PLANNER_PROMPT.format(
            tools_description=tools_desc,
            task=full_task
        )
        
        # Generate plan
        response = self.model.generate(
            prompt,
            temperature=0.3,  # Lower temperature for more deterministic plans
            max_tokens=1024,
        )
        
        # Parse plan
        return self._parse_plan(response)
    
    def _format_tools_description(self) -> str:
        """Format tools for the prompt."""
        lines = []
        for name, tool in self.tools.items():
            params = tool.parameters.get("properties", {})
            param_desc = ", ".join([
                f'{p}="{params[p].get("description", "")}"'
                for p in params
            ])
            lines.append(f"- {name}({param_desc}): {tool.description}")
        return "\n".join(lines)
    
    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse plan from model response.
        
        Returns list of plan steps.
        """
        steps = []
        
        # Pattern: #E1 = ToolName(params)
        pattern = r'(#E\d+)\s*=\s*(\w+)\((.*?)\)'
        
        for match in re.finditer(pattern, response, re.DOTALL):
            step_id = match.group(1)
            tool_name = match.group(2)
            params_str = match.group(3)
            
            # Parse parameters
            params = self._parse_params(params_str)
            
            # Find dependencies (references to other steps like #E1)
            depends_on = re.findall(r'#E\d+', params_str)
            
            steps.append({
                "id": step_id,
                "tool": tool_name,
                "params": params,
                "depends_on": depends_on,
            })
        
        return steps
    
    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dict."""
        params = {}
        
        # Pattern: key="value" or key=value or key=#E1
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|(#E\d+)|([^,\)]+))'
        
        for match in re.finditer(pattern, params_str):
            key = match.group(1)
            # Try each capture group
            value = match.group(2) or match.group(3) or match.group(4)
            if value:
                value = value.strip()
                # Try to convert to number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    pass
                params[key] = value
        
        return params
