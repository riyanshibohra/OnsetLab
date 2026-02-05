"""REWOO Executor - executes plan steps and resolves dependencies."""

import re
from typing import List, Dict, Any

from ..tools.base import BaseTool


class Executor:
    """
    REWOO Executor - executes plan steps in order.
    
    Handles dependency resolution by substituting #E1, #E2, etc.
    with actual results from previous steps.
    """
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = {t.name: t for t in tools}
    
    def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Execute all steps in the plan.
        
        Args:
            plan: List of plan steps from Planner.
            
        Returns:
            Dict mapping step IDs to results.
        """
        results = {}
        
        for step in plan:
            step_id = step["id"]
            tool_name = step["tool"]
            params = step["params"].copy()
            
            # Resolve dependencies
            params = self._resolve_dependencies(params, results)
            
            # Execute tool
            if tool_name not in self.tools:
                results[step_id] = f"Error: Unknown tool '{tool_name}'"
                continue
            
            try:
                tool = self.tools[tool_name]
                result = tool.execute(**params)
                results[step_id] = str(result)
            except TypeError as e:
                # Handle missing/extra parameters gracefully
                results[step_id] = f"Error: {str(e)}"
            except Exception as e:
                results[step_id] = f"Error: {str(e)}"
        
        return results
    
    def _resolve_dependencies(
        self,
        params: Dict[str, Any],
        results: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Replace #E1, #E2, etc. with actual results.
        """
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Check if it's a direct reference (e.g., date=#E1)
                if re.match(r'^#E\d+$', value):
                    resolved[key] = results.get(value, value)
                else:
                    # Replace references within strings/expressions
                    new_value = value
                    for ref in re.findall(r'#E\d+', value):
                        if ref in results:
                            # For expressions, substitute the value directly
                            new_value = new_value.replace(ref, str(results[ref]))
                    resolved[key] = new_value
            else:
                resolved[key] = value
        
        return resolved
