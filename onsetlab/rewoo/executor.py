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
        """
        Initialize executor.
        
        Args:
            tools: Available tools.
        """
        self.tools = {t.name: t for t in tools}
    
    def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Execute all steps in the plan.
        
        Args:
            plan: List of plan steps from Planner.
            
        Returns:
            Dict mapping step IDs to results (e.g., {"#E1": "12.675", "#E2": "..."}).
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
                results[step_id] = result
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
        
        Args:
            params: Parameters that may contain references.
            results: Results from previous steps.
            
        Returns:
            Parameters with references resolved.
        """
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Check if it's a direct reference (e.g., param=#E1)
                if re.match(r'^#E\d+$', value):
                    resolved[key] = results.get(value, value)
                else:
                    # Replace references within strings
                    for ref, result in results.items():
                        value = value.replace(ref, str(result))
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    def execute_parallel_safe(self, plan: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Execute steps, running independent steps in parallel.
        
        Note: This is a future optimization. Currently executes sequentially.
        Independent steps (no shared dependencies) could run in parallel.
        
        Args:
            plan: List of plan steps from Planner.
            
        Returns:
            Dict mapping step IDs to results.
        """
        # TODO: Implement parallel execution for independent steps
        # For now, just execute sequentially
        return self.execute(plan)
