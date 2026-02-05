"""REWOO Executor - executes plan steps and resolves dependencies."""

import re
from typing import List, Dict, Any, Optional

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
            
            # Check for validation errors from planner
            if "error" in step:
                results[step_id] = f"Error: {step['error']}"
                continue
            
            # Execute tool
            if tool_name not in self.tools:
                results[step_id] = f"Error: Unknown tool '{tool_name}'"
                continue
            
            try:
                tool = self.tools[tool_name]
                
                # Map positional arguments to correct parameter names
                params = self._map_positional_params(tool, params)
                
                # Normalize parameter names (handle camelCase/snake_case, typos)
                params = self._normalize_param_names(tool, params)
                
                result = tool.execute(**params)
                results[step_id] = str(result)
            except TypeError as e:
                # Handle missing/extra parameters gracefully
                results[step_id] = f"Error: {str(e)}"
            except Exception as e:
                results[step_id] = f"Error: {str(e)}"
        
        return results
    
    def _map_positional_params(
        self,
        tool: BaseTool,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map positional arguments (_positional_0, _positional_1, etc.) to correct param names.
        
        When the model outputs ToolName("val1", "val2") instead of named params,
        we map positional values to parameters in order (required first, then optional).
        """
        # Collect all positional args
        positional_args = []
        i = 0
        while f"_positional_{i}" in params:
            positional_args.append(params.pop(f"_positional_{i}"))
            i += 1
        
        if not positional_args:
            return params
        
        # Get tool's parameter schema
        tool_params = tool.parameters
        
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
    
    def _normalize_param_names(
        self,
        tool: BaseTool,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize parameter names to match tool's expected parameters.
        
        Handles:
        - camelCase vs snake_case (e.g., pullNumber -> pull_number)
        - Word overlap matching (e.g., pullNumber -> issue_number if 'number' overlaps)
        - Direct matches
        """
        tool_params = set(tool.parameters.keys())
        normalized = {}
        
        for param_name, value in params.items():
            # Direct match - use as is
            if param_name in tool_params:
                normalized[param_name] = value
                continue
            
            # Try to find a matching tool parameter
            best_match = self._find_best_param_match(param_name, tool_params)
            
            if best_match:
                # Don't overwrite if we already have a value for this param
                if best_match not in normalized:
                    normalized[best_match] = value
            else:
                # Keep the original (will likely cause error, but that's expected)
                normalized[param_name] = value
        
        return normalized
    
    def _find_best_param_match(
        self,
        param_name: str,
        tool_params: set
    ) -> Optional[str]:
        """
        Find matching tool parameter - strict matching only.
        
        Only allows:
        1. Case-insensitive exact match
        2. snake_case <-> camelCase conversion
        
        No word overlap guessing - if param doesn't match, return None.
        """
        param_lower = param_name.lower()
        
        # 1. Case-insensitive exact match
        for tp in tool_params:
            if tp.lower() == param_lower:
                return tp
        
        # 2. Convert to snake_case and try matching
        snake_name = self._to_snake_case(param_name)
        for tp in tool_params:
            if tp == snake_name or self._to_snake_case(tp) == snake_name:
                return tp
        
        # No match - don't guess
        return None
    
    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case."""
        result = []
        for i, c in enumerate(name):
            if c.isupper() and i > 0:
                result.append('_')
            result.append(c.lower())
        return ''.join(result)
    
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
