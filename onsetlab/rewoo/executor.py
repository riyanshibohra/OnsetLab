"""REWOO Executor - executes plan steps and resolves dependencies.

Supports parallel execution of independent steps for faster completion.
"""

import inspect
import re
import logging
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..tools.base import BaseTool

logger = logging.getLogger(__name__)


class Executor:
    """
    REWOO Executor - executes plan steps.
    
    Handles dependency resolution by substituting #E1, #E2, etc.
    with actual results from previous steps.
    
    Supports parallel execution: independent steps (no shared
    dependencies) run concurrently via ThreadPoolExecutor.
    """
    
    def __init__(self, tools: List[BaseTool], parallel: bool = False, max_workers: int = 4):
        self.tools = {t.name: t for t in tools}
        self.parallel = parallel
        self.max_workers = max_workers
    
    def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Execute all steps in the plan.
        
        If parallel=True, independent steps run concurrently.
        Otherwise, steps run sequentially (original behavior).
        
        Args:
            plan: List of plan steps from Planner.
            
        Returns:
            Dict mapping step IDs to results.
        """
        if self.parallel and len(plan) > 1:
            return self._execute_parallel(plan)
        return self._execute_sequential(plan)

    def _execute_sequential(self, plan: List[Dict[str, Any]]) -> Dict[str, str]:
        """Execute steps one at a time (original behavior)."""
        results = {}
        
        for step in plan:
            result = self._execute_step(step, results)
            results[step["id"]] = result
        
        return results

    def _execute_parallel(self, plan: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Execute independent steps concurrently, dependent steps sequentially.
        
        Algorithm:
        1. Build dependency graph from #E references in params
        2. Group steps into "waves" via topological sort
        3. Execute each wave in parallel
        """
        results = {}
        waves = self._build_execution_waves(plan)

        logger.info(
            f"Parallel executor: {len(plan)} steps → "
            f"{len(waves)} waves: {[len(w) for w in waves]}"
        )

        for wave_idx, wave in enumerate(waves):
            if len(wave) == 1:
                # Single step — run directly (no thread overhead)
                step = wave[0]
                results[step["id"]] = self._execute_step(step, results)
            else:
                # Multiple independent steps — run concurrently
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {}
                    for step in wave:
                        # Snapshot results so threads don't race on dict reads
                        results_snapshot = dict(results)
                        future = pool.submit(
                            self._execute_step, step, results_snapshot
                        )
                        futures[future] = step["id"]

                    for future in as_completed(futures):
                        step_id = futures[future]
                        try:
                            results[step_id] = future.result()
                        except Exception as e:
                            results[step_id] = f"Error: {e}"

        return results

    def _build_execution_waves(
        self, plan: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group steps into execution waves using topological sort.
        
        Steps in the same wave have no mutual dependencies and can
        run in parallel.  Steps in later waves depend on earlier waves.
        """
        # Build id → step mapping
        step_map = {s["id"]: s for s in plan}

        # Extract dependencies for each step
        deps: Dict[str, Set[str]] = {}
        for step in plan:
            step_deps: Set[str] = set()
            for val in step.get("params", {}).values():
                if isinstance(val, str):
                    for ref in re.findall(r'#E\d+', val):
                        if ref in step_map:
                            step_deps.add(ref)
            # Also check depends_on field if present
            for d in step.get("depends_on", []):
                if d in step_map:
                    step_deps.add(d)
            deps[step["id"]] = step_deps

        # Topological sort into waves (Kahn's algorithm)
        remaining = set(step_map.keys())
        waves: List[List[Dict[str, Any]]] = []

        while remaining:
            # Find all steps whose dependencies are fully resolved
            ready = [
                sid for sid in remaining
                if deps[sid].issubset(set(step_map.keys()) - remaining)
            ]
            if not ready:
                # Cycle detected — just run the rest sequentially
                waves.append([step_map[sid] for sid in remaining])
                break
            waves.append([step_map[sid] for sid in ready])
            remaining -= set(ready)

        return waves

    def _execute_step(
        self, step: Dict[str, Any], results: Dict[str, str]
    ) -> str:
        """Execute a single plan step and return the result string."""
        tool_name = step["tool"]
        params = step["params"].copy()

        # Resolve dependencies
        params = self._resolve_dependencies(params, results)

        # Check for validation errors from planner
        if "error" in step:
            return f"Error: {step['error']}"

        # Execute tool
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            tool = self.tools[tool_name]

            # Map positional arguments to correct parameter names
            params = self._map_positional_params(tool, params)

            # Normalize parameter names (handle camelCase/snake_case, typos)
            params = self._normalize_param_names(tool, params)

            # Filter out params the tool doesn't accept (planner hallucinations)
            params = self._filter_accepted_params(tool, params)

            result = tool.execute(**params)
            return str(result)
        except TypeError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def _filter_accepted_params(tool: BaseTool, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drop any params that the tool's execute() method does not accept.

        If the method already accepts **kwargs, all params pass through.
        Otherwise only recognised parameter names are kept and unknown ones
        are silently dropped (with a debug log).  This prevents crashes when
        the planner hallucinates parameter names like 'range' or 'query'.
        """
        sig = inspect.signature(tool.execute)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_var_keyword:
            return params  # tool accepts anything

        accepted = {
            name for name, p in sig.parameters.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        filtered = {}
        for k, v in params.items():
            if k in accepted:
                filtered[k] = v
            else:
                logger.debug(f"Dropping unknown param '{k}' for {tool.name}")
        return filtered

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
