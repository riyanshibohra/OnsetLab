"""REWOO Verifier - validates execution results."""

from typing import List, Dict, Any, Tuple

from ..model.base import BaseModel


VERIFIER_PROMPT = '''Check if these execution results are valid for the given task.

Task: {task}

Execution:
{execution_summary}

Rules:
- If any result shows "Error:", respond INVALID
- If results seem reasonable for the task, respond VALID
- If results don't make sense, respond INVALID

Respond with ONLY one word: VALID or INVALID'''


class Verifier:
    """
    REWOO Verifier - checks if execution results are valid.
    
    Uses quick verification for simple cases, SLM for complex cases.
    """
    
    def __init__(self, model: BaseModel):
        """
        Initialize verifier.
        
        Args:
            model: The SLM backend.
        """
        self.model = model
    
    def verify(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        Verify execution results.
        
        Args:
            task: Original task.
            plan: Execution plan.
            results: Results from executor.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        # First, do quick verification (no SLM call)
        is_valid, reason = self.quick_verify(results)
        if not is_valid:
            return False, reason
        
        # For simple single-step plans, skip SLM verification
        if len(plan) == 1:
            return True, "Quick verification passed (single step)"
        
        # For multi-step plans, use SLM verification
        return self._slm_verify(task, plan, results)
    
    def _slm_verify(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> Tuple[bool, str]:
        """Use SLM to verify complex results."""
        execution_summary = self._format_execution(plan, results)
        
        prompt = VERIFIER_PROMPT.format(
            task=task,
            execution_summary=execution_summary
        )
        
        response = self.model.generate(
            prompt,
            temperature=0.0,  # Deterministic
            max_tokens=10,    # Just need VALID or INVALID
        )
        
        response = response.strip().upper()
        
        if "VALID" in response and "INVALID" not in response:
            return True, "SLM verification passed"
        elif "INVALID" in response:
            return False, "SLM verification failed"
        else:
            # If unclear, assume valid (fail-open for better UX)
            return True, "Verification inconclusive, proceeding"
    
    def _format_execution(
        self,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> str:
        """Format execution for prompt."""
        lines = []
        for step in plan:
            step_id = step["id"]
            tool_name = step["tool"]
            params = step["params"]
            result = results.get(step_id, "No result")
            
            params_str = ", ".join([f'{k}="{v}"' for k, v in params.items()])
            lines.append(f"{tool_name}({params_str}) → {result}")
        
        return "\n".join(lines)
    
    def quick_verify(self, results: Dict[str, str]) -> Tuple[bool, str]:
        """
        Quick verification without SLM (just check for obvious errors).
        
        Args:
            results: Results from executor.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        for step_id, result in results.items():
            if isinstance(result, str):
                # Check for error messages
                if result.startswith("Error:"):
                    return False, f"{step_id}: {result}"
                # Check for empty/null results
                if result.strip().lower() in ["none", "null", ""]:
                    return False, f"{step_id}: Empty result"
        
        return True, "Quick verification passed"
