"""REWOO Verifier - validates execution results."""

from typing import List, Dict, Any, Tuple

from ..model.base import BaseModel


VERIFIER_PROMPT = '''You are a verification assistant. Check if the execution results make sense.

Task: {task}

Plan that was executed:
{plan_summary}

Results:
{results_summary}

Instructions:
1. Check if each result seems reasonable for its tool
2. Check if the math/logic makes sense
3. Check for any errors in results

Respond with ONLY one of:
- "VALID" if all results look correct
- "INVALID: <reason>" if something is wrong

Response:'''


class Verifier:
    """
    REWOO Verifier - checks if execution results are valid.
    
    Uses the SLM to verify results make sense before synthesizing answer.
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
        # Quick check for obvious errors
        for step_id, result in results.items():
            if isinstance(result, str) and result.startswith("Error:"):
                return False, f"{step_id} failed: {result}"
        
        # Use SLM for semantic verification
        plan_summary = self._format_plan(plan)
        results_summary = self._format_results(results)
        
        prompt = VERIFIER_PROMPT.format(
            task=task,
            plan_summary=plan_summary,
            results_summary=results_summary
        )
        
        response = self.model.generate(
            prompt,
            temperature=0.1,  # Very low temperature for consistent verification
            max_tokens=256,
        )
        
        response = response.strip()
        
        if response.startswith("VALID"):
            return True, "Verification passed"
        elif response.startswith("INVALID"):
            reason = response.replace("INVALID:", "").strip()
            return False, reason
        else:
            # If model doesn't follow format, assume valid (fail-open)
            return True, "Verification inconclusive, proceeding"
    
    def _format_plan(self, plan: List[Dict[str, Any]]) -> str:
        """Format plan for prompt."""
        lines = []
        for step in plan:
            params_str = ", ".join([
                f'{k}="{v}"' for k, v in step["params"].items()
            ])
            lines.append(f'{step["id"]} = {step["tool"]}({params_str})')
        return "\n".join(lines)
    
    def _format_results(self, results: Dict[str, str]) -> str:
        """Format results for prompt."""
        lines = []
        for step_id, result in results.items():
            lines.append(f"{step_id} = {result}")
        return "\n".join(lines)
    
    def quick_verify(self, results: Dict[str, str]) -> Tuple[bool, str]:
        """
        Quick verification without SLM (just check for errors).
        
        Args:
            results: Results from executor.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        for step_id, result in results.items():
            if isinstance(result, str):
                if result.startswith("Error:"):
                    return False, f"{step_id}: {result}"
                if result.lower() in ["none", "null", ""]:
                    return False, f"{step_id}: Empty result"
        return True, "Quick verification passed"
