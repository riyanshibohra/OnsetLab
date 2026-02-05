"""REWOO Verifier - validates plans and execution results."""

import re
from typing import List, Dict, Any, Tuple, Set

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


PLAN_VERIFIER_PROMPT = '''Check if this plan correctly uses the values from the user's task.

User's task: {task}

Planned action: {plan_summary}

Key values in user's task: {extracted_values}

Question: Does the plan use the EXACT values from the user's task?
- If the user mentioned specific numbers/IDs, are they used correctly in the plan?
- If the user mentioned specific names, are they used correctly?

Respond with ONLY: CORRECT or INCORRECT'''


class Verifier:
    """
    REWOO Verifier - checks plans and execution results.
    
    Features:
    - Pre-execution plan verification (check values match user intent)
    - Post-execution result verification
    - Quick verification without SLM for simple cases
    """
    
    def __init__(self, model: BaseModel, debug: bool = False):
        """
        Initialize verifier.
        
        Args:
            model: The SLM backend.
            debug: Enable debug logging.
        """
        self.model = model
        self.debug = debug
    
    def verify_plan(
        self,
        task: str,
        plan: List[Dict[str, Any]]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Verify plan BEFORE execution - check if values match user intent.
        
        Args:
            task: Original task from user.
            plan: Execution plan to verify.
            
        Returns:
            Tuple of (is_valid, reason, corrected_plan).
            If invalid but correctable, returns corrected plan.
        """
        if not plan:
            return False, "Empty plan", plan
        
        # Extract key values from user's task (numbers, quoted strings, etc.)
        task_values = self._extract_values_from_text(task)
        
        if self.debug:
            print(f"[DEBUG] Extracted values from task: {task_values}")
        
        # Check each step's params against task values
        issues = []
        corrected_plan = []
        
        for step in plan:
            step_copy = step.copy()
            step_copy["params"] = step["params"].copy()
            
            # Skip steps with errors (already caught)
            if "error" in step:
                corrected_plan.append(step_copy)
                continue
            
            # Check numeric params - do they match numbers in the task?
            param_issues = self._check_param_values(
                task, task_values, step["params"]
            )
            
            if param_issues:
                issues.extend(param_issues)
                # Try to correct obvious mismatches
                corrected_params = self._try_correct_params(
                    task, task_values, step_copy["params"]
                )
                step_copy["params"] = corrected_params
            
            corrected_plan.append(step_copy)
        
        if issues:
            reason = "; ".join(issues)
            if self.debug:
                print(f"[DEBUG] Plan issues: {reason}")
            # Return corrected plan if we made corrections
            return False, reason, corrected_plan
        
        return True, "Plan values match task", plan
    
    def _extract_values_from_text(self, text: str) -> Dict[str, Set[Any]]:
        """
        Extract key values from text (numbers, quoted strings, etc.).
        
        Returns dict with:
        - 'numbers': set of integers found
        - 'strings': set of quoted strings found
        - 'identifiers': set of potential identifiers (owner/repo patterns)
        """
        values = {
            'numbers': set(),
            'strings': set(),
            'identifiers': set(),
        }
        
        # Extract integers (standalone numbers, not part of larger tokens)
        # Match patterns like "issue 1", "number 42", "#5", etc.
        number_patterns = [
            r'\b(\d+)\b',  # Standalone numbers
            r'#(\d+)',     # GitHub-style #123
            r'issue\s*(\d+)',  # "issue 1"
            r'number\s*(\d+)', # "number 1"
        ]
        for pattern in number_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                values['numbers'].add(int(match.group(1)))
        
        # Extract quoted strings
        for match in re.finditer(r'"([^"]+)"', text):
            values['strings'].add(match.group(1))
        for match in re.finditer(r"'([^']+)'", text):
            values['strings'].add(match.group(1))
        
        # Extract owner/repo patterns
        for match in re.finditer(r'(\w+)/(\w+)', text):
            values['identifiers'].add(match.group(1))  # owner
            values['identifiers'].add(match.group(2))  # repo
        
        return values
    
    def _check_param_values(
        self,
        task: str,
        task_values: Dict[str, Set[Any]],
        params: Dict[str, Any]
    ) -> List[str]:
        """Check if param values match values from task."""
        issues = []
        
        # Check numeric params
        for param_name, param_value in params.items():
            if isinstance(param_value, int):
                # If user mentioned specific numbers, check if this matches any
                if task_values['numbers'] and param_value not in task_values['numbers']:
                    # This number wasn't mentioned by user - potential mismatch
                    # Only flag if it's likely an ID/number field
                    id_like_params = ['number', 'id', 'issue', 'pull', 'pr']
                    if any(p in param_name.lower() for p in id_like_params):
                        expected = list(task_values['numbers'])
                        issues.append(
                            f"'{param_name}={param_value}' but user mentioned: {expected}"
                        )
        
        return issues
    
    def _try_correct_params(
        self,
        task: str,
        task_values: Dict[str, Set[Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Try to correct params to match user's values."""
        corrected = params.copy()
        
        # If there's exactly one number in user's task and one numeric ID param
        # that doesn't match, correct it
        if len(task_values['numbers']) == 1:
            correct_number = list(task_values['numbers'])[0]
            
            for param_name, param_value in params.items():
                if isinstance(param_value, int):
                    id_like_params = ['number', 'id', 'issue', 'pull', 'pr']
                    if any(p in param_name.lower() for p in id_like_params):
                        if param_value != correct_number:
                            corrected[param_name] = correct_number
                            if self.debug:
                                print(f"[DEBUG] Corrected {param_name}: {param_value} -> {correct_number}")
        
        return corrected
    
    def verify(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        Verify execution results (post-execution).
        
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
