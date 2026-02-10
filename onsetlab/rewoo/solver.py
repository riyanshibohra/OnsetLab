"""REWOO Solver - synthesizes final answer from results."""

import re
from typing import List, Dict, Any, Optional

from ..model.base import BaseModel


SOLVER_PROMPT = '''Question: {task}

Data:
{results_summary}

Write a focused answer using ONLY the data above. Stay on topic. Stop when done.

Answer:'''


CASUAL_PROMPT = '''Respond briefly to this casual message.

Context:
{context}

Message: {task}

RULES:
- Be friendly and brief (1 sentence)
- NO tool suggestions
- NO explanations

Response:'''


class Solver:
    """REWOO Solver - synthesizes final answer from execution results."""
    
    def __init__(self, model: BaseModel):
        self.model = model
    
    def solve(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: Dict[str, str],
        context: Optional[str] = None
    ) -> str:
        """Synthesize final answer from results."""
        
        # If no plan and no results, handle as casual/direct
        if not plan and not results:
            return self._handle_casual(task, context)
        
        results_summary = self._format_results(plan, results)
        
        prompt = SOLVER_PROMPT.format(
            task=task,
            results_summary=results_summary
        )
        
        response = self.model.generate(
            prompt,
            temperature=0.1,  # Lower = more focused
            max_tokens=400,   # Reasonable length
            stop_sequences=["\n\n\n", "Question:", "Data:"],
        )
        
        return self._clean_response(response)
    
    def _handle_casual(self, task: str, context: Optional[str] = None) -> str:
        """Handle casual messages without tool results."""
        prompt = CASUAL_PROMPT.format(
            context=context or "No previous context.",
            task=task
        )
        
        response = self.model.generate(
            prompt,
            temperature=0.7,
            max_tokens=50,
            stop_sequences=["\n\n", "(Note", "(note"],
        )
        
        return self._clean_response(response)
    
    def _clean_response(self, response: str) -> str:
        """Clean up model response."""
        answer = response.strip()
        
        # Remove common prefixes
        prefixes = [
            "Answer:", "The answer is:", "Based on the results,",
            "According to the results,", "Here's the answer:",
            "Response:", "The result is:"
        ]
        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Remove parenthetical notes/explanations at the end
        # Pattern: (Note: ...) or (This is ...) etc.
        answer = re.sub(r'\s*\([Nn]ote:.*?\)\s*$', '', answer)
        answer = re.sub(r'\s*\([Tt]his.*?\)\s*$', '', answer)
        answer = re.sub(r'\s*\([Ii] used.*?\)\s*$', '', answer)
        
        # Remove trailing incomplete sentences starting with (
        if '(' in answer and answer.count('(') > answer.count(')'):
            answer = answer[:answer.rfind('(')].strip()
        
        return answer.strip()
    
    def _format_results(
        self,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> str:
        """Format results for prompt - include full content for synthesis."""
        if not plan:
            return "No results available."
        
        lines = []
        error_count = 0
        
        for step in plan:
            step_id = step["id"]
            tool_name = step["tool"]
            result = results.get(step_id, "No result")
            result_str = str(result)
            
            # Check if this is an error
            result_lower = result_str.lower()
            is_error = any(err in result_lower for err in ['error:', 'cannot ', 'failed:', 'invalid', 'unable to'])
            
            if is_error:
                error_count += 1
                lines.append(f"ERROR: {result_str[:300]}")
            else:
                # Limit context for small models - they get confused with too much
                if len(result_str) > 800:
                    result_str = result_str[:800] + "..."
                lines.append(result_str)
        
        if error_count == len(plan) and error_count > 0:
            lines.insert(0, "ALL TOOLS FAILED - explain the errors to the user.\n")
        
        return "\n\n".join(lines)
