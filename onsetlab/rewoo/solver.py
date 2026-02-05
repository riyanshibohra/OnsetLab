"""REWOO Solver - synthesizes final answer from results."""

import re
from typing import List, Dict, Any, Optional

from ..model.base import BaseModel


SOLVER_PROMPT = '''Answer the question using ONLY the results below.

Question: {task}
Results:
{results_summary}

RULES:
- Give ONLY the answer, nothing else
- 1-2 sentences maximum
- NO explanations, NO notes, NO parenthetical comments
- Use the exact values from results

Answer:'''


SOLVER_WITH_CONTEXT_PROMPT = '''Answer the question using context and results.

Context:
{context}

Question: {task}
Results:
{results_summary}

RULES:
- Give ONLY the answer, nothing else
- 1-2 sentences maximum
- NO explanations, NO notes, NO parenthetical comments

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
        
        if context:
            prompt = SOLVER_WITH_CONTEXT_PROMPT.format(
                context=context,
                task=task,
                results_summary=results_summary
            )
        else:
            prompt = SOLVER_PROMPT.format(
                task=task,
                results_summary=results_summary
            )
        
        response = self.model.generate(
            prompt,
            temperature=0.3,
            max_tokens=100,
            stop_sequences=["\n\n", "(Note", "(note", "Note:"],
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
        """Format results for prompt."""
        if not plan:
            return "No results available."
        
        lines = []
        for step in plan:
            step_id = step["id"]
            tool_name = step["tool"]
            result = results.get(step_id, "No result")
            lines.append(f"- {tool_name}: {result}")
        
        return "\n".join(lines)
