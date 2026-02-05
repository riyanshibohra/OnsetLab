"""REWOO Solver - synthesizes final answer from results."""

from typing import List, Dict, Any, Optional

from ..model.base import BaseModel


SOLVER_PROMPT = '''You are a helpful assistant. Based on the task and execution results, provide a clear, concise answer.

Task: {task}

Execution results:
{results_summary}

Instructions:
1. Use the execution results to answer the task
2. Be concise and direct
3. Include relevant numbers/data from results
4. Do NOT make up information not in the results

Answer:'''


SOLVER_WITH_CONTEXT_PROMPT = '''You are a helpful assistant. Based on the conversation and execution results, provide a clear, concise answer.

Conversation context:
{context}

Current task: {task}

Execution results:
{results_summary}

Instructions:
1. Use the execution results to answer the task
2. Consider the conversation context
3. Be concise and direct
4. Include relevant numbers/data from results

Answer:'''


class Solver:
    """
    REWOO Solver - synthesizes final answer from execution results.
    
    Takes the original task and all tool results, generates a
    natural language answer for the user.
    """
    
    def __init__(self, model: BaseModel):
        """
        Initialize solver.
        
        Args:
            model: The SLM backend.
        """
        self.model = model
    
    def solve(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: Dict[str, str],
        context: Optional[str] = None
    ) -> str:
        """
        Synthesize final answer from results.
        
        Args:
            task: Original task.
            plan: Execution plan (for reference).
            results: Results from executor.
            context: Optional conversation context.
            
        Returns:
            Natural language answer.
        """
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
            temperature=0.5,
            max_tokens=512,
        )
        
        return response.strip()
    
    def _format_results(
        self,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> str:
        """Format results with tool context for the prompt."""
        lines = []
        for step in plan:
            step_id = step["id"]
            tool_name = step["tool"]
            result = results.get(step_id, "No result")
            lines.append(f"{tool_name} result: {result}")
        return "\n".join(lines)
