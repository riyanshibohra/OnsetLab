"""REWOO Solver - synthesizes final answer from results."""

from typing import List, Dict, Any, Optional

from ..model.base import BaseModel


SOLVER_PROMPT = '''Answer the user's question using ONLY the execution results below.

Question: {task}

Results:
{results_summary}

Rules:
- Be concise and direct (1-2 sentences max)
- Use the actual values from results
- Do NOT add information not in the results
- Do NOT explain how you got the answer

Answer:'''


SOLVER_WITH_CONTEXT_PROMPT = '''Answer the user's question using the conversation context and execution results.

Context:
{context}

Question: {task}

Results:
{results_summary}

Rules:
- Be concise and direct (1-2 sentences max)
- Use values from results and context
- Do NOT add information not provided
- Do NOT explain your reasoning

Answer:'''


class Solver:
    """
    REWOO Solver - synthesizes final answer from execution results.
    
    Takes the original task and all tool results, generates a
    concise natural language answer for the user.
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
            Concise natural language answer.
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
            temperature=0.3,  # Slightly creative but mostly factual
            max_tokens=150,   # Keep answers short
        )
        
        # Clean up response
        answer = response.strip()
        
        # Remove common prefixes the model might add
        prefixes_to_remove = [
            "Answer:", "The answer is:", "Based on the results,",
            "According to the results,", "Here's the answer:"
        ]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        return answer
    
    def _format_results(
        self,
        plan: List[Dict[str, Any]],
        results: Dict[str, str]
    ) -> str:
        """Format results clearly for the prompt."""
        if not plan:
            return "No tool execution needed."
        
        lines = []
        for step in plan:
            step_id = step["id"]
            tool_name = step["tool"]
            result = results.get(step_id, "No result")
            lines.append(f"- {tool_name}: {result}")
        
        return "\n".join(lines)
