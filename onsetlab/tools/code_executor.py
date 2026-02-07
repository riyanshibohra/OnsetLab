"""
CodeExecutor Tool — sandboxed code execution for OnsetLab agents.

Lets the agent write and run code to solve problems:
- Math computations
- Data processing
- String manipulation
- Algorithm testing

Runs in a sandbox (local subprocess or Docker container).
"""

from typing import Any, Dict
from .base import BaseTool
from ..execution import CodeExecutor as _CodeExecutor


class CodeExecutorTool(BaseTool):
    """
    Execute code in a sandboxed environment.
    
    Supports Python, JavaScript, and Bash.
    Code runs in isolation with timeout and memory limits.
    """

    name = "CodeExecutor"
    description = (
        "Execute code in a sandboxed environment. "
        "Supports Python, JavaScript, and Bash. "
        "Use print() / console.log() to output results. "
        "No network or filesystem access."
    )

    def __init__(self, sandbox: str = "local", timeout: int = 30):
        self._executor = _CodeExecutor(sandbox=sandbox, default_timeout=timeout)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "description": "The code to execute. Use print() to output results.",
                "required": True,
            },
            "language": {
                "type": "string",
                "description": "Programming language: python, javascript, or bash",
                "default": "python",
            },
        }

    def execute(self, code: str = "", language: str = "python", **kwargs) -> str:
        """Execute code and return the output."""
        if not code:
            return "Error: No code provided."

        result = self._executor.execute(code, language=language)

        if result.success:
            return result.output.strip() or "(executed successfully, no output)"
        else:
            return f"Error: {result.error.strip() or f'Exit code {result.exit_code}'}"
