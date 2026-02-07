"""
OnsetLab Execution — sandboxed code execution for AI agents.

Two sandbox backends:
- LocalSandbox: subprocess with resource limits (fast, moderate security)
- DockerSandbox: full container isolation (slower, high security)

Usage:
    from onsetlab.execution import CodeExecutor

    # Local sandbox (default)
    executor = CodeExecutor(sandbox="local")
    result = executor.execute("print(2 + 2)", language="python")
    # result.output == "4"

    # Docker sandbox (requires Docker)
    executor = CodeExecutor(sandbox="docker")
    result = executor.execute("console.log('hi')", language="javascript")
"""

from .sandbox import CodeExecutor, ExecutionResult, LocalSandbox, DockerSandbox

__all__ = ["CodeExecutor", "ExecutionResult", "LocalSandbox", "DockerSandbox"]
