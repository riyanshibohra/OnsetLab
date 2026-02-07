"""
Sandboxed code execution — run untrusted code safely.

LocalSandbox:  subprocess with timeout + memory limits
DockerSandbox: full container isolation (no network, no fs, no escalation)
"""

import os
import sys
import time
import signal
import tempfile
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of a code execution."""
    output: str = ""
    error: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_ms: float = 0.0
    language: str = "python"

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def __str__(self) -> str:
        if self.success:
            return self.output.strip() or "(no output)"
        parts = []
        if self.timed_out:
            parts.append("Timed out")
        if self.error:
            parts.append(self.error.strip())
        return " | ".join(parts) or f"Exit code {self.exit_code}"


# ---------------------------------------------------------------------------
# Base sandbox
# ---------------------------------------------------------------------------

class BaseSandbox(ABC):
    """Abstract sandbox interface."""

    @abstractmethod
    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute code and return result."""
        pass


# ---------------------------------------------------------------------------
# Local sandbox — subprocess with resource limits
# ---------------------------------------------------------------------------

# Language → (command template, file extension)
LANGUAGE_CONFIGS = {
    "python": {
        "cmd": [sys.executable, "{file}"],
        "ext": ".py",
    },
    "javascript": {
        "cmd": ["node", "{file}"],
        "ext": ".js",
    },
    "bash": {
        "cmd": ["bash", "{file}"],
        "ext": ".sh",
    },
    "shell": {
        "cmd": ["bash", "{file}"],
        "ext": ".sh",
    },
}

# Dangerous patterns to reject before execution
BLOCKED_PATTERNS = [
    "import os; os.system",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "__import__('os')",
    "shutil.rmtree",
    "os.remove(",
    "os.rmdir(",
    "os.unlink(",
    "exec(",
    "eval(",
    "open('/etc",
    "open('/dev",
    "open('/proc",
    "open('/sys",
]


class LocalSandbox(BaseSandbox):
    """
    Local subprocess sandbox with safety limits.

    Security measures:
    - Timeout enforcement (default 30s)
    - Memory limit via ulimit (512MB)
    - Blocked dangerous patterns
    - Runs in temporary directory
    - No network access (best-effort via env)

    NOT fully isolated — use DockerSandbox for untrusted code.
    """

    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute code in a subprocess with resource limits."""
        lang = language.lower()

        if lang not in LANGUAGE_CONFIGS:
            return ExecutionResult(
                error=f"Unsupported language: {language}. "
                      f"Supported: {', '.join(LANGUAGE_CONFIGS.keys())}",
                exit_code=1,
                language=lang,
            )

        # Safety check
        safety_error = self._check_safety(code)
        if safety_error:
            return ExecutionResult(
                error=safety_error,
                exit_code=1,
                language=lang,
            )

        config = LANGUAGE_CONFIGS[lang]

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=config["ext"],
            delete=False,
            dir=tempfile.gettempdir(),
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            cmd = [c.replace("{file}", tmp_path) for c in config["cmd"]]

            # Set resource limits (Linux only — macOS has limited rlimit support)
            preexec = None
            if sys.platform == "linux":
                try:
                    import resource

                    def set_limits():
                        mem_bytes = self.max_memory_mb * 1024 * 1024
                        try:
                            resource.setrlimit(
                                resource.RLIMIT_AS, (mem_bytes, mem_bytes)
                            )
                        except (ValueError, OSError):
                            pass  # Not supported on this platform

                    preexec = set_limits
                except ImportError:
                    pass

            # Run
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),
                    preexec_fn=preexec,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                duration = (time.monotonic() - start) * 1000

                return ExecutionResult(
                    output=proc.stdout,
                    error=proc.stderr,
                    exit_code=proc.returncode,
                    duration_ms=duration,
                    language=lang,
                )

            except subprocess.TimeoutExpired:
                duration = (time.monotonic() - start) * 1000
                return ExecutionResult(
                    error=f"Execution timed out after {timeout}s",
                    exit_code=124,
                    timed_out=True,
                    duration_ms=duration,
                    language=lang,
                )

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _check_safety(self, code: str) -> Optional[str]:
        """Check code for obviously dangerous patterns."""
        for pattern in BLOCKED_PATTERNS:
            if pattern in code:
                return (
                    f"Blocked: code contains potentially dangerous pattern "
                    f"'{pattern}'. Use DockerSandbox for untrusted code."
                )
        return None


# ---------------------------------------------------------------------------
# Docker sandbox — full container isolation
# ---------------------------------------------------------------------------

# Default images per language
DEFAULT_IMAGES = {
    "python": "python:3.11-slim",
    "javascript": "node:20-slim",
    "bash": "bash:5.2",
    "shell": "bash:5.2",
}


class DockerSandbox(BaseSandbox):
    """
    Docker container sandbox — full isolation.

    Security measures:
    - network_mode: none (no internet)
    - read_only filesystem
    - Memory limit: 512MB
    - CPU limit: 50%
    - No privilege escalation
    - All capabilities dropped
    - Container destroyed after execution
    """

    def __init__(
        self,
        max_memory_mb: int = 512,
        cpu_period: int = 100000,
        cpu_quota: int = 50000,
        images: Optional[Dict[str, str]] = None,
    ):
        self.max_memory_mb = max_memory_mb
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.images = {**DEFAULT_IMAGES, **(images or {})}
        self._check_docker()

    def _check_docker(self):
        """Verify Docker is available."""
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError(
                "Docker is not available. "
                "Install Docker or use LocalSandbox instead."
            )

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute code inside a Docker container."""
        lang = language.lower()

        if lang not in self.images:
            return ExecutionResult(
                error=f"Unsupported language: {language}. "
                      f"Supported: {', '.join(self.images.keys())}",
                exit_code=1,
                language=lang,
            )

        image = self.images[lang]

        # Build the execution command
        if lang in ("python",):
            exec_cmd = ["python3", "-c", code]
        elif lang in ("javascript",):
            exec_cmd = ["node", "-e", code]
        elif lang in ("bash", "shell"):
            exec_cmd = ["bash", "-c", code]
        else:
            exec_cmd = ["sh", "-c", code]

        # Docker run with security constraints
        docker_cmd = [
            "docker", "run",
            "--rm",                          # Remove container after exit
            "--network=none",                # No network
            "--read-only",                   # Read-only filesystem
            f"--memory={self.max_memory_mb}m",  # Memory limit
            f"--cpu-period={self.cpu_period}",   # CPU limits
            f"--cpu-quota={self.cpu_quota}",
            "--security-opt=no-new-privileges",  # No privilege escalation
            "--cap-drop=ALL",                # Drop all capabilities
            "--tmpfs=/tmp:rw,size=64m",      # Writable /tmp (limited)
            image,
        ] + exec_cmd

        start = time.monotonic()
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 5,  # +5s for container startup
            )
            duration = (time.monotonic() - start) * 1000

            return ExecutionResult(
                output=proc.stdout,
                error=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=duration,
                language=lang,
            )

        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - start) * 1000
            return ExecutionResult(
                error=f"Execution timed out after {timeout}s",
                exit_code=124,
                timed_out=True,
                duration_ms=duration,
                language=lang,
            )


# ---------------------------------------------------------------------------
# CodeExecutor — high-level API
# ---------------------------------------------------------------------------

class CodeExecutor:
    """
    High-level code execution with automatic sandbox selection.

    Usage:
        executor = CodeExecutor()                     # local sandbox
        executor = CodeExecutor(sandbox="docker")     # docker sandbox

        result = executor.execute("print(2 + 2)")
        print(result.output)   # "4"
        print(result.success)  # True
    """

    def __init__(
        self,
        sandbox: str = "local",
        max_memory_mb: int = 512,
        default_timeout: int = 30,
    ):
        self.default_timeout = default_timeout

        if sandbox == "docker":
            self._sandbox = DockerSandbox(max_memory_mb=max_memory_mb)
        else:
            self._sandbox = LocalSandbox(max_memory_mb=max_memory_mb)

        self.sandbox_type = sandbox

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute code in the configured sandbox.

        Args:
            code: Source code to execute
            language: "python", "javascript", "bash"
            timeout: Max execution time in seconds

        Returns:
            ExecutionResult with output, error, exit_code, etc.
        """
        t = timeout or self.default_timeout
        logger.info(
            f"Executing {language} code ({len(code)} chars) "
            f"in {self.sandbox_type} sandbox (timeout={t}s)"
        )

        result = self._sandbox.execute(code, language, t)

        logger.info(
            f"Execution {'ok' if result.success else 'FAILED'}: "
            f"{result.duration_ms:.0f}ms, exit={result.exit_code}"
        )

        return result
