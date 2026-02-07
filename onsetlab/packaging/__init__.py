"""
OnsetLab Packaging - Export agents in multiple formats.

Supported formats:
- config: YAML/JSON configuration file
- docker: Dockerfile + supporting files (engine: "ollama" or "vllm")
- binary: Standalone executable (requires PyInstaller)
"""

from .config_export import ConfigExporter
from .docker_export import DockerExporter
from .binary_export import BinaryExporter

__all__ = ["ConfigExporter", "DockerExporter", "BinaryExporter", "export_agent"]


def export_agent(agent, format: str, output: str, **kwargs) -> str:
    """
    Export an agent in the specified format.
    
    Args:
        agent: Agent instance to export
        format: Export format - "config", "docker", or "binary"
        output: Output path (file or directory)
        **kwargs: Format-specific options
            For docker: engine="vllm" for GPU-accelerated inference
        
    Returns:
        Path to exported artifact
        
    Examples:
        # Standard Ollama deployment
        export_agent(agent, "docker", "./deploy")
        
        # vLLM deployment (5-10x faster, requires GPU)
        export_agent(agent, "docker", "./deploy", engine="vllm")
    """
    if format == "config":
        return ConfigExporter.export(agent, output, **kwargs)
    elif format == "docker":
        return DockerExporter.export(agent, output, **kwargs)
    elif format == "binary":
        return BinaryExporter.export(agent, output, **kwargs)
    else:
        raise ValueError(f"Unknown export format: {format}. Use 'config', 'docker', or 'binary'")
