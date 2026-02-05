"""
Config Export - Export agent configuration as YAML/JSON.

This creates a portable configuration that can be:
- Shared with others
- Version controlled
- Used to recreate the same agent
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigExporter:
    """Export agent configuration to YAML/JSON."""
    
    @classmethod
    def export(
        cls,
        agent,
        output: str,
        format: str = "yaml",
        include_mcp_auth: bool = False,
    ) -> str:
        """
        Export agent configuration.
        
        Args:
            agent: Agent instance
            output: Output file path (.yaml or .json)
            format: "yaml" or "json" (auto-detected from extension if not specified)
            include_mcp_auth: Include MCP authentication details (security risk!)
            
        Returns:
            Path to exported config
        """
        config = cls._extract_config(agent, include_mcp_auth)
        
        output_path = Path(output)
        
        # Auto-detect format from extension
        if output_path.suffix in [".yaml", ".yml"]:
            format = "yaml"
        elif output_path.suffix == ".json":
            format = "json"
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write config
        with open(output_path, "w") as f:
            if format == "yaml":
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(config, f, indent=2)
        
        return str(output_path)
    
    @classmethod
    def _extract_config(cls, agent, include_mcp_auth: bool) -> Dict[str, Any]:
        """Extract configuration from agent."""
        config = {
            "version": "1.0",
            "onsetlab": {
                "model": agent.model_name,
                "settings": {
                    "memory": agent._memory is not None,
                    "verify": agent._verify,
                    "routing": agent._routing_enabled,
                    "react_fallback": agent._react_fallback_enabled,
                    "max_replans": agent._max_replans,
                },
            },
            "tools": [],
            "mcp_servers": [],
        }
        
        # Export built-in tools
        for name, tool in agent.tools.items():
            config["tools"].append({
                "name": tool.name,
                "class": tool.__class__.__name__,
            })
        
        # Export MCP servers
        for server in agent._mcp_servers:
            server_config = {
                "name": server.name,
                "command": server._command,
                "args": server._args,
            }
            
            if include_mcp_auth and hasattr(server, '_env'):
                server_config["env"] = server._env
            
            config["mcp_servers"].append(server_config)
        
        return config
    
    @classmethod
    def load(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dict
        """
        path = Path(config_path)
        
        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            else:
                return json.load(f)
