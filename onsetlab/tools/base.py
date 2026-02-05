"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Base class for tools that can be used by the agent."""
    
    name: str
    description: str
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool with given parameters.
        
        Returns:
            Result as a string.
        """
        pass
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema format for the planner."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
