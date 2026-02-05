"""Conversation memory for multi-turn interactions."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class ConversationMemory:
    """
    Simple conversation memory that stores message history.
    
    Supports persistence to JSON file.
    """
    
    def __init__(self, max_turns: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum conversation turns to keep (user + assistant = 1 turn).
        """
        self.max_turns = max_turns
        self._messages: List[Dict[str, str]] = []
        self._metadata: Dict[str, Any] = {}
    
    def add_user_message(self, content: str):
        """Add a user message."""
        self._messages.append({
            "role": "user",
            "content": content
        })
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self._messages.append({
            "role": "assistant",
            "content": content
        })
        self._trim_history()
    
    def add_tool_result(self, tool_name: str, result: str):
        """Store tool result in metadata for reference."""
        if "tool_results" not in self._metadata:
            self._metadata["tool_results"] = []
        self._metadata["tool_results"].append({
            "tool": tool_name,
            "result": result
        })
        # Keep only last 20 tool results
        self._metadata["tool_results"] = self._metadata["tool_results"][-20:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self._messages.copy()
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get last N messages."""
        return self._messages[-n:] if n > 0 else []
    
    def get_context_string(self) -> str:
        """Get conversation as a formatted string for the model."""
        lines = []
        for msg in self._messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
    
    def clear(self):
        """Clear all memory."""
        self._messages = []
        self._metadata = {}
    
    def save(self, path: str):
        """Save memory to JSON file."""
        data = {
            "messages": self._messages,
            "metadata": self._metadata,
            "max_turns": self.max_turns,
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    def load(self, path: str):
        """Load memory from JSON file."""
        data = json.loads(Path(path).read_text())
        self._messages = data.get("messages", [])
        self._metadata = data.get("metadata", {})
        self.max_turns = data.get("max_turns", 20)
    
    def _trim_history(self):
        """Trim history to max_turns."""
        # Each turn = 2 messages (user + assistant)
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __bool__(self) -> bool:
        return len(self._messages) > 0
