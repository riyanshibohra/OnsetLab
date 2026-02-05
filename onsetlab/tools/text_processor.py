"""TextProcessor tool for text manipulation operations."""

import re
from typing import Any, Dict
from .base import BaseTool


class TextProcessor(BaseTool):
    """
    Process and analyze text: count words/chars, find/replace, extract patterns, transform case.
    """
    
    name = "TextProcessor"
    description = "Process text: count words/characters/lines, find/replace, extract patterns, change case, reverse, trim."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": [
                        "word_count", "char_count", "line_count",
                        "find", "replace", "extract",
                        "uppercase", "lowercase", "titlecase",
                        "reverse", "trim", "split"
                    ]
                },
                "text": {
                    "type": "string",
                    "description": "The text to process"
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to find/replace/extract (for find, replace, extract, split)"
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement text (for replace operation)"
                }
            },
            "required": ["operation", "text"]
        }
    
    def execute(
        self,
        operation: str,
        text: str,
        pattern: str = None,
        replacement: str = None
    ) -> str:
        """Execute text processing operation."""
        operation = operation.lower().strip()
        
        try:
            if operation == "word_count":
                return self._word_count(text)
            
            elif operation == "char_count":
                return self._char_count(text)
            
            elif operation == "line_count":
                return self._line_count(text)
            
            elif operation == "find":
                if not pattern:
                    return "Error: 'pattern' required for find operation"
                return self._find(text, pattern)
            
            elif operation == "replace":
                if not pattern:
                    return "Error: 'pattern' required for replace operation"
                return self._replace(text, pattern, replacement or "")
            
            elif operation == "extract":
                if not pattern:
                    return "Error: 'pattern' required for extract operation"
                return self._extract(text, pattern)
            
            elif operation == "uppercase":
                return text.upper()
            
            elif operation == "lowercase":
                return text.lower()
            
            elif operation == "titlecase":
                return text.title()
            
            elif operation == "reverse":
                return text[::-1]
            
            elif operation == "trim":
                return text.strip()
            
            elif operation == "split":
                if not pattern:
                    return "Error: 'pattern' required for split operation"
                return self._split(text, pattern)
            
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _word_count(self, text: str) -> str:
        """Count words in text."""
        words = text.split()
        count = len(words)
        return f"{count} word{'s' if count != 1 else ''}"
    
    def _char_count(self, text: str) -> str:
        """Count characters in text."""
        total = len(text)
        no_spaces = len(text.replace(" ", ""))
        return f"{total} characters ({no_spaces} without spaces)"
    
    def _line_count(self, text: str) -> str:
        """Count lines in text."""
        lines = text.split('\n')
        count = len(lines)
        non_empty = len([l for l in lines if l.strip()])
        return f"{count} line{'s' if count != 1 else ''} ({non_empty} non-empty)"
    
    def _find(self, text: str, pattern: str) -> str:
        """Find pattern in text."""
        # Try as regex first, fallback to literal
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
        except re.error:
            # Treat as literal string
            matches = []
            start = 0
            pattern_lower = pattern.lower()
            text_lower = text.lower()
            while True:
                pos = text_lower.find(pattern_lower, start)
                if pos == -1:
                    break
                matches.append(text[pos:pos + len(pattern)])
                start = pos + 1
        
        if not matches:
            return f"No matches found for '{pattern}'"
        
        count = len(matches)
        unique = list(set(matches))[:5]  # Show up to 5 unique matches
        
        if count == 1:
            return f"Found 1 match: '{matches[0]}'"
        else:
            examples = ", ".join(f"'{m}'" for m in unique)
            return f"Found {count} matches: {examples}"
    
    def _replace(self, text: str, pattern: str, replacement: str) -> str:
        """Replace pattern in text."""
        # Try regex first
        try:
            result = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            count = len(re.findall(pattern, text, re.IGNORECASE))
        except re.error:
            # Literal replacement (case-insensitive)
            count = text.lower().count(pattern.lower())
            result = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)
        
        if count == 0:
            return f"No matches found for '{pattern}'"
        
        return result
    
    def _extract(self, text: str, pattern: str) -> str:
        """Extract pattern matches from text."""
        try:
            matches = re.findall(pattern, text)
        except re.error:
            return f"Error: Invalid regex pattern '{pattern}'"
        
        if not matches:
            return f"No matches found for pattern '{pattern}'"
        
        # If pattern has groups, flatten
        if matches and isinstance(matches[0], tuple):
            matches = [m[0] if len(m) == 1 else m for m in matches]
        
        # Return as comma-separated list
        if len(matches) <= 10:
            return ", ".join(str(m) for m in matches)
        else:
            return ", ".join(str(m) for m in matches[:10]) + f"... ({len(matches)} total)"
    
    def _split(self, text: str, pattern: str) -> str:
        """Split text by pattern."""
        try:
            parts = re.split(pattern, text)
        except re.error:
            parts = text.split(pattern)
        
        # Filter empty parts
        parts = [p for p in parts if p.strip()]
        
        if len(parts) <= 10:
            return " | ".join(parts)
        else:
            return " | ".join(parts[:10]) + f"... ({len(parts)} parts total)"
