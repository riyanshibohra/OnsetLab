"""
Hybrid Router - Intelligent strategy selection for task execution.

Routes tasks to the optimal execution strategy:
- REWOO: Plan-first for predictable, multi-step tasks
- REACT: Iterative reasoning for exploratory/dynamic tasks  
- DIRECT: No tool execution for meta/conversational queries

The router uses tool-semantic matching (not fragile keywords) to classify tasks.
"""

import re
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .tools.base import BaseTool


class Strategy(Enum):
    """Execution strategy."""
    REWOO = "rewoo"    # Plan-first: fast, efficient for predictable tasks
    REACT = "react"    # Iterative: flexible for exploratory tasks
    DIRECT = "direct"  # No tools: meta questions, help, etc.


@dataclass
class RoutingDecision:
    """Result of routing analysis."""
    strategy: Strategy
    confidence: float  # 0.0 - 1.0
    reason: str
    matched_tools: List[str]


class Router:
    """
    Intelligent task router that selects the best execution strategy.
    
    Decision logic:
    1. Check for meta/help questions → DIRECT
    2. Check for exploratory indicators → REACT  
    3. Match task against tool descriptions → REWOO if clear match
    4. Default to REWOO (with fallback to REACT on failure)
    """
    
    # Patterns indicating no tools needed (meta questions)
    META_PATTERNS = [
        r'\bwhat tools\b',
        r'\blist.*(tools|capabilities)\b',
        r'\bwhat can you do\b',
        r'\bhelp\b',
        r'\bhow do (you|i) use\b',
        r'\bwhat are your\b',
    ]
    
    # Patterns indicating exploratory/dynamic tasks (favor ReAct)
    EXPLORATORY_PATTERNS = [
        r'\bsearch\b',
        r'\bfind\b(?!.*file)',  # "find" but not "find file" (file ops are REWOO)
        r'\blook for\b',
        r'\bexplore\b',
        r'\bwhat.*(available|exist)\b',
        r'\bbrowse\b',
        r'\bdiscover\b',
    ]
    
    # Patterns indicating multi-step predictable tasks (favor REWOO)
    SEQUENTIAL_PATTERNS = [
        r'\bthen\b',
        r'\bafter that\b',
        r'\bnext\b',
        r'\bfinally\b',
        r'\bfirst\b.*\bthen\b',
        r'\band\b.*\band\b',  # Multiple "and"s suggest multi-step
    ]
    
    def __init__(self, tools: List[BaseTool], debug: bool = False):
        self.tools = {t.name: t for t in tools}
        self.debug = debug
    
    def route(self, task: str) -> RoutingDecision:
        """
        Analyze task and return optimal execution strategy.
        
        Args:
            task: User's task/query
            
        Returns:
            RoutingDecision with strategy, confidence, and reasoning
        """
        task_lower = task.lower().strip()
        
        # 1. Check for meta/help questions (DIRECT - no tools needed)
        if self._is_meta_question(task_lower):
            return RoutingDecision(
                strategy=Strategy.DIRECT,
                confidence=0.9,
                reason="Meta/help question - no tool execution needed",
                matched_tools=[]
            )
        
        # 2. Find matching tools based on semantic relevance
        matched_tools = self._find_matching_tools(task_lower)
        
        # 3. Check for exploratory patterns
        is_exploratory = self._is_exploratory(task_lower)
        
        # 4. Check for sequential/multi-step patterns
        is_sequential = self._is_sequential(task_lower)
        
        # 5. Make routing decision
        if len(matched_tools) == 0:
            # No tools match - might be conversational or model should handle directly
            return RoutingDecision(
                strategy=Strategy.DIRECT,
                confidence=0.6,
                reason="No tools matched - treating as conversational",
                matched_tools=[]
            )
        
        if is_exploratory and not is_sequential:
            # Exploratory task - use ReAct for flexibility
            return RoutingDecision(
                strategy=Strategy.REACT,
                confidence=0.8,
                reason="Exploratory task - using iterative reasoning",
                matched_tools=matched_tools
            )
        
        # Default: Use REWOO for structured execution
        # Works well for single-tool and multi-step predictable tasks
        confidence = 0.9 if len(matched_tools) == 1 else 0.7
        reason = "Clear tool match" if len(matched_tools) == 1 else "Multi-tool task"
        
        return RoutingDecision(
            strategy=Strategy.REWOO,
            confidence=confidence,
            reason=f"{reason} - using plan-first approach",
            matched_tools=matched_tools
        )
    
    def _is_meta_question(self, task: str) -> bool:
        """Check if task is a meta/help question."""
        for pattern in self.META_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                return True
        return False
    
    def _is_exploratory(self, task: str) -> bool:
        """Check if task is exploratory/search-like."""
        for pattern in self.EXPLORATORY_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                return True
        return False
    
    def _is_sequential(self, task: str) -> bool:
        """Check if task has sequential/multi-step indicators."""
        for pattern in self.SEQUENTIAL_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                return True
        return False
    
    def _find_matching_tools(self, task: str) -> List[str]:
        """
        Find tools that semantically match the task.
        
        Uses tool name, description keywords, and action verbs for matching.
        More robust than simple keyword matching.
        """
        matches = []
        task_words = set(task.lower().split())
        
        for name, tool in self.tools.items():
            score = 0
            
            # Check tool name (exact or partial match)
            name_lower = name.lower().replace('_', ' ')
            name_parts = name_lower.split()
            
            if name_lower in task:
                score += 5
            elif any(part in task for part in name_parts if len(part) > 3):
                score += 3
            
            # Check description keywords (first 15 words)
            desc_words = set(tool.description.lower().split()[:15])
            common_words = task_words.intersection(desc_words)
            # Filter out common stop words
            stop_words = {'a', 'an', 'the', 'is', 'are', 'to', 'for', 'of', 'and', 'or'}
            common_words = common_words - stop_words
            score += len(common_words) * 2
            
            # Check for action verbs that match tool purpose (highest weight)
            if self._task_matches_tool_purpose(task, tool):
                score += 4
            
            if score >= 2:  # Threshold for match
                matches.append(name)
        
        return matches
    
    def _task_matches_tool_purpose(self, task: str, tool: BaseTool) -> bool:
        """Check if task intent matches tool purpose."""
        # Map common verbs/patterns to tool types (use normalized names without underscores)
        verb_tool_map = {
            'calculator': [
                'calculate', 'compute', 'what is', 'solve', 'math', 
                'add', 'subtract', 'multiply', 'divide', 
                '+', '-', '*', '/', '=',
                'sum', 'product', 'difference', 'quotient',
                'power', 'square', 'root', 'percent',
            ],
            'datetime': [
                'time', 'date', 'today', 'now', 'current',
                'day', 'month', 'year', 'week', 'hour', 'minute',
            ],
            'unitconverter': [  # No underscore
                'convert', 'how many', 'to', 'from', 'units',
                'km', 'miles', 'meters', 'feet', 'inches',
                'celsius', 'fahrenheit', 'kelvin',
                'kg', 'pounds', 'lbs', 'ounces', 'grams',
                'liters', 'gallons', 'ml',
            ],
            'textprocessor': [  # No underscore
                'text', 'string', 'uppercase', 'lowercase', 
                'reverse', 'count', 'words', 'characters',
                'upper', 'lower', 'make',
            ],
            'random': ['random', 'generate', 'pick', 'roll', 'dice'],
        }
        
        # Normalize tool name (remove underscores, lowercase)
        tool_name = tool.name.lower().replace('_', '')
        
        for tool_type, patterns in verb_tool_map.items():
            if tool_type in tool_name:
                if any(pattern in task for pattern in patterns):
                    return True
        
        return False


def classify_task(task: str, tools: List[BaseTool], debug: bool = False) -> RoutingDecision:
    """
    Convenience function to classify a task.
    
    Args:
        task: User's task/query
        tools: Available tools
        debug: Print debug info
        
    Returns:
        RoutingDecision
    """
    router = Router(tools, debug)
    decision = router.route(task)
    
    if debug:
        print(f"\n[Router] Task: {task[:50]}...")
        print(f"[Router] Strategy: {decision.strategy.value}")
        print(f"[Router] Confidence: {decision.confidence:.0%}")
        print(f"[Router] Reason: {decision.reason}")
        print(f"[Router] Matched tools: {decision.matched_tools}")
    
    return decision
