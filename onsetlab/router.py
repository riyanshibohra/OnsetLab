"""
Model-Driven Router — strategy selection via LLM classification.

Routes tasks to the optimal execution strategy:
- REWOO: Plan-first for tasks that need tool calls (with ReAct fallback)
- DIRECT: No tool execution for conversational/meta queries

The model itself decides whether tools are needed — no hardcoded regex patterns.
"""

import re
import logging
from enum import Enum
from typing import List
from dataclasses import dataclass

from .tools.base import BaseTool

logger = logging.getLogger(__name__)


class Strategy(Enum):
    """Execution strategy."""
    REWOO = "rewoo"    # Plan-first: tool call via REWOO (with ReAct fallback)
    DIRECT = "direct"  # No tools: answer from model knowledge / conversation


@dataclass
class RoutingDecision:
    """Result of routing analysis."""
    strategy: Strategy
    confidence: float  # 0.0 - 1.0
    reason: str
    matched_tools: List[str]


# Trivial greetings that never need tools — no LLM call needed
_TRIVIAL = re.compile(
    r"^(hi|hey|hello|yo|sup|hiya|howdy|"
    r"bye|goodbye|see ya|later|cya|"
    r"thanks|thank you|thx|ty|"
    r"ok|okay|sure|got it|understood|"
    r"yes|no|yeah|nope|yep|"
    r"good|great|nice|cool|awesome)"
    r"[\s!.?]*$",
    re.IGNORECASE,
)

ROUTER_PROMPT = """Does this message need a tool/API call, or can you answer directly?

Tools available: {tool_names}

Conversation:
{context}

User: {query}

Reply with ONLY one word: TOOL or DIRECT
- TOOL = needs a tool call, OR the user is asking to DO something a tool can do
- DIRECT = purely factual/conversational, no tool needed

If unsure, say TOOL."""


class Router:
    """
    Model-driven task router.

    Uses the LLM itself (one ~5-token call) to decide whether the query
    needs tools.  Only trivial greetings are hardcoded; everything else
    goes through the model.
    """

    def __init__(self, model, tools: List[BaseTool], debug: bool = False):
        """
        Args:
            model: Any object with a ``generate(prompt, ...)`` method
                   (BaseModel, OllamaModel, OpenRouterModel, etc.)
            tools: Available tools (used to build the compact name list
                   shown to the model).
            debug: Print debug info.
        """
        self.model = model
        self.tools = tools
        self.debug = debug

    def route(self, query: str, context: str = "") -> RoutingDecision:
        """
        Classify the query and return a routing decision.

        Args:
            query:   User's message.
            context: Conversation history (optional).

        Returns:
            RoutingDecision with strategy, confidence, and reasoning.
        """
        # Fast path: trivial greetings never need tools
        if _TRIVIAL.match(query.strip()):
            return RoutingDecision(
                strategy=Strategy.DIRECT,
                confidence=1.0,
                reason="Trivial greeting",
                matched_tools=[],
            )

        # Fast path: no tools loaded — nothing to call
        if not self.tools:
            return RoutingDecision(
                strategy=Strategy.DIRECT,
                confidence=1.0,
                reason="No tools available",
                matched_tools=[],
            )

        # Model-driven classification
        tool_names = [t.name for t in self.tools[:20]]  # cap for prompt size
        prompt = ROUTER_PROMPT.format(
            tool_names=", ".join(tool_names),
            context=context or "(no prior conversation)",
            query=query,
        )

        try:
            raw = self.model.generate(prompt, max_tokens=5, temperature=0.0)
            answer = raw.strip().upper()
            logger.info(f"Router model response: {answer!r}")

            if "TOOL" in answer and "DIRECT" not in answer:
                return RoutingDecision(
                    strategy=Strategy.REWOO,
                    confidence=0.9,
                    reason="Model classified as TOOL",
                    matched_tools=tool_names[:5],
                )
            return RoutingDecision(
                strategy=Strategy.DIRECT,
                confidence=0.9,
                reason="Model classified as DIRECT",
                matched_tools=[],
            )

        except Exception as e:
            logger.warning(f"Router model call failed ({e}), defaulting to REWOO")
            # Safe default: attempt tool use
            return RoutingDecision(
                strategy=Strategy.REWOO,
                confidence=0.5,
                reason=f"Router fallback (model error: {e})",
                matched_tools=[],
            )
