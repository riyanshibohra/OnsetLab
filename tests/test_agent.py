"""Integration tests — full Agent runs with Ollama.

These tests require Ollama running with a model pulled.
They are skipped in CI (no Ollama available).
"""

import pytest
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime, UnitConverter, TextProcessor
from .conftest import requires_ollama


@requires_ollama
class TestAgentBasic:
    """Basic Agent functionality with Ollama."""

    def setup_method(self):
        self.agent = Agent(
            model="qwen2.5:3b",
            tools=[Calculator(), DateTime()],
            memory=False,
            verify=False,
        )

    def test_agent_creates(self):
        assert self.agent is not None

    def test_calculator_query(self):
        result = self.agent.run("What is 25 * 4?")
        assert result.answer is not None
        assert len(result.answer) > 0
        # The answer should contain "100" somewhere
        assert "100" in result.answer

    def test_direct_query(self):
        result = self.agent.run("hi")
        assert result.answer is not None
        assert result.strategy_used == "direct"

    def test_result_has_fields(self):
        result = self.agent.run("What is 2 + 2?")
        assert hasattr(result, "answer")
        assert hasattr(result, "plan")
        assert hasattr(result, "results")
        assert hasattr(result, "strategy_used")
        assert hasattr(result, "slm_calls")


@requires_ollama
class TestAgentTools:
    """Test Agent with different tool combinations."""

    def test_unit_converter(self):
        agent = Agent(
            model="qwen2.5:3b",
            tools=[UnitConverter()],
            memory=False,
            verify=False,
        )
        result = agent.run("Convert 10 kilometers to miles")
        assert result.answer is not None
        assert "6.2" in result.answer or "6.21" in result.answer

    def test_text_processor(self):
        agent = Agent(
            model="qwen2.5:3b",
            tools=[TextProcessor()],
            memory=False,
            verify=False,
        )
        result = agent.run("How many words in 'the quick brown fox'?")
        assert result.answer is not None

    def test_no_tools(self):
        agent = Agent(
            model="qwen2.5:3b",
            tools=[],
            memory=False,
            verify=False,
            routing=False,
        )
        result = agent.run("What is the capital of France?")
        assert result.answer is not None


@requires_ollama
class TestAgentMemory:
    """Test conversation memory."""

    def test_memory_tracks_messages(self):
        agent = Agent(
            model="qwen2.5:3b",
            tools=[Calculator()],
            memory=True,
            verify=False,
        )
        agent.run("What is 2 + 2?")
        agent.run("What about 3 + 3?")
        # Memory should have messages
        assert len(agent._memory) >= 2
