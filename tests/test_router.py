"""Test the model-driven Router.

Router tests need Ollama for the model-driven classification.
Trivial greeting tests work without it.
"""

import pytest
from onsetlab.router import Router, Strategy, RoutingDecision, _TRIVIAL
from onsetlab.tools import Calculator, DateTime, UnitConverter
from .conftest import requires_ollama


class TestTrivialGreetings:
    """These don't need a model — hardcoded fast path."""

    def test_hi(self):
        assert _TRIVIAL.match("hi")

    def test_hello_exclaim(self):
        assert _TRIVIAL.match("hello!")

    def test_bye(self):
        assert _TRIVIAL.match("bye")

    def test_thanks(self):
        assert _TRIVIAL.match("thanks")

    def test_ok(self):
        assert _TRIVIAL.match("ok")

    def test_not_trivial(self):
        assert not _TRIVIAL.match("what is 2 + 2")

    def test_not_trivial_sentence(self):
        assert not _TRIVIAL.match("hi, can you help me calculate something?")


class TestRouterNoTools:
    """Router with no tools should always return DIRECT."""

    def setup_method(self):
        class DummyModel:
            def generate(self, *a, **kw):
                return "DIRECT"
            @property
            def model_name(self):
                return "dummy"

        self.router = Router(DummyModel(), [])

    def test_no_tools_returns_direct(self):
        decision = self.router.route("what is 2+2?")
        assert decision.strategy == Strategy.DIRECT
        assert decision.confidence == 1.0

    def test_trivial_returns_direct(self):
        decision = self.router.route("hi")
        assert decision.strategy == Strategy.DIRECT
        assert decision.confidence == 1.0


@requires_ollama
class TestRouterWithOllama:
    """Integration tests that require Ollama running."""

    def setup_method(self):
        from onsetlab.model.ollama import OllamaModel
        model = OllamaModel("qwen2.5:3b")
        tools = [Calculator(), DateTime(), UnitConverter()]
        self.router = Router(model, tools)

    def test_math_routes_to_tool(self):
        decision = self.router.route("what is 25 * 17?")
        assert decision.strategy == Strategy.REWOO

    def test_greeting_routes_direct(self):
        decision = self.router.route("hi")
        assert decision.strategy == Strategy.DIRECT

    def test_conversational_routes_direct(self):
        decision = self.router.route("what is the capital of France?")
        assert decision.strategy == Strategy.DIRECT
