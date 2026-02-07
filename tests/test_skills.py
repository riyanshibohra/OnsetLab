"""Test auto-generated tool rules from schemas."""

from onsetlab.tools import Calculator, DateTime, UnitConverter, TextProcessor
from onsetlab.skills import generate_tool_rules, generate_examples, HINTS


class TestGenerateToolRules:
    def test_returns_string(self):
        tools = [Calculator(), DateTime()]
        result = generate_tool_rules(tools)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_tool_names(self):
        tools = [Calculator(), DateTime(), UnitConverter()]
        result = generate_tool_rules(tools)
        assert "Calculator" in result
        assert "DateTime" in result
        assert "UnitConverter" in result

    def test_shows_required_params(self):
        tools = [Calculator()]
        result = generate_tool_rules(tools)
        assert "expression" in result
        assert "REQUIRED" in result

    def test_shows_enum_values(self):
        tools = [DateTime()]
        result = generate_tool_rules(tools)
        # DateTime has operation with enum values
        assert "now" in result or "day_of_week" in result

    def test_empty_tools(self):
        result = generate_tool_rules([])
        assert result == ""

    def test_max_tools_cap(self):
        # Create many tools — should cap output
        tools = [Calculator()] * 20
        result = generate_tool_rules(tools, max_tools=3)
        # Should only have 3 entries (though they'll be the same tool)
        lines = [l for l in result.strip().split("\n") if l.startswith("- ")]
        assert len(lines) <= 3


class TestGenerateExamples:
    def test_returns_string(self):
        tools = [Calculator(), DateTime()]
        result = generate_examples(tools)
        assert isinstance(result, str)

    def test_includes_tool_names(self):
        tools = [Calculator()]
        result = generate_examples(tools)
        assert "Calculator" in result

    def test_shows_param_format(self):
        tools = [Calculator()]
        result = generate_examples(tools)
        # Should show something like Calculator(expression="example")
        assert "expression=" in result

    def test_empty_tools(self):
        result = generate_examples([])
        assert result == ""


class TestHints:
    def test_github_hints_exist(self):
        assert "github" in HINTS
        assert "owner" in HINTS["github"].lower() or "repo" in HINTS["github"].lower()

    def test_slack_hints_exist(self):
        assert "slack" in HINTS

    def test_notion_hints_exist(self):
        assert "notion" in HINTS
