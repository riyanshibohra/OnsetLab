"""Test Planner prompt parsing and THINK/PLAN splitting.

These tests verify the parser logic without needing a model (Ollama).
"""

import pytest

from onsetlab.rewoo.planner import Planner


class TestSplitThinkPlan:
    """Test the THINK/PLAN section splitter."""

    def test_full_format(self):
        response = (
            "THINK: I need Calculator to add these numbers\n"
            "PLAN:\n"
            '#E1 = Calculator(expression="2+2")'
        )
        think, plan = Planner._split_think_plan(response)
        assert "Calculator" in think
        assert "#E1" in plan
        assert "Calculator" in plan

    def test_no_plan_label(self):
        response = (
            "THINK: Use DateTime to get the current time\n"
            '#E1 = DateTime(operation="now")'
        )
        think, plan = Planner._split_think_plan(response)
        assert "DateTime" in think
        assert "#E1" in plan

    def test_no_think(self):
        response = '#E1 = Calculator(expression="5*3")'
        think, plan = Planner._split_think_plan(response)
        assert think == ""
        assert "#E1" in plan

    def test_just_tool_call(self):
        response = 'Calculator(expression="5*3")'
        think, plan = Planner._split_think_plan(response)
        assert "Calculator" in plan

    def test_empty(self):
        think, plan = Planner._split_think_plan("")
        assert think == ""
        assert plan == ""


class TestParsePlan:
    """Test plan parsing from model output."""

    def setup_method(self):
        # Create a Planner with a dummy model (we won't call .plan())
        from onsetlab.tools import Calculator, DateTime

        class DummyModel:
            def generate(self, *a, **kw):
                return ""
            def chat(self, *a, **kw):
                return ""
            @property
            def model_name(self):
                return "dummy"

        self.planner = Planner(DummyModel(), [Calculator(), DateTime()])

    def test_simple_step(self):
        steps = self.planner._parse_plan('#E1 = Calculator(expression="2+2")')
        assert len(steps) == 1
        assert steps[0]["tool"] == "Calculator"
        assert steps[0]["params"]["expression"] == "2+2"

    def test_multi_step(self):
        plan_text = (
            '#E1 = DateTime(operation="now")\n'
            '#E2 = Calculator(expression="100+200")'
        )
        steps = self.planner._parse_plan(plan_text)
        assert len(steps) == 2
        assert steps[0]["tool"] == "DateTime"
        assert steps[1]["tool"] == "Calculator"

    def test_depends_on(self):
        plan_text = (
            '#E1 = DateTime(operation="now")\n'
            "#E2 = Calculator(expression=#E1)"
        )
        steps = self.planner._parse_plan(plan_text)
        assert len(steps) == 2
        assert "#E1" in steps[1]["depends_on"]

    def test_integer_params(self):
        steps = self.planner._parse_plan(
            '#E1 = DateTime(operation="add_days", date="2025-01-01", days=10)'
        )
        assert len(steps) == 1
        assert steps[0]["params"]["days"] == 10

    def test_boolean_params(self):
        steps = self.planner._parse_plan(
            "#E1 = SomeTool(flag=true, other=false)"
        )
        assert len(steps) == 1
        assert steps[0]["params"]["flag"] is True
        assert steps[0]["params"]["other"] is False

    def test_colon_syntax_fix(self):
        # Model sometimes outputs param: "value" instead of param="value"
        steps = self.planner._parse_plan(
            '#E1 = Calculator(expression: "2+2")'
        )
        assert len(steps) == 1
        assert steps[0]["params"]["expression"] == "2+2"

    def test_no_prefix(self):
        # Model outputs tool call without #E1 =
        steps = self.planner._parse_plan('Calculator(expression="2+2")')
        assert len(steps) == 1
        assert steps[0]["tool"] == "Calculator"

    def test_deduplication(self):
        plan_text = (
            '#E1 = Calculator(expression="2+2")\n'
            '#E2 = Calculator(expression="2+2")'
        )
        steps = self.planner._parse_plan(plan_text)
        steps = self.planner._deduplicate_steps(steps)
        assert len(steps) == 1

    def test_empty_plan(self):
        steps = self.planner._parse_plan("")
        assert steps == []

    def test_garbage_input(self):
        steps = self.planner._parse_plan("I don't know what tool to use.")
        assert steps == []


class TestValidateSteps:
    """Test step validation (tool name matching, required params)."""

    def setup_method(self):
        from onsetlab.tools import Calculator, DateTime

        class DummyModel:
            def generate(self, *a, **kw):
                return ""
            def chat(self, *a, **kw):
                return ""
            @property
            def model_name(self):
                return "dummy"

        self.planner = Planner(DummyModel(), [Calculator(), DateTime()])

    def test_valid_tool(self):
        steps = [{"id": "#E1", "tool": "Calculator", "params": {"expression": "2+2"}, "depends_on": []}]
        validated = self.planner._validate_steps(steps)
        assert len(validated) == 1
        assert "error" not in validated[0]

    def test_case_insensitive_match(self):
        steps = [{"id": "#E1", "tool": "calculator", "params": {"expression": "2+2"}, "depends_on": []}]
        validated = self.planner._validate_steps(steps)
        assert validated[0]["tool"] == "Calculator"

    def test_unknown_tool_has_error(self):
        steps = [{"id": "#E1", "tool": "NonExistentTool", "params": {}, "depends_on": []}]
        validated = self.planner._validate_steps(steps)
        assert "error" in validated[0]
        assert "not found" in validated[0]["error"]

    def test_missing_required_param(self):
        steps = [{"id": "#E1", "tool": "DateTime", "params": {}, "depends_on": []}]
        validated = self.planner._validate_steps(steps)
        assert "error" in validated[0]
        assert "missing" in validated[0]["error"].lower()
