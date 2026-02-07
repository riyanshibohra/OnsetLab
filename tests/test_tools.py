"""Test all built-in tools work correctly."""

import re
import pytest

from onsetlab.tools import (
    Calculator,
    DateTime,
    UnitConverter,
    TextProcessor,
    RandomGenerator,
)


# ──────────────────────────────────────────────────────────────
# Calculator
# ──────────────────────────────────────────────────────────────

class TestCalculator:
    def setup_method(self):
        self.calc = Calculator()

    def test_basic_arithmetic(self):
        assert self.calc.execute(expression="2 + 2") == "4"
        assert self.calc.execute(expression="10 - 3") == "7"
        assert self.calc.execute(expression="6 * 7") == "42"
        assert self.calc.execute(expression="100 / 4") == "25"

    def test_float_result(self):
        result = self.calc.execute(expression="10 / 3")
        assert result.startswith("3.333")

    def test_exponentiation(self):
        assert self.calc.execute(expression="2 ^ 10") == "1024"

    def test_math_functions(self):
        assert self.calc.execute(expression="sqrt(16)") == "4"
        assert self.calc.execute(expression="abs(-5)") == "5"

    def test_division_by_zero(self):
        result = self.calc.execute(expression="1 / 0")
        assert "Error" in result

    def test_has_required_params(self):
        params = self.calc.parameters
        assert "properties" in params
        assert "expression" in params["properties"]
        assert "expression" in params["required"]

    def test_name_and_description(self):
        assert self.calc.name == "Calculator"
        assert len(self.calc.description) > 0


# ──────────────────────────────────────────────────────────────
# DateTime
# ──────────────────────────────────────────────────────────────

class TestDateTime:
    def setup_method(self):
        self.dt = DateTime()

    def test_now(self):
        result = self.dt.execute(operation="now")
        # Should return a date string like "2026-02-07 12:00:00"
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_day_of_week(self):
        result = self.dt.execute(operation="day_of_week", date="2000-01-01")
        assert "Saturday" in result

    def test_add_days(self):
        result = self.dt.execute(operation="add_days", date="2025-01-01", days=10)
        assert result == "2025-01-11"

    def test_difference(self):
        result = self.dt.execute(
            operation="difference", date="2025-01-01", date2="2025-01-31"
        )
        assert "30 days" in result

    def test_missing_param(self):
        result = self.dt.execute(operation="day_of_week")
        assert "Error" in result

    def test_has_required_params(self):
        params = self.dt.parameters
        assert "operation" in params["required"]


# ──────────────────────────────────────────────────────────────
# UnitConverter
# ──────────────────────────────────────────────────────────────

class TestUnitConverter:
    def setup_method(self):
        self.uc = UnitConverter()

    def test_km_to_miles(self):
        result = self.uc.execute(value=10, from_unit="km", to_unit="miles")
        assert "6.21" in result

    def test_celsius_to_fahrenheit(self):
        result = self.uc.execute(value=100, from_unit="celsius", to_unit="fahrenheit")
        assert "212" in result

    def test_kg_to_pounds(self):
        result = self.uc.execute(value=1, from_unit="kg", to_unit="lb")
        assert "2.2" in result

    def test_unknown_unit(self):
        result = self.uc.execute(value=1, from_unit="foo", to_unit="bar")
        assert "Error" in result or "Unknown" in result or "not" in result.lower()

    def test_name(self):
        assert self.uc.name == "UnitConverter"


# ──────────────────────────────────────────────────────────────
# TextProcessor
# ──────────────────────────────────────────────────────────────

class TestTextProcessor:
    def setup_method(self):
        self.tp = TextProcessor()

    def test_word_count(self):
        result = self.tp.execute(operation="word_count", text="hello world foo")
        assert "3" in result

    def test_char_count(self):
        result = self.tp.execute(operation="char_count", text="hello")
        assert "5" in result

    def test_uppercase(self):
        result = self.tp.execute(operation="uppercase", text="hello")
        assert result == "HELLO"

    def test_lowercase(self):
        result = self.tp.execute(operation="lowercase", text="HELLO")
        assert result == "hello"

    def test_reverse(self):
        result = self.tp.execute(operation="reverse", text="abc")
        assert result == "cba"

    def test_name(self):
        assert self.tp.name == "TextProcessor"


# ──────────────────────────────────────────────────────────────
# RandomGenerator
# ──────────────────────────────────────────────────────────────

class TestRandomGenerator:
    def setup_method(self):
        self.rg = RandomGenerator()

    def test_integer(self):
        result = self.rg.execute(operation="integer", min=1, max=10)
        num = int(result)
        assert 1 <= num <= 10

    def test_uuid(self):
        result = self.rg.execute(operation="uuid")
        # UUID format: 8-4-4-4-12 hex chars
        assert re.match(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            result,
            re.IGNORECASE,
        )

    def test_coin_flip(self):
        result = self.rg.execute(operation="coin")
        assert result.lower() in ("heads", "tails")

    def test_choice(self):
        result = self.rg.execute(operation="choice", choices="red,blue,green")
        assert result in ("red", "blue", "green")

    def test_name(self):
        assert self.rg.name == "RandomGenerator"
