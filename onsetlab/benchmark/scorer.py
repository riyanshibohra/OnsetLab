"""Benchmark scoring - AST-based evaluation."""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    category: str
    passed: bool
    expected_tool: str
    actual_tool: Optional[str]
    expected_params: Dict[str, Any]
    actual_params: Dict[str, Any]
    time_ms: float
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    model: str
    total_tests: int
    passed: int
    failed: int
    results_by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    test_results: List[TestResult] = field(default_factory=list)
    avg_time_ms: float = 0.0
    
    @property
    def accuracy(self) -> float:
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"\n{'='*50}",
            f"Benchmark Results: {self.model}",
            f"{'='*50}",
            f"Overall: {self.passed}/{self.total_tests} ({self.accuracy:.1%})",
            f"Avg Time: {self.avg_time_ms:.0f}ms",
            "",
            "By Category:",
        ]
        
        for cat, scores in self.results_by_category.items():
            cat_acc = scores['passed'] / scores['total'] if scores['total'] > 0 else 0
            lines.append(f"  {cat}: {scores['passed']}/{scores['total']} ({cat_acc:.1%})")
        
        # Show failed tests
        failed = [r for r in self.test_results if not r.passed]
        if failed:
            lines.append(f"\nFailed Tests ({len(failed)}):")
            for r in failed[:5]:  # Show first 5
                lines.append(f"  - {r.test_id}: expected {r.expected_tool}, got {r.actual_tool}")
            if len(failed) > 5:
                lines.append(f"  ... and {len(failed) - 5} more")
        
        lines.append("="*50)
        return "\n".join(lines)
    
    def print_summary(self):
        """Print summary to console."""
        print(self.summary())


class Scorer:
    """Score model outputs against expected results."""
    
    @staticmethod
    def parse_tool_call(output: str) -> tuple:
        """
        Parse a tool call from model output.
        Returns (tool_name, params_dict) or (None, {}) if parsing fails.
        """
        # Find tool name followed by opening paren
        match = re.search(r'(?:#E\d+\s*=\s*)?([\w-]+)\s*\(', output)
        if not match:
            return None, {}
        
        tool_name = match.group(1)
        start = match.end()
        
        # Find matching closing paren (handle nested parens in quotes)
        depth = 1
        in_quotes = False
        quote_char = None
        end = start
        
        while end < len(output) and depth > 0:
            c = output[end]
            if in_quotes:
                if c == quote_char and output[end-1] != '\\':
                    in_quotes = False
            else:
                if c in '"\'':
                    in_quotes = True
                    quote_char = c
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
            end += 1
        
        params_str = output[start:end-1] if depth == 0 else output[start:]
        params = Scorer._parse_params(params_str)
        
        return tool_name, params
    
    @staticmethod
    def _parse_params(params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dict."""
        params = {}
        if not params_str.strip():
            return params
        
        # Extract quoted values: param="value"
        for match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
            params[match.group(1)] = match.group(2)
        
        # Extract unquoted values: param=123
        for match in re.finditer(r'(\w+)\s*=\s*([^,\s\)"]+)', params_str):
            key = match.group(1)
            if key in params:
                continue
            value = match.group(2)
            
            # Try to convert to number
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                else:
                    params[key] = value
        
        return params
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize tool name for comparison (handles snake_case vs CamelCase)."""
        # Convert to lowercase and remove underscores
        return name.lower().replace("_", "")
    
    @staticmethod
    def check_tool_match(expected: str, actual: Optional[str]) -> bool:
        """Check if tool names match (handles case and naming convention differences)."""
        if actual is None:
            return False
        return Scorer._normalize_name(expected) == Scorer._normalize_name(actual)
    
    # Equivalent unit names
    UNIT_EQUIVALENTS = {
        "fahrenheit": ["f", "°f", "degf"],
        "celsius": ["c", "°c", "degc"],
        "kilometers": ["km", "kilometer"],
        "miles": ["mi", "mile"],
        "meters": ["m", "meter"],
        "centimeters": ["cm", "centimeter"],
        "kilograms": ["kg", "kilogram"],
        "pounds": ["lbs", "lb", "pound"],
        "word_count": ["count_words", "count words", "wordcount"],
        "char_count": ["count_chars", "count characters", "charcount", "character_count"],
        "uppercase": ["upper", "to_upper", "toupper", "make uppercase"],
        "lowercase": ["lower", "to_lower", "tolower", "make lowercase", "make_lower"],
        "reverse": ["reversed", "rev"],
    }
    
    @staticmethod
    def _normalize_value(val: Any) -> str:
        """Normalize a value for comparison."""
        if isinstance(val, str):
            # Lowercase, remove spaces and quotes
            return val.lower().replace(" ", "").replace("'", "").replace('"', '')
        return str(val).lower()
    
    @staticmethod
    def _are_equivalent(expected: Any, actual: Any) -> bool:
        """Check if two values are equivalent (handles unit aliases, etc.)."""
        exp_norm = Scorer._normalize_value(expected)
        act_norm = Scorer._normalize_value(actual)
        
        # Direct match
        if exp_norm == act_norm:
            return True
        
        # Check equivalents
        for canonical, aliases in Scorer.UNIT_EQUIVALENTS.items():
            all_forms = [canonical] + aliases
            all_forms_norm = [Scorer._normalize_value(f) for f in all_forms]
            if exp_norm in all_forms_norm and act_norm in all_forms_norm:
                return True
        
        # Math expression equivalents: ^ and ** for power
        if "^" in act_norm or "**" in exp_norm:
            if exp_norm.replace("**", "^") == act_norm.replace("**", "^"):
                return True
        
        return False
    
    @staticmethod
    def check_params_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """
        Check if params match.
        Handles equivalent values (unit aliases, math notation, etc.)
        """
        if not expected:
            return True  # No expected params = pass
        
        for key, expected_val in expected.items():
            if key not in actual:
                return False
            
            if not Scorer._are_equivalent(expected_val, actual[key]):
                return False
        
        return True
    
    @staticmethod
    def score_test(
        test: Dict[str, Any],
        model_output: str,
        time_ms: float
    ) -> TestResult:
        """Score a single test."""
        expected_tool = test.get("expected", {}).get("tool")
        expected_params = test.get("expected", {}).get("params", {})
        should_abstain = test.get("expected", {}).get("abstain", False)
        
        actual_tool, actual_params = Scorer.parse_tool_call(model_output)
        
        # Check abstention
        if should_abstain:
            # Should NOT call any tool
            passed = actual_tool is None
        else:
            # Should call correct tool with correct params
            tool_match = Scorer.check_tool_match(expected_tool, actual_tool)
            params_match = Scorer.check_params_match(expected_params, actual_params)
            passed = tool_match and params_match
        
        return TestResult(
            test_id=test["id"],
            category=test["category"],
            passed=passed,
            expected_tool=expected_tool or "(abstain)",
            actual_tool=actual_tool,
            expected_params=expected_params,
            actual_params=actual_params,
            time_ms=time_ms,
        )
