"""Benchmark runner - execute tests and collect results."""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..model.ollama import OllamaModel
from ..tools.calculator import Calculator
from ..tools.datetime_tool import DateTime
from ..tools.unit_converter import UnitConverter
from ..tools.text_processor import TextProcessor
from .scorer import Scorer, TestResult, BenchmarkResult


# Minimal benchmark prompt - tests raw tool-calling ability
BENCHMARK_PROMPT = '''Pick ONE tool and fill in parameters.

Tools:
{tools}

Reply with ONLY: tool_name(param="value")

Query: {query}
Answer:'''


class Benchmark:
    """Run benchmarks on SLM models."""
    
    # Default tools for benchmarking
    DEFAULT_TOOLS = [
        Calculator(),
        DateTime(),
        UnitConverter(),
        TextProcessor(),
    ]
    
    @classmethod
    def _format_tools(cls) -> str:
        """Format tools for benchmark prompt."""
        lines = []
        for tool in cls.DEFAULT_TOOLS:
            params = tool.parameters
            if "properties" in params:
                props = params["properties"]
                required = params.get("required", [])
            else:
                props = params
                required = []
            
            param_strs = []
            for name, info in props.items():
                if isinstance(info, dict):
                    ptype = info.get("type", "string")
                    if name in required:
                        param_strs.append(f'{name}=<{ptype}>')
            
            params_desc = ", ".join(param_strs) if param_strs else ""
            lines.append(f"- {tool.name}({params_desc}): {tool.description[:60]}")
        
        return "\n".join(lines)
    
    @classmethod
    def load_tests(cls, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load test cases from suite.json."""
        suite_path = Path(__file__).parent / "tests" / "suite.json"
        
        if not suite_path.exists():
            raise FileNotFoundError(f"Test suite not found: {suite_path}")
        
        with open(suite_path) as f:
            all_tests = json.load(f)
        
        # Filter by category if specified
        if categories:
            all_tests = [t for t in all_tests if t["category"] in categories]
        
        return all_tests
    
    @classmethod
    def run(
        cls,
        model: str = "phi3.5",
        categories: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark on a model.
        
        Args:
            model: Ollama model name
            categories: Filter to specific categories (None = all)
            verbose: Print progress
            
        Returns:
            BenchmarkResult with scores
        """
        # Load tests
        tests = cls.load_tests(categories)
        
        if not tests:
            raise ValueError("No tests to run")
        
        if verbose:
            print(f"\nRunning {len(tests)} tests on {model}...")
        
        # Setup model
        llm = OllamaModel(model)
        tools_desc = cls._format_tools()
        
        # Run tests
        results: List[TestResult] = []
        total_time = 0.0
        
        for i, test in enumerate(tests):
            if verbose:
                print(f"  [{i+1}/{len(tests)}] {test['id']}: {test['query'][:40]}...")
            
            # Build prompt
            prompt = BENCHMARK_PROMPT.format(
                tools=tools_desc,
                query=test["query"]
            )
            
            # Time the model call
            start = time.time()
            try:
                model_output = llm.generate(
                    prompt,
                    temperature=0.0,
                    max_tokens=100,
                    stop_sequences=["\n\n", "Query:", "Tools:"]
                )
            except Exception as e:
                model_output = f"Error: {e}"
            
            elapsed_ms = (time.time() - start) * 1000
            total_time += elapsed_ms
            
            # Score
            result = Scorer.score_test(test, model_output, elapsed_ms)
            results.append(result)
            
            if verbose and not result.passed:
                print(f"    FAIL: expected {result.expected_tool}, got {result.actual_tool}")
                if verbose and result.actual_tool:
                    print(f"          output: {model_output[:80]}")
        
        # Aggregate results
        passed = sum(1 for r in results if r.passed)
        
        # Group by category
        by_category: Dict[str, Dict[str, int]] = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = {"passed": 0, "total": 0}
            by_category[r.category]["total"] += 1
            if r.passed:
                by_category[r.category]["passed"] += 1
        
        return BenchmarkResult(
            model=model,
            total_tests=len(tests),
            passed=passed,
            failed=len(tests) - passed,
            results_by_category=by_category,
            test_results=results,
            avg_time_ms=total_time / len(tests) if tests else 0,
        )
    
    @classmethod
    def compare(
        cls,
        models: List[str],
        categories: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple models.
        
        Args:
            models: List of model names
            categories: Filter to specific categories
            verbose: Print progress
            
        Returns:
            Dict mapping model name to BenchmarkResult
        """
        results = {}
        
        for model in models:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Benchmarking: {model}")
                print("="*50)
            
            results[model] = cls.run(model, categories, verbose)
        
        return results
    
    @classmethod
    def print_comparison(cls, results: Dict[str, BenchmarkResult]):
        """Print comparison table."""
        print(f"\n{'='*60}")
        print("Model Comparison")
        print("="*60)
        print(f"{'Model':<20} {'Accuracy':<12} {'Avg Time':<12}")
        print("-"*60)
        
        for model, result in sorted(results.items(), key=lambda x: -x[1].accuracy):
            print(f"{model:<20} {result.accuracy:>10.1%} {result.avg_time_ms:>10.0f}ms")
        
        print("="*60)
