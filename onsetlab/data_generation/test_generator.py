"""
OnsetLab Test Generator
=======================
End-to-end pipeline test for the data generation system.

Runs:
1. Load MCP tool schemas
2. Generate training data (small sample)
3. Validate generated data
4. Show sample examples for manual review
5. Report pass/fail

Purpose: Verify the generator works before using in production.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import DataGenerator, GeneratorConfig, ToolSchema
from validator import Validator, ValidationResult


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def colored(text: str, color: str) -> str:
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.END}"


class TestGenerator:
    """
    End-to-end test for the data generation pipeline.
    
    Usage:
        tester = TestGenerator(
            tools_path="tools.json",
            api_key="sk-...",
            problem="calendar agent"
        )
        passed = tester.run()
    """
    
    # Thresholds
    MIN_QUALITY_SCORE = 90.0  # Minimum % valid to pass
    SAMPLE_COUNT = 10  # How many examples to generate for test
    EXAMPLES_TO_SHOW = 5  # How many to display for review
    
    def __init__(
        self,
        tools_path: str = None,
        tools: list[dict] = None,
        api_key: str = None,
        problem: str = "test agent",
        provider: str = None,
        verbose: bool = True
    ):
        """
        Initialize the test generator.
        
        Args:
            tools_path: Path to JSON file with MCP tool schemas
            tools: List of tool schema dicts (alternative to tools_path)
            api_key: LLM API key (or uses env var)
            problem: Problem statement for generation
            provider: API provider (auto-detected if not specified)
            verbose: Print detailed output
        """
        self.verbose = verbose
        self.problem = problem
        self.provider = provider
        
        # Load tools
        if tools:
            self.tools_data = tools
        elif tools_path:
            with open(tools_path) as f:
                self.tools_data = json.load(f)
        else:
            raise ValueError("Must provide either 'tools' or 'tools_path'")
        
        self.tools = [ToolSchema.from_mcp(t) for t in self.tools_data]
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        # Results
        self.generated_examples = []
        self.validation_result = None
        self.system_prompt = None
    
    def _print(self, *args, **kwargs):
        """Print if verbose mode."""
        if self.verbose:
            print(*args, **kwargs)
    
    def _print_header(self, text: str):
        """Print a section header."""
        self._print(f"\n{colored('═' * 60, Colors.BLUE)}")
        self._print(colored(f"  {text}", Colors.BOLD))
        self._print(colored('═' * 60, Colors.BLUE))
    
    def _print_step(self, step: int, text: str):
        """Print a step indicator."""
        self._print(f"\n{colored(f'Step {step}:', Colors.CYAN)} {text}")
    
    def run(self) -> bool:
        """
        Run the full end-to-end test.
        
        Returns:
            True if test passes, False otherwise
        """
        start_time = datetime.now()
        
        self._print_header("🧪 OnsetLab Generator Test")
        self._print(f"\n   Problem: {self.problem}")
        self._print(f"   Tools: {len(self.tools)}")
        self._print(f"   Test sample size: {self.SAMPLE_COUNT}")
        
        # Step 1: Generate data
        self._print_step(1, "Generating training data...")
        try:
            generated_path = self._generate_data()
            self._print(colored(f"   ✅ Generated {len(self.generated_examples)} examples", Colors.GREEN))
        except Exception as e:
            self._print(colored(f"   ❌ Generation failed: {e}", Colors.RED))
            return False
        
        # Step 2: Validate data
        self._print_step(2, "Validating generated data...")
        try:
            self._validate_data(generated_path)
            score = self.validation_result.quality_score
            if score >= self.MIN_QUALITY_SCORE:
                self._print(colored(f"   ✅ Quality score: {score:.1f}%", Colors.GREEN))
            else:
                self._print(colored(f"   ⚠️ Quality score: {score:.1f}% (below {self.MIN_QUALITY_SCORE}%)", Colors.YELLOW))
        except Exception as e:
            self._print(colored(f"   ❌ Validation failed: {e}", Colors.RED))
            return False
        
        # Step 3: Show samples
        self._print_step(3, "Sample examples for review...")
        self._show_samples()
        
        # Step 4: Show system prompt
        self._print_step(4, "Generated system prompt...")
        self._show_system_prompt()
        
        # Step 5: Final verdict
        elapsed = (datetime.now() - start_time).total_seconds()
        passed = self._show_verdict(elapsed)
        
        # Cleanup temp file
        try:
            os.unlink(generated_path)
        except:
            pass
        
        return passed
    
    def _generate_data(self) -> str:
        """Generate test data and return path to temp file."""
        # Create temp file for output
        fd, temp_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        
        # Configure generator
        config = GeneratorConfig(
            problem_statement=self.problem,
            tools=self.tools,
            api_key=self.api_key,
            api_provider=self.provider,
            num_examples=self.SAMPLE_COUNT,
            output_path=temp_path
        )
        
        # Run generator (suppress output)
        generator = DataGenerator(config)
        
        # Generate system prompt
        self.system_prompt = generator.generate_system_prompt()
        
        # Generate examples (quietly)
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            generator.generate_all()
            generator.save()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        
        self.generated_examples = generator.examples
        
        return temp_path
    
    def _validate_data(self, data_path: str):
        """Validate the generated data."""
        validator = Validator(tools=self.tools_data)
        self.validation_result = validator.validate(data_path)
    
    def _show_samples(self):
        """Display sample examples for manual review."""
        # Show a mix of tool calls and non-tool responses
        tool_examples = []
        other_examples = []
        
        for ex in self.generated_examples:
            messages = ex.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    if "<tool_call>" in msg.get("content", ""):
                        tool_examples.append(ex)
                    else:
                        other_examples.append(ex)
                    break
        
        # Show examples
        samples_shown = 0
        
        # Prioritize tool call examples
        for ex in tool_examples[:self.EXAMPLES_TO_SHOW - 1]:
            self._print_example(ex)
            samples_shown += 1
        
        # Show one non-tool example if available
        if other_examples and samples_shown < self.EXAMPLES_TO_SHOW:
            self._print_example(other_examples[0])
    
    def _print_example(self, example: dict):
        """Print a single example in readable format."""
        messages = example.get("messages", [])
        
        user_msg = ""
        assistant_msg = ""
        
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
        
        self._print(f"\n   {colored('User:', Colors.CYAN)} {user_msg}")
        
        # Format assistant message
        if "<tool_call>" in assistant_msg:
            # Extract and pretty-print tool call
            import re
            match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', assistant_msg, re.DOTALL)
            if match:
                try:
                    tool_call = json.loads(match.group(1))
                    tool_name = tool_call.get("tool", "?")
                    params = tool_call.get("parameters", {})
                    params_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in list(params.items())[:3])
                    if len(params) > 3:
                        params_str += ", ..."
                    self._print(f"   {colored('Tool:', Colors.GREEN)} {tool_name}({params_str})")
                except:
                    self._print(f"   {colored('Assistant:', Colors.GREEN)} {assistant_msg[:100]}...")
        else:
            self._print(f"   {colored('Assistant:', Colors.GREEN)} {assistant_msg[:100]}")
        
        self._print(colored("   " + "-" * 50, Colors.BLUE))
    
    def _show_system_prompt(self):
        """Display the generated system prompt."""
        if self.system_prompt:
            lines = self.system_prompt.split("\n")
            self._print(f"\n   {colored('Preview (first 8 lines):', Colors.CYAN)}")
            for line in lines[:8]:
                self._print(f"   │ {line[:70]}")
            if len(lines) > 8:
                self._print(f"   │ ... ({len(lines) - 8} more lines)")
    
    def _show_verdict(self, elapsed: float) -> bool:
        """Show final pass/fail verdict."""
        self._print_header("📋 Test Results")
        
        result = self.validation_result
        score = result.quality_score if result else 0
        passed = score >= self.MIN_QUALITY_SCORE and result.valid_examples > 0
        
        self._print(f"\n   {'✅' if passed else '❌'} Quality Score: {score:.1f}%")
        self._print(f"   📊 Valid: {result.valid_examples}/{result.total_examples}")
        
        if result.error_counts:
            self._print(f"\n   ⚠️ Errors found:")
            for error_type, count in result.error_counts.items():
                self._print(f"      - {error_type}: {count}")
        
        self._print(f"\n   ⏱️ Time: {elapsed:.1f}s")
        
        # Final verdict
        self._print(f"\n   {colored('═' * 40, Colors.BLUE)}")
        if passed:
            self._print(colored("   ✅ TEST PASSED - Generator is working!", Colors.GREEN + Colors.BOLD))
        else:
            self._print(colored("   ❌ TEST FAILED - Check errors above", Colors.RED + Colors.BOLD))
        self._print(colored("   " + "═" * 40, Colors.BLUE))
        
        return passed


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for the test generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="End-to-end test for the data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with tool schemas
  python test_generator.py --tools tools.json --problem "calendar agent"
  
  # Quick test (uses example tools)
  python test_generator.py --quick
  
  # Quiet mode (just pass/fail)
  python test_generator.py --tools tools.json --quiet
        """
    )
    
    parser.add_argument(
        "--tools", "-t",
        default=None,
        help="Path to JSON file with MCP tool schemas"
    )
    
    parser.add_argument(
        "--problem", "-p",
        default="test agent",
        help="Problem statement (default: 'test agent')"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test using example_tools.json in same directory"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show pass/fail result"
    )
    
    args = parser.parse_args()
    
    # Handle quick mode
    tools_path = args.tools
    if args.quick:
        tools_path = Path(__file__).parent / "example_tools.json"
        if not tools_path.exists():
            print("❌ Quick mode requires example_tools.json in same directory")
            return 1
        tools_path = str(tools_path)
    
    if not tools_path:
        print("❌ Error: Must provide --tools or use --quick")
        parser.print_help()
        return 1
    
    # Run test
    try:
        tester = TestGenerator(
            tools_path=tools_path,
            api_key=args.api_key,
            problem=args.problem,
            verbose=not args.quiet
        )
        passed = tester.run()
        
        if args.quiet:
            print("PASSED" if passed else "FAILED")
        
        return 0 if passed else 1
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
