#!/usr/bin/env python3
"""
OnsetLab CLI - Interactive agent interface and benchmarking.

Usage:
    python -m onsetlab                              # Interactive mode
    python -m onsetlab --model qwen2.5:3b           # Specify model
    python -m onsetlab benchmark                    # Run benchmark
    python -m onsetlab benchmark --model phi3.5    # Benchmark specific model
"""

import argparse
import sys
from datetime import datetime

from .agent import Agent
from .tools import Calculator, DateTime, UnitConverter, TextProcessor, RandomGenerator


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 60)
    print("  OnsetLab - REWOO Agent with Tool Calling")
    print("=" * 60)
    print()


def print_help():
    """Print available commands."""
    print("""
Commands:
  /help      - Show this help
  /tools     - List available tools
  /memory    - Show conversation memory
  /clear     - Clear memory
  /debug     - Toggle debug mode
  /quit      - Exit

Just type your question to interact with the agent.
""")


def format_tools(agent):
    """Format tools list."""
    lines = ["\nAvailable tools:"]
    for name, tool in agent.tools.items():
        lines.append(f"  • {name}: {tool.description[:60]}...")
    return "\n".join(lines)


def format_memory(agent):
    """Format memory contents."""
    if not agent._memory or len(agent._memory) == 0:
        return "\nMemory is empty."
    
    lines = ["\nConversation memory:"]
    for msg in agent._memory.get_messages():
        role = "You" if msg["role"] == "user" else "Agent"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


def run_benchmark(args):
    """Run benchmark command."""
    from .benchmark import Benchmark
    
    print(f"\nOnsetLab Benchmark")
    print("=" * 50)
    
    try:
        if args.compare:
            # Compare multiple models
            models = [m.strip() for m in args.compare.split(",")]
            print(f"Comparing models: {', '.join(models)}")
            results = Benchmark.compare(models, verbose=args.verbose)
            Benchmark.print_comparison(results)
            for model, result in results.items():
                result.print_summary()
        else:
            # Single model benchmark
            result = Benchmark.run(
                model=args.model,
                categories=args.categories.split(",") if args.categories else None,
                verbose=args.verbose,
            )
            result.print_summary()
        
        return 0
    
    except RuntimeError as e:
        print(f"\nError: {e}")
        if "not responding" in str(e).lower():
            print(f"\nMake sure Ollama is running:")
            print(f"  ollama serve")
            print(f"  ollama pull {args.model}")
        return 1


def run_interactive(args):
    """Run interactive CLI."""
    print_banner()
    
    # Initialize agent
    print(f"Loading model: {args.model}")
    print("Initializing tools...")
    
    try:
        agent = Agent(
            model=args.model,
            tools=[
                Calculator(),
                DateTime(),
                UnitConverter(),
                TextProcessor(),
                RandomGenerator(),
            ],
            memory=not args.no_memory,
            verify=not args.no_verify,
            debug=args.debug,
        )
    except RuntimeError as e:
        print(f"\nError: {e}")
        if "not found" in str(e).lower() or "not responding" in str(e).lower():
            print(f"\nMake sure Ollama is running and the model is available:")
            print(f"  ollama serve")
            print(f"  ollama pull {args.model}")
        sys.exit(1)
    
    print(f"\nAgent ready!")
    print(f"   Model: {agent.model_name}")
    print(f"   Tools: {len(agent.tools)}")
    print(f"   Memory: {'enabled' if not args.no_memory else 'disabled'}")
    print(f"   Verify: {'enabled' if not args.no_verify else 'disabled'}")
    print_help()
    
    debug_mode = args.debug
    
    # Main loop
    while True:
        try:
            # Get input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd in ["/quit", "/exit", "/q"]:
                    print("\nGoodbye!")
                    break
                
                elif cmd == "/help":
                    print_help()
                
                elif cmd == "/tools":
                    print(format_tools(agent))
                
                elif cmd == "/memory":
                    print(format_memory(agent))
                
                elif cmd == "/clear":
                    agent.clear_memory()
                    print("\nMemory cleared.")
                
                elif cmd == "/debug":
                    debug_mode = not debug_mode
                    agent._debug = debug_mode
                    agent._planner.debug = debug_mode
                    print(f"\nDebug mode: {'ON' if debug_mode else 'OFF'}")
                
                else:
                    print(f"\nUnknown command: {user_input}")
                    print("Type /help for available commands.")
                
                continue
            
            # Run agent
            result = agent.run(user_input)
            
            # Display result
            print(f"\nAgent: {result.answer}")
            
            # Show stats in debug mode
            if debug_mode:
                print(f"\n   [Plan: {len(result.plan)} steps | SLM calls: {result.slm_calls}]")
                if result.results:
                    print(f"   [Results: {result.results}]")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        
        except Exception as e:
            print(f"\nError: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()


def run_export(args):
    """Run export command."""
    from .packaging import ConfigExporter
    
    # Load config if provided
    if args.config:
        config = ConfigExporter.load(args.config)
        model = config.get("onsetlab", {}).get("model", "phi3.5")
        tools_config = config.get("tools", [])
    else:
        model = args.model
        tools_config = None
    
    # Create agent with tools
    tools = []
    if tools_config:
        tool_map = {
            "Calculator": Calculator,
            "DateTime": DateTime,
            "UnitConverter": UnitConverter,
            "TextProcessor": TextProcessor,
            "RandomGenerator": RandomGenerator,
        }
        for tc in tools_config:
            tool_class = tool_map.get(tc.get("class"))
            if tool_class:
                tools.append(tool_class())
    else:
        # Default tools
        tools = [Calculator(), DateTime(), UnitConverter(), TextProcessor()]
    
    print(f"\nOnsetLab Export")
    print("=" * 50)
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    
    try:
        # Create agent (skip Ollama check for export)
        # We just need the configuration, not a running model
        from .packaging import ConfigExporter, DockerExporter, BinaryExporter
        
        # Create a minimal agent-like object for export
        class ExportableAgent:
            def __init__(self, model_name, tools, settings):
                self.model_name = model_name
                self._tools = tools
                self._mcp_servers = []
                self._memory = settings.get("memory", True)
                self._verify = settings.get("verify", True)
                self._routing_enabled = settings.get("routing", True)
                self._react_fallback_enabled = settings.get("react_fallback", True)
                self._max_replans = settings.get("max_replans", 1)
            
            @property
            def tools(self):
                return {t.name: t for t in self._tools}
        
        settings = {}
        if args.config:
            settings = config.get("onsetlab", {}).get("settings", {})
        
        agent = ExportableAgent(model, tools, settings)
        
        # Export based on format
        if args.format == "config":
            result = ConfigExporter.export(agent, args.output)
        elif args.format == "docker":
            result = DockerExporter.export(
                agent, 
                args.output, 
                include_ollama=args.include_ollama
            )
        elif args.format == "binary":
            result = BinaryExporter.export(agent, args.output)
        else:
            print(f"Unknown format: {args.format}")
            return 1
        
        print(f"\nExported to: {result}")
        
        if args.format == "docker":
            print(f"\nNext steps:")
            print(f"  cd {args.output}")
            print(f"  docker-compose up --build")
        elif args.format == "binary":
            print(f"\nUsage:")
            print(f"  python {result}              # Interactive mode")
            print(f"  python {result} 'What is 2+2?'  # Single query")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="OnsetLab - Local SLM Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m onsetlab                           # Interactive mode
  python -m onsetlab benchmark --model phi3.5  # Benchmark a model
  python -m onsetlab export --format docker    # Export as Docker
  python -m onsetlab export --format config    # Export config
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--model", default="phi3.5", help="Model to benchmark")
    bench_parser.add_argument("--compare", help="Compare models (comma-separated)")
    bench_parser.add_argument("--categories", help="Filter categories (comma-separated)")
    bench_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export agent")
    export_parser.add_argument("--format", required=True, 
                               choices=["config", "docker", "binary"],
                               help="Export format")
    export_parser.add_argument("--output", "-o", required=True,
                               help="Output path")
    export_parser.add_argument("--config", "-c",
                               help="Load from existing config file")
    export_parser.add_argument("--model", default="phi3.5",
                               help="Model name (if not using --config)")
    export_parser.add_argument("--include-ollama", action="store_true",
                               help="Bundle Ollama in Docker image")
    
    # Interactive mode args (default)
    parser.add_argument("--model", default="phi3.5", help="Ollama model name (default: phi3.5)")
    parser.add_argument("--no-memory", action="store_true", help="Disable conversation memory")
    parser.add_argument("--no-verify", action="store_true", help="Disable verification step")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Route to command
    if args.command == "benchmark":
        sys.exit(run_benchmark(args))
    elif args.command == "export":
        sys.exit(run_export(args))
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
