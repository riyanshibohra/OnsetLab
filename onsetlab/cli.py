#!/usr/bin/env python3
"""
OnsetLab CLI - Interactive agent interface.

Usage:
    python -m onsetlab.cli
    python -m onsetlab.cli --model qwen2.5:3b
    python -m onsetlab.cli --no-memory
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


def main():
    parser = argparse.ArgumentParser(description="OnsetLab Interactive CLI")
    parser.add_argument("--model", default="phi3.5", help="Ollama model name (default: phi3.5)")
    parser.add_argument("--no-memory", action="store_true", help="Disable conversation memory")
    parser.add_argument("--no-verify", action="store_true", help="Disable verification step")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
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
        print(f"\n❌ Error: {e}")
        if "not found" in str(e).lower() or "not responding" in str(e).lower():
            print(f"\nMake sure Ollama is running and the model is available:")
            print(f"  ollama serve")
            print(f"  ollama pull {args.model}")
        sys.exit(1)
    
    print(f"\n✅ Agent ready!")
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
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd in ["/quit", "/exit", "/q"]:
                    print("\nGoodbye! 👋")
                    break
                
                elif cmd == "/help":
                    print_help()
                
                elif cmd == "/tools":
                    print(format_tools(agent))
                
                elif cmd == "/memory":
                    print(format_memory(agent))
                
                elif cmd == "/clear":
                    agent.clear_memory()
                    print("\n✓ Memory cleared.")
                
                elif cmd == "/debug":
                    debug_mode = not debug_mode
                    agent._debug = debug_mode
                    agent._planner.debug = debug_mode
                    print(f"\n✓ Debug mode: {'ON' if debug_mode else 'OFF'}")
                
                else:
                    print(f"\n❓ Unknown command: {user_input}")
                    print("Type /help for available commands.")
                
                continue
            
            # Run agent
            result = agent.run(user_input)
            
            # Display result
            print(f"\n🤖 Agent: {result.answer}")
            
            # Show stats in debug mode
            if debug_mode:
                print(f"\n   [Plan: {len(result.plan)} steps | SLM calls: {result.slm_calls}]")
                if result.results:
                    print(f"   [Results: {result.results}]")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
