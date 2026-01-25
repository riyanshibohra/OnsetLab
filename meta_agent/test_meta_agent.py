"""
Meta-Agent End-to-End Test
==========================
Test script to verify the complete meta-agent flow.

Usage:
    # From the OnsetLab root directory:
    
    # 1. Set up environment
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 2. Create .env file in meta_agent/ with your keys:
    #    OPENAI_API_KEY=sk-...
    #    TAVILY_API_KEY=tvly-...
    
    # 3. Run the test
    python3 meta_agent/test_meta_agent.py
    
    # Or with custom problem
    python3 meta_agent/test_meta_agent.py "I need an agent that manages my GitHub issues"
    
    # Alternative: Run as module
    python3 -m meta_agent.test_meta_agent
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Load .env file if it exists
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")


def check_env_vars():
    """Check required environment variables are set."""
    missing = []
    
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nSet them with:")
        print("   export OPENAI_API_KEY=sk-...")
        print("   export TAVILY_API_KEY=tvly-...")
        return False
    
    return True


async def test_meta_agent(problem_statement: str):
    """
    Run a full test of the meta-agent.
    
    Args:
        problem_statement: The test problem to solve
    """
    from meta_agent.graph import run_meta_agent
    
    print("\n" + "=" * 70)
    print("🧪 META-AGENT END-TO-END TEST")
    print("=" * 70)
    print(f"\n📝 Problem Statement:\n   {problem_statement}")
    print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-" * 70)
    
    # Run the meta-agent
    result = await run_meta_agent(
        problem_statement=problem_statement,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    
    print("\n" + "-" * 70)
    print("📊 TEST RESULTS")
    print("-" * 70)
    
    # Check results
    mcp_servers = result.get("mcp_servers", [])
    api_servers = result.get("api_servers", [])
    tool_schemas = result.get("tool_schemas", [])
    token_guides = result.get("token_guides", [])
    colab_notebook = result.get("colab_notebook", "")
    errors = result.get("errors", [])
    
    print(f"\n✅ MCP Servers Discovered: {len(mcp_servers)}")
    for server in mcp_servers:
        print(f"   • {server['service']}: {server['package']}")
        print(f"     Auth: {server['auth_type']}, Tools: {len(server.get('tools', []))}")
    
    print(f"\n🔧 API Servers (Fallback): {len(api_servers)}")
    for api in api_servers:
        print(f"   • {api['service']}: {api.get('base_url', 'N/A')}")
        print(f"     Auth: {api.get('auth_type', 'N/A')}, Endpoints: {len(api.get('endpoints', []))}")
    
    print(f"\n🔨 Total Tool Schemas: {len(tool_schemas)}")
    for tool in tool_schemas[:5]:
        desc = tool.get('description', '')[:50]
        print(f"   • {tool['name']}: {desc}")
    if len(tool_schemas) > 5:
        print(f"   ... and {len(tool_schemas) - 5} more")
    
    print(f"\n📚 Token Guides Generated: {len(token_guides)}")
    for guide in token_guides:
        print(f"   • {guide['service']}: {guide['env_var']} ({len(guide['steps'])} steps)")
    
    print(f"\n📓 Notebook Generated: {'Yes' if colab_notebook else 'No'}")
    if colab_notebook:
        try:
            notebook = json.loads(colab_notebook)
            cell_count = len(notebook.get("cells", []))
            print(f"   Cells: {cell_count}")
            print(f"   Size: {len(colab_notebook)} bytes")
        except json.JSONDecodeError:
            print("   ⚠️ Invalid JSON in notebook")
    
    if errors:
        print(f"\n⚠️ Errors Encountered: {len(errors)}")
        for error in errors:
            print(f"   • {error}")
    
    # Save outputs for inspection
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save notebook
    if colab_notebook:
        notebook_path = os.path.join(output_dir, "generated_notebook.ipynb")
        with open(notebook_path, "w") as f:
            f.write(colab_notebook)
        print(f"\n💾 Notebook saved to: {notebook_path}")
    
    # Save full results
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, "w") as f:
        # Convert to JSON-serializable format
        output = {
            "problem_statement": problem_statement,
            "mcp_servers": mcp_servers,
            "api_servers": api_servers,
            "tool_schemas": tool_schemas,
            "token_guides": token_guides,
            "errors": errors,
            "notebook_size": len(colab_notebook),
        }
        json.dump(output, f, indent=2, default=str)
    print(f"💾 Results saved to: {results_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    if not errors:
        print("✅ TEST PASSED - Meta-agent completed successfully!")
    else:
        print("⚠️ TEST COMPLETED WITH WARNINGS")
    print("=" * 70)
    
    return result


def main():
    """Main entry point for test script."""
    # Check environment
    if not check_env_vars():
        sys.exit(1)
    
    # Get problem statement from args or use default
    if len(sys.argv) > 1:
        problem_statement = " ".join(sys.argv[1:])
    else:
        # Default test problem
        problem_statement = """
        I need an agent that manages GitHub issues and sends Slack notifications
        """
    
    # Run the test
    try:
        result = asyncio.run(test_meta_agent(problem_statement))
        
        # Return exit code based on errors
        if result.get("errors"):
            sys.exit(1)
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
