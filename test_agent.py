#!/usr/bin/env python3
"""
Quick test script for OnsetLab agent.

Run: python test_agent.py

Requires: Ollama running with phi3.5 model
  ollama pull phi3.5
  ollama serve
"""

from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime


def test_calculator():
    """Test basic calculator functionality."""
    print("\n=== Testing Calculator Tool ===")
    calc = Calculator()
    
    # Test basic math
    result = calc.execute(expression="15 * 0.20")
    print(f"15 * 0.20 = {result}")
    assert result == "3", f"Expected 3, got {result}"
    
    # Test percentage
    result = calc.execute(expression="84.50 * 0.15")
    print(f"84.50 * 0.15 = {result}")
    assert float(result) == 12.675, f"Expected 12.675, got {result}"
    
    print("✅ Calculator tests passed!")


def test_datetime():
    """Test DateTime tool functionality."""
    print("\n=== Testing DateTime Tool ===")
    dt = DateTime()
    
    # Test day of week
    result = dt.execute(operation="day_of_week", date="2000-01-01")
    print(f"2000-01-01: {result}")
    assert "Saturday" in result, f"Expected Saturday, got {result}"
    
    # Test add days
    result = dt.execute(operation="add_days", date="2024-01-01", days=30)
    print(f"2024-01-01 + 30 days = {result}")
    assert result == "2024-01-31", f"Expected 2024-01-31, got {result}"
    
    print("✅ DateTime tests passed!")


def test_agent_without_ollama():
    """Test agent setup (without actually calling Ollama)."""
    print("\n=== Testing Agent Setup ===")
    
    # This will fail if Ollama isn't running, which is expected
    try:
        agent = Agent(
            model="phi3.5",
            tools=[Calculator(), DateTime()],
            memory=True,
        )
        print(f"Agent created: {agent}")
        print(f"Tools: {list(agent.tools.keys())}")
        print("✅ Agent setup successful!")
        return agent
    except RuntimeError as e:
        print(f"⚠️  Ollama not running: {e}")
        print("   Start Ollama with: ollama serve")
        return None


def test_full_agent(agent):
    """Test full agent with Ollama."""
    if not agent:
        print("\n⚠️  Skipping full agent test (Ollama not running)")
        return
    
    print("\n=== Testing Full Agent (with Ollama) ===")
    
    try:
        # Test 1: Simple calculation
        print("\nTest 1: What's 15% of 84.50?")
        result = agent.run("What's 15% of 84.50?")
        print(f"Answer: {result.answer}")
        print(f"Plan: {result.plan}")
        print(f"Results: {result.results}")
        print(f"SLM calls: {result.slm_calls}")
    except RuntimeError as e:
        if "404" in str(e):
            print(f"\n⚠️  Model not found. Pull it with: ollama pull phi3.5")
            return
        raise
    
    # Test 2: Day of week
    print("\nTest 2: What day was January 1, 2000?")
    result = agent.run("What day of the week was January 1, 2000?")
    print(f"Answer: {result.answer}")
    print(f"SLM calls: {result.slm_calls}")
    
    # Test 3: Memory (follow-up)
    print("\nTest 3: Follow-up question")
    result = agent.run("What about January 2?")
    print(f"Answer: {result.answer}")
    
    print("\n✅ Full agent tests completed!")


if __name__ == "__main__":
    print("=" * 50)
    print("OnsetLab Agent Test Suite")
    print("=" * 50)
    
    # Test tools independently
    test_calculator()
    test_datetime()
    
    # Test agent
    agent = test_agent_without_ollama()
    test_full_agent(agent)
    
    print("\n" + "=" * 50)
    print("Test suite complete!")
    print("=" * 50)
