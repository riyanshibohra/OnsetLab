#!/usr/bin/env python3
"""
Test the hybrid REWOO/ReAct routing approach.

Tests all routing scenarios:
1. DIRECT - Meta/help questions (no tools)
2. REWOO - Predictable single/multi-tool tasks
3. REACT - Exploratory/search tasks
4. FALLBACK - REWOO failure triggers ReAct
"""

import sys
import pytest
from onsetlab import Agent, Calculator, DateTime, UnitConverter, TextProcessor
from onsetlab.router import Router, Strategy
from .conftest import requires_ollama

pytestmark = requires_ollama

# Test cases for each strategy
TEST_CASES = [
    # DIRECT cases - should NOT execute tools
    {
        "query": "What tools do you have?",
        "expected_strategy": Strategy.DIRECT,
        "description": "Meta question about tools"
    },
    {
        "query": "Help me understand how to use this",
        "expected_strategy": Strategy.DIRECT,
        "description": "Help request"
    },
    {
        "query": "What can you do?",
        "expected_strategy": Strategy.DIRECT,
        "description": "Capability question"
    },
    
    # REWOO cases - predictable, structured tasks
    {
        "query": "What is 15 + 27?",
        "expected_strategy": Strategy.REWOO,
        "description": "Simple calculation"
    },
    {
        "query": "Calculate 8 * 9",
        "expected_strategy": Strategy.REWOO,
        "description": "Math with keyword"
    },
    {
        "query": "What time is it?",
        "expected_strategy": Strategy.REWOO,
        "description": "Time query"
    },
    {
        "query": "Convert 100 km to miles",
        "expected_strategy": Strategy.REWOO,
        "description": "Unit conversion"
    },
    {
        "query": "Make this uppercase: hello world",
        "expected_strategy": Strategy.REWOO,
        "description": "Text processing"
    },
    {
        "query": "Calculate 5+3, then tell me the time",
        "expected_strategy": Strategy.REWOO,
        "description": "Multi-step predictable"
    },
    
    # REACT cases - exploratory/dynamic tasks
    # Note: Without search tools, these correctly route to DIRECT
    # When search tools (e.g., Tavily MCP) are added, they would route to REACT
    {
        "query": "Search for Python documentation",
        "expected_strategy": Strategy.DIRECT,  # No search tools available
        "description": "Search task (no search tools)"
    },
    {
        "query": "Find information about MCP servers",
        "expected_strategy": Strategy.DIRECT,  # No search tools available
        "description": "Find/explore task (no search tools)"
    },
    {
        "query": "Look for recent news",
        "expected_strategy": Strategy.DIRECT,  # No search tools available
        "description": "Look for task (no search tools)"
    },
]


def test_router():
    """Test the router classification."""
    print("\n" + "="*60)
    print("TESTING ROUTER CLASSIFICATION")
    print("="*60)
    
    from onsetlab.model.ollama import OllamaModel
    model = OllamaModel("qwen2.5:3b")
    tools = [Calculator(), DateTime(), UnitConverter(), TextProcessor()]
    router = Router(model, tools)
    
    passed = 0
    failed = 0
    
    for test in TEST_CASES:
        decision = router.route(test["query"])
        
        if decision.strategy == test["expected_strategy"]:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"\n[{status}] {test['description']}")
        print(f"  Query: {test['query'][:50]}...")
        print(f"  Expected: {test['expected_strategy'].value}")
        print(f"  Got: {decision.strategy.value} ({decision.confidence:.0%})")
        if status == "FAIL":
            print(f"  Reason: {decision.reason}")
    
    print(f"\n{'='*60}")
    print(f"Router Results: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
    print("="*60)
    
    assert failed == 0, f"{failed} router tests failed"


def test_agent_execution():
    """Test full agent execution with routing."""
    print("\n" + "="*60)
    print("TESTING AGENT EXECUTION")
    print("="*60)
    
    agent = Agent(
        model="phi3.5",
        tools=[Calculator(), DateTime(), UnitConverter(), TextProcessor()],
        routing=True,
        debug=True,
    )
    
    # Test cases with expected outcomes
    execution_tests = [
        {
            "query": "What tools do you have?",
            "expect_strategy": "direct",
            "expect_answer_contains": ["calculator", "datetime"],
        },
        {
            "query": "What is 25 + 17?",
            "expect_strategy": "rewoo",
            "expect_answer_contains": ["42"],
        },
        {
            "query": "What time is it?",
            "expect_strategy": "rewoo",
            "expect_answer_contains": [],  # Just check it runs
        },
        {
            "query": "Convert 100 celsius to fahrenheit",
            "expect_strategy": "rewoo",
            "expect_answer_contains": ["212"],
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in execution_tests:
        print(f"\n{'─'*50}")
        print(f"Query: {test['query']}")
        print("─"*50)
        
        try:
            result = agent.run(test["query"])
            
            # Check strategy
            strategy_match = test["expect_strategy"] in result.strategy_used
            
            # Check answer contains expected content
            answer_match = True
            for expected in test["expect_answer_contains"]:
                if expected.lower() not in result.answer.lower():
                    answer_match = False
                    break
            
            if strategy_match and answer_match:
                print(f"[PASS] Strategy: {result.strategy_used}")
                print(f"  Answer: {result.answer[:100]}...")
                passed += 1
            else:
                print(f"[FAIL]")
                print(f"  Expected strategy: {test['expect_strategy']}")
                print(f"  Got strategy: {result.strategy_used}")
                print(f"  Answer: {result.answer[:100]}...")
                failed += 1
                
        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Execution Results: {passed}/{passed+failed} passed")
    print("="*60)
    
    assert failed == 0, f"{failed} execution tests failed"


def test_fallback():
    """Test that REWOO fallback to ReAct works."""
    print("\n" + "="*60)
    print("TESTING REWOO->REACT FALLBACK")
    print("="*60)
    
    # This is harder to test directly without a tool that fails
    # We'll just verify the fallback mechanism exists and runs
    
    agent = Agent(
        model="phi3.5",
        tools=[Calculator()],
        routing=True,
        react_fallback=True,
        debug=True,
    )
    
    # Query that might cause issues (intentionally ambiguous)
    result = agent.run("Calculate something complicated: the square root of negative one")
    
    print(f"\nQuery: Calculate something complicated...")
    print(f"Strategy used: {result.strategy_used}")
    print(f"Used ReAct fallback: {result.used_react_fallback}")
    print(f"Answer: {result.answer[:100]}...")
    
    assert result.answer is not None  # Just checking it doesn't crash


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ONSETLAB HYBRID ROUTING TESTS")
    print("="*60)
    
    # Run all tests
    router_passed, router_failed = test_router()
    
    print("\n\nStarting agent execution tests...")
    print("(This requires Ollama to be running)")
    
    try:
        exec_passed, exec_failed = test_agent_execution()
        fb_passed, fb_failed = test_fallback()
    except RuntimeError as e:
        print(f"\nSkipping execution tests: {e}")
        exec_passed, exec_failed = 0, 0
        fb_passed, fb_failed = 0, 0
    
    # Summary
    total_passed = router_passed + exec_passed + fb_passed
    total_failed = router_failed + exec_failed + fb_failed
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Router tests:    {router_passed}/{router_passed+router_failed}")
    print(f"Execution tests: {exec_passed}/{exec_passed+exec_failed}")
    print(f"Fallback tests:  {fb_passed}/{fb_passed+fb_failed}")
    print(f"─"*60)
    print(f"TOTAL:           {total_passed}/{total_passed+total_failed}")
    print("="*60)
    
    sys.exit(0 if total_failed == 0 else 1)
