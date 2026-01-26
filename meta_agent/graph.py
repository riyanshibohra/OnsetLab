"""
Meta-Agent LangGraph Definition (Registry-Based v2.0)
======================================================
Simplified workflow using registry instead of MCP discovery.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from meta_agent.state import MetaAgentState, create_initial_state
from meta_agent.nodes import (
    parse_problem,
    load_registry,
    filter_tools,
    process_feedback,
    generate_token_guides,
    generate_notebook,
)


def route_after_feedback(state: MetaAgentState) -> Literal["load_registry", "filter_tools", "generate_token_guides"]:
    """
    Route based on user feedback action.
    
    - "add_tools" → back to load_registry (might need new services)
    - "remove_tools" → back to filter_tools (re-filter with removals)
    - "approved" → continue to generate_token_guides
    """
    action = state.get("feedback_action", "approved")
    
    if action == "add_tools":
        return "load_registry"
    elif action == "remove_tools":
        return "filter_tools"
    else:
        return "generate_token_guides"


def create_meta_agent_graph() -> StateGraph:
    """
    Create the registry-based Meta-Agent LangGraph.
    
    Graph Flow:
    
                    ┌──────────────────┐
                    │  parse_problem   │  (Extract services from problem)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  load_registry   │  (Load tools from JSON files)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  filter_tools    │  (LLM selects relevant tools)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ HITL: User reviews│  (Human-in-the-Loop)
                    │  process_feedback│
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
          add/remove│                  │approved
                    │                  │
        ┌───────────┘                  └──────────┐
        │                                         │
        ▼                                         ▼
   (loop back)                        ┌──────────────────┐
   load_registry or                   │generate_guides   │
   filter_tools                       └────────┬─────────┘
        │                                      │
        │                                      ▼
        │                          ┌──────────────────────┐
        └────────────────────────► │ generate_notebook    │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                           ┌──────┐
                                           │ END  │
                                           └──────┘
    
    Returns:
        Compiled StateGraph with HITL interrupt point
    """
    # Build the graph
    workflow = StateGraph(MetaAgentState)
    
    # Add nodes
    workflow.add_node("parse_problem", parse_problem)
    workflow.add_node("load_registry", load_registry)
    workflow.add_node("filter_tools", filter_tools)
    workflow.add_node("process_feedback", process_feedback)
    workflow.add_node("generate_token_guides", generate_token_guides)
    workflow.add_node("generate_notebook", generate_notebook)
    
    # Set entry point
    workflow.set_entry_point("parse_problem")
    
    # Linear flow to HITL point
    workflow.add_edge("parse_problem", "load_registry")
    workflow.add_edge("load_registry", "filter_tools")
    workflow.add_edge("filter_tools", "process_feedback")
    
    # Conditional routing after feedback
    workflow.add_conditional_edges(
        "process_feedback",
        route_after_feedback,
        {
            "load_registry": "load_registry",
            "filter_tools": "filter_tools",
            "generate_token_guides": "generate_token_guides"
        }
    )
    
    # Final pipeline
    workflow.add_edge("generate_token_guides", "generate_notebook")
    workflow.add_edge("generate_notebook", END)
    
    # Compile with interrupt before feedback processing
    # This allows UI to pause, show tools to user, collect input, then resume
    return workflow.compile(interrupt_before=["process_feedback"])


# Create singleton graph instance
_graph = None

def get_graph():
    """Get or create the compiled graph."""
    global _graph
    if _graph is None:
        _graph = create_meta_agent_graph()
    return _graph


async def run_meta_agent(
    problem_statement: str,
    anthropic_api_key: str,
) -> dict:
    """
    Run the meta-agent to load registry and generate a Colab notebook.
    
    This is the simplified registry-based flow (no Tavily search needed).
    
    Args:
        problem_statement: Description of what the agent should do
        anthropic_api_key: Anthropic API key for Claude LLM calls
        
    Returns:
        Dictionary with:
        - colab_notebook: The notebook JSON string
        - colab_notebook_url: URL to the hosted notebook (if uploaded)
        - final_tools: User-approved tools
        - mcp_servers: MCP server configs from registry
        - token_guides: Setup instructions for each service
        - errors: Any errors encountered
    """
    # Create the graph
    graph = get_graph()
    
    # Initialize state
    initial_state = create_initial_state(
        problem_statement=problem_statement,
        anthropic_api_key=anthropic_api_key,
    )
    
    print("\n" + "=" * 60)
    print("🤖 OnsetLab Meta-Agent (Registry-Based)")
    print("=" * 60)
    print(f"\n📝 Problem Statement:\n{problem_statement}\n")
    
    # Run the graph (will pause at HITL point)
    result = await graph.ainvoke(initial_state)
    
    print("\n" + "=" * 60)
    print("✅ Meta-Agent Complete!")
    print("=" * 60)
    
    return {
        "colab_notebook": result.get("colab_notebook", ""),
        "colab_notebook_url": result.get("colab_notebook_url"),
        "final_tools": result.get("final_tools", []),
        "mcp_servers": result.get("mcp_servers", []),
        "token_guides": result.get("token_guides", []),
        "registry_services": result.get("registry_services", []),
        "errors": result.get("errors", []),
    }


def run_meta_agent_sync(
    problem_statement: str,
    anthropic_api_key: str,
) -> dict:
    """
    Synchronous version of run_meta_agent.
    
    Useful for testing and CLI usage.
    """
    import asyncio
    return asyncio.run(run_meta_agent(
        problem_statement=problem_statement,
        anthropic_api_key=anthropic_api_key,
    ))


def run_with_hitl(
    problem_statement: str,
    anthropic_api_key: str,
    feedback_handler = None,
):
    """
    Run meta-agent with Human-in-the-Loop support.
    
    Args:
        problem_statement: What the agent should do
        anthropic_api_key: Anthropic API key
        feedback_handler: Callable that takes (state) and returns user_feedback string
                         If None, uses input() in terminal
    
    Returns:
        Final state after user approval
    
    Example:
        # Terminal mode
        result = run_with_hitl(
            "Manage GitHub issues",
            "sk-ant-..."
        )
        
        # UI mode
        def ui_feedback(state):
            tools = state["filtered_tools"]
            # Show in UI, wait for user input
            return user_input_from_ui
        
        result = run_with_hitl(
            "Manage GitHub issues",
            "sk-ant-...",
            feedback_handler=ui_feedback
        )
    """
    import asyncio
    
    async def run_with_loop():
        graph = get_graph()
        
        state = create_initial_state(
            problem_statement=problem_statement,
            anthropic_api_key=anthropic_api_key,
        )
        
        print("\n" + "=" * 60)
        print("🤖 OnsetLab Meta-Agent (Registry-Based)")
        print("=" * 60)
        print(f"\n📝 Problem: {problem_statement}\n")
        
        # Run until HITL point
        state = await graph.ainvoke(state)
        
        # Loop until user approves
        while state.get("feedback_action") != "approved":
            # Show tools to user
            filtered_tools = state.get("filtered_tools", [])
            
            print("\n" + "=" * 60)
            print(f"📋 Selected {len(filtered_tools)} tools:")
            print("=" * 60)
            
            by_service = {}
            for tool in filtered_tools:
                service = tool.get("_service", "unknown")
                if service not in by_service:
                    by_service[service] = []
                by_service[service].append(tool)
            
            for service, tools in sorted(by_service.items()):
                print(f"\n{service.upper()}:")
                for tool in tools:
                    print(f"  • {tool['name']}: {tool['description'][:50]}...")
            
            print("\n" + "-" * 60)
            print("Options:")
            print("  - Type 'looks good' to continue")
            print("  - Type 'add TOOL_NAME' to add a tool")
            print("  - Type 'remove TOOL_NAME' to remove a tool")
            print("-" * 60)
            
            # Get user feedback
            if feedback_handler:
                user_input = feedback_handler(state)
            else:
                user_input = input("\nYour feedback: ").strip()
            
            # Update state with feedback and resume
            state["user_feedback"] = user_input
            state = await graph.ainvoke(state)
        
        print("\n✅ User approved tools!")
        print("   Generating notebook...\n")
        
        return state
    
    return asyncio.run(run_with_loop())
