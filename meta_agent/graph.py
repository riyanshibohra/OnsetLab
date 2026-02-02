"""
Meta-Agent LangGraph Definition (Registry-Based v3.2)
======================================================
Dynamic MCP server discovery from official registry.
Falls back to curated registry for known services.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from meta_agent.state import MetaAgentState, create_initial_state
from meta_agent.nodes import (
    # Discovery (NEW)
    discover_servers,
    verify_servers,
    prepare_tools,
    # Existing
    load_registry,
    filter_tools,
    process_feedback,
    generate_skill,
    generate_token_guides,
    generate_notebook,
)


def route_after_feedback(state: MetaAgentState) -> Literal["discover_servers", "filter_tools", "generate_skill"]:
    """
    Route based on user feedback action.
    
    - "add_tools" → back to discover_servers (search for new services)
    - "remove_tools" → back to filter_tools (re-filter with removals)
    - "approved" → continue to generate_skill (then guides, then notebook)
    """
    action = state.get("feedback_action", "approved")
    
    if action == "add_tools":
        return "discover_servers"
    elif action == "remove_tools":
        return "filter_tools"
    else:
        return "generate_skill"


def route_after_verify(state: MetaAgentState) -> Literal["load_registry", "prepare_tools"]:
    """
    Route after verification:
    - If we have verified servers → prepare_tools → filter_tools
    - If verification failed → fallback to load_registry
    """
    verified = state.get("verified_servers", [])
    
    # Check if we have any successfully verified servers
    has_verified = any(v.get("verification", {}).get("verified", False) for v in verified)
    
    if has_verified:
        return "prepare_tools"
    else:
        # Fallback to curated registry
        return "load_registry"


def create_meta_agent_graph() -> StateGraph:
    """
    Create the Meta-Agent LangGraph with dynamic MCP discovery.
    
    Graph Flow (v3.2 - Dynamic Discovery):
    
        [UI: User selects services]
                    │
                    ▼
           ┌──────────────────┐
           │ discover_servers │  (Search MCP Registry)
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │  verify_servers  │  (Verify NPM/GitHub/etc)
           └────────┬─────────┘
                    │
           ┌────────┴─────────┐
           │                  │
     found │                  │not found
           │                  │
           ▼                  ▼
    ┌──────────────┐   ┌──────────────┐
    │ filter_tools │   │load_registry │  (Fallback to curated)
    └──────┬───────┘   └──────┬───────┘
           │                  │
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │ HITL: User reviews│
           │  process_feedback│
           └────────┬─────────┘
                    │
           ┌────────┴─────────┐
 add/remove│                  │approved
           ▼                  ▼
    (loop back)      ┌──────────────────┐
                     │  generate_skill  │
                     └────────┬─────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │ generate_guides  │
                   └────────┬─────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  generate_notebook   │
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
    
    # Discovery nodes (NEW)
    workflow.add_node("discover_servers", discover_servers)
    workflow.add_node("verify_servers", verify_servers)
    workflow.add_node("prepare_tools", prepare_tools)
    
    # Existing nodes
    workflow.add_node("load_registry", load_registry)  # Fallback
    workflow.add_node("filter_tools", filter_tools)
    workflow.add_node("process_feedback", process_feedback)
    workflow.add_node("generate_skill", generate_skill)
    workflow.add_node("generate_token_guides", generate_token_guides)
    workflow.add_node("generate_notebook", generate_notebook)
    
    # Set entry point - start with discovery
    workflow.set_entry_point("discover_servers")
    
    # Discovery flow
    workflow.add_edge("discover_servers", "verify_servers")
    
    # After verification, route based on results
    workflow.add_conditional_edges(
        "verify_servers",
        route_after_verify,
        {
            "prepare_tools": "prepare_tools",  # Verified → prepare tools
            "load_registry": "load_registry"   # Fallback if discovery failed
        }
    )
    
    # Prepare tools feeds into filter
    workflow.add_edge("prepare_tools", "filter_tools")
    
    # Fallback registry also goes to filter_tools
    workflow.add_edge("load_registry", "filter_tools")
    
    # Filter to feedback
    workflow.add_edge("filter_tools", "process_feedback")
    
    # Conditional routing after feedback
    workflow.add_conditional_edges(
        "process_feedback",
        route_after_feedback,
        {
            "discover_servers": "discover_servers",  # Loop back to discovery
            "filter_tools": "filter_tools",
            "generate_skill": "generate_skill"
        }
    )
    
    # Final pipeline: skill → guides → notebook
    workflow.add_edge("generate_skill", "generate_token_guides")
    workflow.add_edge("generate_token_guides", "generate_notebook")
    workflow.add_edge("generate_notebook", END)
    
    # Compile with interrupt before feedback processing
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
    selected_services: list[str],
    anthropic_api_key: str,
) -> dict:
    """
    Run the meta-agent to load registry and generate a Colab notebook.
    
    Args:
        problem_statement: Description of what the agent should do
        selected_services: Services selected by user in UI (e.g., ["github", "slack"])
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
    
    # Initialize state with UI-selected services
    initial_state = create_initial_state(
        problem_statement=problem_statement,
        selected_services=selected_services,
        anthropic_api_key=anthropic_api_key,
    )
    
    print("\n" + "=" * 60)
    print("🤖 OnsetLab Meta-Agent")
    print("=" * 60)
    print(f"\n📝 Problem: {problem_statement}")
    print(f"🔧 Services: {', '.join(selected_services)}\n")
    
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
        "full_skill": result.get("full_skill", ""),
        "condensed_rules": result.get("condensed_rules", ""),
        "errors": result.get("errors", []),
    }


def run_meta_agent_sync(
    problem_statement: str,
    selected_services: list[str],
    anthropic_api_key: str,
) -> dict:
    """
    Synchronous version of run_meta_agent.
    
    Useful for testing and CLI usage.
    """
    import asyncio
    return asyncio.run(run_meta_agent(
        problem_statement=problem_statement,
        selected_services=selected_services,
        anthropic_api_key=anthropic_api_key,
    ))


def run_with_hitl(
    problem_statement: str,
    selected_services: list[str],
    anthropic_api_key: str,
    feedback_handler = None,
):
    """
    Run meta-agent with Human-in-the-Loop support.
    
    Args:
        problem_statement: What the agent should do
        selected_services: Services selected by user (e.g., ["github", "slack"])
        anthropic_api_key: Anthropic API key
        feedback_handler: Callable that takes (state) and returns user_feedback string
                         If None, uses input() in terminal
    
    Returns:
        Final state after user approval
    
    Example:
        # Terminal mode
        result = run_with_hitl(
            problem_statement="Manage GitHub issues and send Slack notifications",
            selected_services=["github", "slack"],
            anthropic_api_key="sk-ant-..."
        )
        
        # UI mode
        def ui_feedback(state):
            tools = state["filtered_tools"]
            # Show in UI, wait for user input
            return user_input_from_ui
        
        result = run_with_hitl(
            problem_statement="...",
            selected_services=["github"],
            anthropic_api_key="sk-ant-...",
            feedback_handler=ui_feedback
        )
    """
    import asyncio
    
    async def run_with_loop():
        graph = get_graph()
        
        state = create_initial_state(
            problem_statement=problem_statement,
            selected_services=selected_services,
            anthropic_api_key=anthropic_api_key,
        )
        
        print("\n" + "=" * 60)
        print("🤖 OnsetLab Meta-Agent")
        print("=" * 60)
        print(f"\n📝 Problem: {problem_statement}")
        print(f"🔧 Services: {', '.join(selected_services)}\n")
        
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
