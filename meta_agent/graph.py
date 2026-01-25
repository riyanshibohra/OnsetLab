"""
Meta-Agent LangGraph Definition
===============================
Defines the LangGraph workflow for MCP discovery and notebook generation.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from meta_agent.state import MetaAgentState, create_initial_state

# Node imports
from meta_agent.nodes import (
    parse_problem,
    search_mcp_servers,
    evaluate_mcp_results,
    extract_schemas,
    mark_as_api,
    compile_results,
    filter_tools,
    generate_token_guides,
    generate_notebook,
)


def route_after_evaluate(state: MetaAgentState) -> Literal["extract_schemas", "mark_as_api"]:
    """Route based on MCP evaluation result."""
    if state.get("result_quality") == "good_mcp":
        return "extract_schemas"
    return "mark_as_api"


def route_after_service(state: MetaAgentState) -> Literal["search_mcp_servers", "compile_results"]:
    """Check if more services to process."""
    current_index = state.get("current_service_index", 0)
    services = state.get("identified_services", [])
    
    if current_index < len(services):
        return "search_mcp_servers"
    return "compile_results"


def route_after_parse(state: MetaAgentState) -> Literal["search_mcp_servers", "compile_results"]:
    """Check if any services were identified."""
    services = state.get("identified_services", [])
    
    if services:
        return "search_mcp_servers"
    # No services found - skip to compile (will generate empty notebook)
    return "compile_results"


def create_meta_agent_graph() -> StateGraph:
    """
    Create and compile the Meta-Agent LangGraph.
    
    Graph Flow:
    
                        ┌─────────────────┐
                        │  parse_problem  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ has services?   │
                        └────────┬────────┘
                           yes   │   no
                     ┌───────────┴───────────┐
                     ▼                       ▼
            ┌─────────────────┐    ┌─────────────────┐
            │search_mcp_servers│    │ compile_results │──┐
            └────────┬────────┘    └─────────────────┘  │
                     │                                   │
                     ▼                                   │
            ┌─────────────────────┐                      │
            │ evaluate_mcp_results │                     │
            └─────────┬───────────┘                      │
                      │                                  │
               good   │   no_mcp                         │
            ┌─────────┴─────────┐                        │
            ▼                   ▼                        │
    ┌───────────────┐   ┌─────────────┐                  │
    │extract_schemas│   │ mark_as_api │                  │
    └───────┬───────┘   └──────┬──────┘                  │
            │                  │                         │
            └────────┬─────────┘                         │
                     │                                   │
                     ▼                                   │
            ┌─────────────────┐                          │
            │ more services?  │                          │
            └────────┬────────┘                          │
               yes   │   no                              │
        ┌────────────┴────────────┐                      │
        │ (loop back)             ▼                      │
        │               ┌─────────────────┐              │
        └──────────────►│ compile_results │◄─────────────┘
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  filter_tools   │ (max 15-20)
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌──────────────────────┐
                        │generate_token_guides │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  generate_notebook  │
                        └──────────┬──────────┘
                                   │
                                   ▼
                              ┌─────────┐
                              │   END   │
                              └─────────┘
    
    See: meta_agent/meta_agent_graph.png for visual diagram
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    # Build the graph
    workflow = StateGraph(MetaAgentState)
    
    # Add nodes
    workflow.add_node("parse_problem", parse_problem)
    workflow.add_node("search_mcp_servers", search_mcp_servers)
    workflow.add_node("evaluate_mcp_results", evaluate_mcp_results)
    workflow.add_node("extract_schemas", extract_schemas)
    workflow.add_node("mark_as_api", mark_as_api)
    workflow.add_node("compile_results", compile_results)
    workflow.add_node("filter_tools", filter_tools)
    workflow.add_node("generate_token_guides", generate_token_guides)
    workflow.add_node("generate_notebook", generate_notebook)
    
    # Set entry point
    workflow.set_entry_point("parse_problem")
    
    # After parsing, check if we have services
    workflow.add_conditional_edges(
        "parse_problem",
        route_after_parse,
        {
            "search_mcp_servers": "search_mcp_servers",
            "compile_results": "compile_results"
        }
    )
    
    # Search -> Evaluate
    workflow.add_edge("search_mcp_servers", "evaluate_mcp_results")
    
    # Conditional: good MCP or fallback to API
    workflow.add_conditional_edges(
        "evaluate_mcp_results",
        route_after_evaluate,
        {
            "extract_schemas": "extract_schemas",
            "mark_as_api": "mark_as_api"
        }
    )
    
    # Both paths check for more services
    workflow.add_conditional_edges(
        "extract_schemas",
        route_after_service,
        {
            "search_mcp_servers": "search_mcp_servers",
            "compile_results": "compile_results"
        }
    )
    workflow.add_conditional_edges(
        "mark_as_api",
        route_after_service,
        {
            "search_mcp_servers": "search_mcp_servers",
            "compile_results": "compile_results"
        }
    )
    
    # Final pipeline
    workflow.add_edge("compile_results", "filter_tools")
    workflow.add_edge("filter_tools", "generate_token_guides")
    workflow.add_edge("generate_token_guides", "generate_notebook")
    workflow.add_edge("generate_notebook", END)
    
    # Compile and return
    return workflow.compile()


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
    tavily_api_key: str,
) -> dict:
    """
    Run the meta-agent to discover MCP servers and generate a Colab notebook.
    
    Args:
        problem_statement: Description of what the agent should do
        anthropic_api_key: Anthropic API key for Claude LLM calls
        tavily_api_key: Tavily API key for web search
        
    Returns:
        Dictionary with:
        - colab_notebook: The notebook JSON string
        - colab_notebook_url: URL to the hosted notebook (if uploaded)
        - mcp_servers: List of discovered MCP servers
        - api_servers: List of services needing API implementation
        - token_guides: Setup instructions for each service
        - tool_schemas: All discovered tool schemas
        - errors: Any errors encountered
    """
    # Create the graph
    graph = get_graph()
    
    # Initialize state
    initial_state = create_initial_state(
        problem_statement=problem_statement,
        anthropic_api_key=anthropic_api_key,
        tavily_api_key=tavily_api_key,
    )
    
    print("\n" + "=" * 60)
    print("🤖 OnsetLab Meta-Agent")
    print("=" * 60)
    print(f"\n📝 Problem Statement:\n{problem_statement}\n")
    
    # Run the graph
    result = await graph.ainvoke(initial_state)
    
    print("\n" + "=" * 60)
    print("✅ Meta-Agent Complete!")
    print("=" * 60)
    
    # Use filtered results if available, fall back to unfiltered
    mcp_servers = result.get("filtered_mcp_servers") or result.get("mcp_servers", [])
    api_servers = result.get("filtered_api_servers") or result.get("api_servers", [])
    tool_schemas = result.get("filtered_tool_schemas") or result.get("tool_schemas", [])
    
    return {
        "colab_notebook": result.get("colab_notebook", ""),
        "colab_notebook_url": result.get("colab_notebook_url"),
        "mcp_servers": mcp_servers,
        "api_servers": api_servers,
        "token_guides": result.get("token_guides", []),
        "tool_schemas": tool_schemas,
        "errors": result.get("errors", []),
    }


def run_meta_agent_sync(
    problem_statement: str,
    anthropic_api_key: str,
    tavily_api_key: str,
) -> dict:
    """
    Synchronous version of run_meta_agent.
    
    Useful for testing and CLI usage.
    """
    import asyncio
    return asyncio.run(run_meta_agent(
        problem_statement=problem_statement,
        anthropic_api_key=anthropic_api_key,
        tavily_api_key=tavily_api_key,
    ))
