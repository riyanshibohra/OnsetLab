"""
OnsetLab Agent - Hybrid REWOO/ReAct agent with intelligent routing.

The agent uses a hybrid approach:
1. ANALYZE task complexity using the Router
2. ROUTE to optimal strategy:
   - REWOO: Plan-first for predictable, structured tasks
   - REACT: Iterative reasoning for exploratory/dynamic tasks
   - DIRECT: No tool execution for meta/conversational queries
3. FALLBACK: If primary strategy fails, automatically try alternate
"""

import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .model.base import BaseModel
from .model.ollama import OllamaModel
from .tools.base import BaseTool
from .memory.conversation import ConversationMemory
from .rewoo.planner import Planner
from .rewoo.executor import Executor
from .rewoo.verifier import Verifier
from .rewoo.solver import Solver
from .rewoo.react_fallback import ReactFallback
from .mcp.server import MCPServer
from .router import Router, Strategy, RoutingDecision



@dataclass
class AgentResult:
    """Result from agent execution."""
    answer: str
    plan: List[Dict[str, Any]]
    results: Dict[str, str]
    verified: bool
    slm_calls: int
    used_react_fallback: bool = False
    strategy_used: str = "rewoo"  # rewoo, react, direct
    routing_decision: Optional[RoutingDecision] = None


class Agent:
    """
    OnsetLab Agent - Hybrid REWOO/ReAct agent with intelligent routing.
    
    Uses a hybrid approach that automatically selects the best execution strategy:
    - REWOO (plan-first): Fast for predictable, structured tasks
    - ReAct (iterative): Flexible for exploratory/dynamic tasks
    - Direct: No tools needed for meta/conversational queries
    
    Includes automatic fallback: if primary strategy fails, tries alternate.
    """
    
    def __init__(
        self,
        model: Union[str, BaseModel] = "phi3.5",
        tools: List[BaseTool] = None,
        mcp_servers: List[MCPServer] = None,
        memory: bool = True,
        verify: bool = True,
        max_replans: int = 1,
        react_fallback: bool = True,
        routing: bool = True,  # Enable intelligent routing
        debug: bool = False,
    ):
        """
        Create an agent.
        
        Args:
            model: Ollama model name or BaseModel instance
            tools: List of tools to use
            mcp_servers: List of MCP servers to connect
            memory: Enable conversation memory
            verify: Enable plan/result verification
            max_replans: Max replan attempts on failure
            react_fallback: Enable ReAct fallback when REWOO fails
            routing: Enable intelligent strategy routing (hybrid approach)
            debug: Print debug info
        """
        # Settings (set early for use in other methods)
        self._verify = verify
        self._max_replans = max_replans
        self._react_fallback_enabled = react_fallback
        self._routing_enabled = routing
        self._debug = debug
        
        # Setup model
        if isinstance(model, str):
            self._model = OllamaModel(model)
        else:
            self._model = model
        
        # Setup tools
        self._tools = tools or []
        self._mcp_servers = mcp_servers or []
        
        # Setup components
        all_tools = self._collect_all_tools()
        self._all_tools = all_tools  # Store for router
        self._planner = Planner(self._model, all_tools, debug=debug)
        self._executor = Executor(all_tools)
        self._verifier = Verifier(self._model, debug=debug) if verify else None
        self._solver = Solver(self._model)
        self._react = ReactFallback(self._model, all_tools, debug=debug) if react_fallback else None
        self._router = Router(self._model, all_tools, debug=debug) if routing else None
        
        # Setup memory - stores conversation with results
        self._memory = ConversationMemory(max_turns=10) if memory else None
        self._last_results: Dict[str, str] = {}  # Store last tool results
    
    def _collect_all_tools(self) -> List[BaseTool]:
        """Collect all tools including MCP tools."""
        all_tools = list(self._tools)
        
        # Connect to MCP servers and get their tools
        for server in self._mcp_servers:
            try:
                if not server.connected:
                    server.connect()
                mcp_tools = server.get_tools()
                all_tools.extend(mcp_tools)
                if self._debug:
                    print(f"[DEBUG] Loaded {len(mcp_tools)} tools from MCP server: {server.name}")
            except Exception as e:
                if self._debug:
                    print(f"[DEBUG] Failed to load MCP server {server.name}: {e}")
        
        return all_tools
    
    def _get_system_context(self) -> str:
        """Get system context with current datetime and MCP info."""
        now = datetime.now()
        parts = [f"Current datetime: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"]

        # Include MCP allowed directories so the planner uses correct paths
        for server in self._mcp_servers:
            if hasattr(server, '_config') and hasattr(server._config, 'args'):
                # Extract directory paths from MCP server args
                dirs = [a for a in server._config.args if a.startswith('/')]
                if dirs:
                    parts.append(f"Working directory: {dirs[-1]}")

        return "\n".join(parts)
    
    def _get_context(self) -> Optional[str]:
        """Get full context including system info and conversation history."""
        lines = []
        
        # Always include system context (current datetime)
        lines.append(self._get_system_context())
        
        # Add conversation memory if available
        if self._memory is not None and len(self._memory) > 0:
            if self._debug:
                print(f"[DEBUG] Memory has {len(self._memory)} messages")
            
            messages = self._memory.get_last_n_messages(6)
            if messages:
                lines.append("\nConversation:")
                for msg in messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = msg["content"]
                    if len(content) > 300:
                        content = content[:300] + "..."
                    lines.append(f"  {role}: {content}")
        
        # Add last results if available
        if self._last_results:
            lines.append("\nPrevious tool results:")
            for step_id, result in self._last_results.items():
                lines.append(f"  {step_id} = {result}")
        
        return "\n".join(lines)
    
    def _is_error_result(self, result: str) -> bool:
        """Check if a result looks like an error."""
        result_lower = result.lower()
        
        # Common error prefixes
        error_prefixes = [
            "error:", "error -", "failed:", "exception:",
            # OS/filesystem errors
            "eisdir:", "enoent:", "eacces:", "eperm:", "eexist:",
            "enotdir:", "enotempty:", "einval:", "eio:",
            # Common error patterns
            "illegal operation", "cannot ", "could not ", "unable to ",
            "invalid ", "no such ", "not found", "permission denied",
            "access denied", "operation not permitted",
        ]
        
        for prefix in error_prefixes:
            if result_lower.startswith(prefix) or prefix in result_lower:
                return True
        
        return False
    
    def _check_rewoo_failed(self, plan: List[Dict], results: Dict[str, str]) -> bool:
        """
        Check if REWOO failed and should fallback to ReAct.
        
        Returns True if:
        - No plan was generated
        - All plan steps have errors (validation failed)
        - All execution results are errors
        """
        # No plan at all
        if not plan:
            return True
        
        # All steps have errors (validation failures)
        all_steps_errored = all("error" in step for step in plan)
        if all_steps_errored:
            return True
        
        # All results are errors
        if results:
            all_results_errored = all(
                self._is_error_result(str(v)) for v in results.values()
            )
            if all_results_errored:
                return True
        
        return False
    
    def run(self, query: str) -> AgentResult:
        """
        Run a query using the hybrid approach.
        
        Flow:
        1. Route: Model classifies DIRECT vs TOOL
        2. Execute: DIRECT answers without tools; REWOO plans + executes
        3. Fallback: If REWOO fails, ReAct corrects
        """
        slm_calls = 0
        used_react = False
        strategy_used = "rewoo"
        routing_decision = None
        
        # Get context
        context = self._get_context()
        
        if self._debug:
            print(f"\n[DEBUG] Context:\n{context}\n")
        
        # ========================================
        # STEP 1: ROUTE — model decides DIRECT vs TOOL
        # ========================================
        if self._routing_enabled and self._router:
            routing_decision = self._router.route(query, context or "")
            slm_calls += 1  # Router uses one model call
            
            if self._debug:
                print(f"\n[DEBUG] Routing Decision:")
                print(f"  Strategy: {routing_decision.strategy.value}")
                print(f"  Confidence: {routing_decision.confidence:.0%}")
                print(f"  Reason: {routing_decision.reason}")
        else:
            # Default to REWOO if routing disabled
            routing_decision = RoutingDecision(
                strategy=Strategy.REWOO,
                confidence=1.0,
                reason="Routing disabled - defaulting to REWOO",
                matched_tools=[]
            )
        
        # ========================================
        # STEP 2: EXECUTE
        # ========================================
        
        # DIRECT: No tool execution needed
        if routing_decision.strategy == Strategy.DIRECT:
            if self._debug:
                print(f"[DEBUG] Using DIRECT strategy (no tools)")
            
            answer = self._handle_direct(query, context)
            slm_calls += 1
            strategy_used = "direct"
            
            if self._memory is not None:
                self._memory.add_user_message(query)
                self._memory.add_assistant_message(answer)
            
            return AgentResult(
                answer=answer,
                plan=[],
                results={},
                verified=True,
                slm_calls=slm_calls,
                strategy_used=strategy_used,
                routing_decision=routing_decision
            )
        
        # REWOO: Plan-first execution (with ReAct fallback on failure)
        if self._debug:
            print(f"[DEBUG] Using REWOO strategy (plan-first)")
        
        strategy_used = "rewoo"
        
        # Plan (1 SLM call)
        plan = self._planner.plan(query, context)
        slm_calls += 1
        
        # PRE-EXECUTION: Verify plan values match user intent
        if plan and self._verify and self._verifier:
            is_plan_valid, plan_reason, corrected_plan = self._verifier.verify_plan(query, plan)
            
            if not is_plan_valid:
                if self._debug:
                    print(f"[DEBUG] Plan verification issue: {plan_reason}")
                
                # Use corrected plan if available
                if corrected_plan != plan:
                    if self._debug:
                        print(f"[DEBUG] Using corrected plan")
                    plan = corrected_plan
        
        # Execute if we have a plan
        if plan:
            results = self._executor.execute(plan)
            self._last_results = results
        else:
            results = {}
            self._last_results = {}
        
        # ========================================
        # STEP 3: FALLBACK - If REWOO failed, try ReAct
        # ========================================
        if self._react_fallback_enabled and self._react and self._check_rewoo_failed(plan, results):
            if self._debug:
                print(f"[DEBUG] REWOO failed, using ReAct fallback")
            
            # Build failure context so ReAct knows what didn't work
            failure_context = ""
            if plan and results:
                for step in plan:
                    step_id = step.get("id", "?")
                    tool = step.get("tool", "?")
                    params = step.get("params", {})
                    result = results.get(step_id, "no result")
                    
                    failure_context = f"""PREVIOUS ATTEMPT FAILED - FIX IT:
Tool used: {tool}
Parameters: {params}
Error: {result}

Fix the parameters and try again with the SAME tool ({tool})."""
            
            answer, react_steps, react_results = self._react.run(query, failure_context)
            used_react = True
            strategy_used = "rewoo->react"  # Indicate fallback was used
            slm_calls += len(react_steps) + 1
            
            plan = react_steps
            results = react_results
            self._last_results = results
            
            if self._memory is not None:
                self._memory.add_user_message(query)
                if results:
                    result_summary = ", ".join([f"{k}={str(v)[:50]}" for k, v in results.items()])
                    self._memory.add_assistant_message(f"{answer} [Results: {result_summary}]")
                else:
                    self._memory.add_assistant_message(answer)
            
            return AgentResult(
                answer=answer,
                plan=plan,
                results=results,
                verified=True,
                slm_calls=slm_calls,
                used_react_fallback=True,
                strategy_used=strategy_used,
                routing_decision=routing_decision
            )
        
        # POST-EXECUTION: Verify results
        verified = True
        if plan and self._verify and self._verifier:
            is_valid, reason = self._verifier.verify(query, plan, results)
            if "SLM" in reason:
                slm_calls += 1
            
            if not is_valid and self._max_replans > 0:
                error_context = f"{context or ''}\nError: {reason}"
                plan = self._planner.plan(query, error_context)
                slm_calls += 1
                if plan:
                    results = self._executor.execute(plan)
                    self._last_results = results
        
        # Solve (1 SLM call)
        answer = self._solver.solve(query, plan, results, context)
        slm_calls += 1
        
        # Update memory
        if self._memory is not None:
            self._memory.add_user_message(query)
            if results:
                result_summary = ", ".join([f"{k}={str(v)}" for k, v in results.items()])
                self._memory.add_assistant_message(f"{answer} [Results: {result_summary}]")
            else:
                self._memory.add_assistant_message(answer)
        
        return AgentResult(
            answer=answer,
            plan=plan,
            results=results,
            verified=verified,
            slm_calls=slm_calls,
            used_react_fallback=used_react,
            strategy_used=strategy_used,
            routing_decision=routing_decision
        )
    
    def _handle_direct(self, query: str, context: Optional[str]) -> str:
        """Handle DIRECT strategy - answer without tool execution."""
        # Check if asking about tools
        query_lower = query.lower()
        
        if "what tools" in query_lower or "list" in query_lower and "tool" in query_lower:
            # List available tools
            tool_list = []
            for tool in self._all_tools:
                tool_list.append(f"- {tool.name}: {tool.description[:60]}...")
            return "Available tools:\n" + "\n".join(tool_list)
        
        if "help" in query_lower or "how" in query_lower:
            return f"I'm an AI assistant with {len(self._all_tools)} tools. Ask me to calculate, convert units, get the time, or process text. For MCP tools, I can interact with external services."
        
        # Generic conversational response
        return self._solver.solve(query, [], {}, context)
    
    def chat(self, message: str) -> str:
        """Interactive chat."""
        return self.run(message).answer
    
    def clear_memory(self):
        """Clear memory."""
        if self._memory is not None:
            self._memory.clear()
        self._last_results = {}
    
    def save_memory(self, path: str):
        """Save memory."""
        if self._memory is not None:
            self._memory.save(path)
    
    def load_memory(self, path: str):
        """Load memory."""
        if self._memory is not None:
            self._memory.load(path)
    
    def add_mcp_server(self, server: MCPServer) -> None:
        """
        Add an MCP server and load its tools.
        
        Args:
            server: MCPServer instance (will be connected automatically)
        """
        if not server.connected:
            server.connect()
        self._mcp_servers.append(server)
        
        # Reload all components with updated tools
        self._reload_components()
    
    def remove_mcp_server(self, name: str) -> None:
        """
        Remove an MCP server.
        
        Args:
            name: Name of the server to remove
        """
        for i, server in enumerate(self._mcp_servers):
            if server.name == name:
                server.disconnect()
                self._mcp_servers.pop(i)
                break
        
        # Reload all components with updated tools
        self._reload_components()
    
    def _reload_components(self) -> None:
        """Reload all components with current tools."""
        all_tools = self._collect_all_tools()
        self._all_tools = all_tools
        self._planner = Planner(self._model, all_tools, debug=self._debug)
        self._executor = Executor(all_tools)
        if self._react_fallback_enabled:
            self._react = ReactFallback(self._model, all_tools, debug=self._debug)
        if self._routing_enabled:
            self._router = Router(self._model, all_tools, debug=self._debug)
    
    def disconnect_mcp_servers(self) -> None:
        """Disconnect all MCP servers."""
        for server in self._mcp_servers:
            try:
                server.disconnect()
            except Exception:
                pass
    
    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {t.name: t for t in self._tools}
    
    @property
    def mcp_servers(self) -> List[MCPServer]:
        return self._mcp_servers
    
    @property
    def model_name(self) -> str:
        return self._model.model_name
    
    def export(self, format: str, output: str, **kwargs) -> str:
        """
        Export agent in the specified format.
        
        Args:
            format: Export format - "config", "docker", or "binary"
            output: Output path (file for config/binary, directory for docker)
            **kwargs: Format-specific options:
                - config: format="yaml"|"json", include_mcp_auth=False
                - docker: include_ollama=False, api_mode=True
                - binary: use_pyinstaller=False
                
        Returns:
            Path to exported artifact
            
        Examples:
            # Export as YAML config
            agent.export("config", "my_agent.yaml")
            
            # Export as Docker setup
            agent.export("docker", "./docker_agent/")
            
            # Export as Docker with bundled Ollama
            agent.export("docker", "./docker_agent/", include_ollama=True)
            
            # Export as standalone Python script
            agent.export("binary", "./my_agent.py")
        """
        from .packaging import export_agent
        return export_agent(self, format, output, **kwargs)
    
    def __repr__(self) -> str:
        return f"<Agent model={self.model_name} tools={len(self._tools)} mcp={len(self._mcp_servers)}>"
