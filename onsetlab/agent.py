"""
OnsetLab Agent - REWOO-based agent with tool calling.
"""

import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
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


# Patterns for casual messages that don't need tools
CASUAL_PATTERNS = [
    r'^(hi|hey|hello|yo|sup|hiya|howdy)[\s!.?]*$',
    r'^(bye|goodbye|see ya|later|cya)[\s!.?]*$',
    r'^(thanks|thank you|thx|ty)[\s!.?]*$',
    r'^(ok|okay|sure|got it|understood)[\s!.?]*$',
    r'^(yes|no|yeah|nope|yep)[\s!.?]*$',
    r'^(good|great|nice|cool|awesome)[\s!.?]*$',
    r"^(how are you|how's it going|what's up)[\s!.?]*$",
]


@dataclass
class AgentResult:
    """Result from agent execution."""
    answer: str
    plan: List[Dict[str, Any]]
    results: Dict[str, str]
    verified: bool
    slm_calls: int
    used_react_fallback: bool = False


class Agent:
    """
    OnsetLab Agent - REWOO-based agent for reliable tool calling.
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
        debug: bool = False,
    ):
        """Create an agent."""
        # Settings (set early for use in other methods)
        self._verify = verify
        self._max_replans = max_replans
        self._react_fallback_enabled = react_fallback
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
        self._planner = Planner(self._model, all_tools, debug=debug)
        self._executor = Executor(all_tools)
        self._verifier = Verifier(self._model, debug=debug) if verify else None
        self._solver = Solver(self._model)
        self._react = ReactFallback(self._model, all_tools, debug=debug) if react_fallback else None
        
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
    
    def _is_casual(self, query: str) -> bool:
        """Check if query is a casual message that doesn't need tools."""
        query_lower = query.lower().strip()
        for pattern in CASUAL_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        return False
    
    def _get_system_context(self) -> str:
        """Get system context with current datetime."""
        now = datetime.now()
        return f"Current datetime: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
    
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
        """Run a query."""
        slm_calls = 0
        used_react = False
        
        # Get context
        context = self._get_context()
        
        if self._debug:
            print(f"\n[DEBUG] Context:\n{context}\n")
        
        # Handle casual messages directly (skip planner)
        if self._is_casual(query):
            if self._debug:
                print(f"[DEBUG] Detected casual message, skipping planner")
            answer = self._solver.solve(query, [], {}, context)
            slm_calls = 1
            
            if self._memory is not None:
                self._memory.add_user_message(query)
                self._memory.add_assistant_message(answer)
            
            return AgentResult(
                answer=answer,
                plan=[],
                results={},
                verified=True,
                slm_calls=slm_calls
            )
        
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
            self._last_results = results  # Store for next turn
        else:
            results = {}
            self._last_results = {}
        
        # Check if REWOO failed - fallback to ReAct if enabled
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
            slm_calls += len(react_steps) + 1  # +1 for potential final answer
            
            # Convert ReAct steps to plan format
            plan = react_steps
            results = react_results
            self._last_results = results
            
            # Update memory
            if self._memory is not None:
                self._memory.add_user_message(query)
                if results:
                    result_summary = ", ".join([f"{k}={v[:50]}" for k, v in results.items()])
                    self._memory.add_assistant_message(f"{answer} [Results: {result_summary}]")
                else:
                    self._memory.add_assistant_message(answer)
            
            return AgentResult(
                answer=answer,
                plan=plan,
                results=results,
                verified=True,  # ReAct is self-correcting
                slm_calls=slm_calls,
                used_react_fallback=True
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
        
        # Update memory (use 'is not None' because empty memory is falsy)
        if self._memory is not None:
            self._memory.add_user_message(query)
            # Include the result in the assistant message for memory
            if results:
                result_summary = ", ".join([f"{k}={v}" for k, v in results.items()])
                self._memory.add_assistant_message(f"{answer} [Results: {result_summary}]")
            else:
                self._memory.add_assistant_message(answer)
        
        return AgentResult(
            answer=answer,
            plan=plan,
            results=results,
            verified=verified,
            slm_calls=slm_calls,
            used_react_fallback=used_react
        )
    
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
        all_tools = self._collect_all_tools()
        self._planner = Planner(self._model, all_tools, debug=self._debug)
        self._executor = Executor(all_tools)
        if self._react_fallback_enabled:
            self._react = ReactFallback(self._model, all_tools, debug=self._debug)
    
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
        all_tools = self._collect_all_tools()
        self._planner = Planner(self._model, all_tools, debug=self._debug)
        self._executor = Executor(all_tools)
        if self._react_fallback_enabled:
            self._react = ReactFallback(self._model, all_tools, debug=self._debug)
    
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
    
    def __repr__(self) -> str:
        return f"<Agent model={self.model_name} tools={len(self._tools)} mcp={len(self._mcp_servers)}>"
