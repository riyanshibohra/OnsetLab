"""
OnsetLab Agent - REWOO-based agent with tool calling.
"""

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
from .mcp.server import MCPServer


@dataclass
class AgentResult:
    """Result from agent execution."""
    answer: str
    plan: List[Dict[str, Any]]
    results: Dict[str, str]
    verified: bool
    slm_calls: int


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
        debug: bool = False,
    ):
        """Create an agent."""
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
        self._verifier = Verifier(self._model) if verify else None
        self._solver = Solver(self._model)
        
        # Setup memory - stores conversation with results
        self._memory = ConversationMemory(max_turns=10) if memory else None
        self._last_results: Dict[str, str] = {}  # Store last tool results
        
        # Settings
        self._verify = verify
        self._max_replans = max_replans
        self._debug = debug
    
    def _collect_all_tools(self) -> List[BaseTool]:
        """Collect all tools."""
        return list(self._tools)
    
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
    
    def run(self, query: str) -> AgentResult:
        """Run a query."""
        slm_calls = 0
        
        # Get context
        context = self._get_context()
        
        if self._debug:
            print(f"\n[DEBUG] Context:\n{context}\n")
        
        # Plan (1 SLM call)
        plan = self._planner.plan(query, context)
        slm_calls += 1
        
        # Execute if we have a plan
        if plan:
            results = self._executor.execute(plan)
            self._last_results = results  # Store for next turn
        else:
            results = {}
            self._last_results = {}
        
        # Verify (0-1 SLM calls) - skip for empty plans
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
            slm_calls=slm_calls
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
    
    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {t.name: t for t in self._tools}
    
    @property
    def model_name(self) -> str:
        return self._model.model_name
    
    def __repr__(self) -> str:
        return f"<Agent model={self.model_name} tools={len(self._tools)} mcp={len(self._mcp_servers)}>"
