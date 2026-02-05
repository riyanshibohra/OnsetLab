"""
OnsetLab Agent - REWOO-based agent with tool calling.

Plan once, execute fast, verify always.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

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
    
    Uses the REWOO (Reasoning Without Observation) strategy:
    1. Plan all tool calls upfront
    2. Execute tools (parallel where possible)
    3. Verify results
    4. Synthesize answer
    
    This results in 2-3 SLM calls vs 5-10 for ReAct agents.
    """
    
    def __init__(
        self,
        model: Union[str, BaseModel] = "phi3.5",
        tools: List[BaseTool] = None,
        mcp_servers: List[MCPServer] = None,
        memory: bool = True,
        verify: bool = True,
        max_replans: int = 2,
    ):
        """
        Create an agent.
        
        Args:
            model: Model name (str) or BaseModel instance.
                   String format: "phi3.5", "qwen2.5:3b", "llama3.2:3b"
            tools: List of built-in tools.
            mcp_servers: List of MCP server configs.
            memory: Enable conversation memory.
            verify: Enable verification step.
            max_replans: Max replanning attempts on failure.
        """
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
        self._planner = Planner(self._model, all_tools)
        self._executor = Executor(all_tools)
        self._verifier = Verifier(self._model) if verify else None
        self._solver = Solver(self._model)
        
        # Setup memory
        self._memory = ConversationMemory() if memory else None
        
        # Settings
        self._verify = verify
        self._max_replans = max_replans
    
    def _collect_all_tools(self) -> List[BaseTool]:
        """Collect all tools from built-in and MCP servers."""
        all_tools = list(self._tools)
        # TODO: Add MCP tools when implemented
        return all_tools
    
    def run(self, query: str) -> AgentResult:
        """
        Run a query and return the result.
        
        Args:
            query: User query/task.
            
        Returns:
            AgentResult with answer, plan, results, etc.
        """
        slm_calls = 0
        
        # Get conversation context
        context = None
        if self._memory and len(self._memory) > 0:
            context = self._memory.get_context_string()
        
        # Add user message to memory
        if self._memory:
            self._memory.add_user_message(query)
        
        # Plan
        plan = self._planner.plan(query, context)
        slm_calls += 1
        
        # Handle case where no tools needed
        if not plan:
            answer = self._solver.solve(query, [], {}, context)
            slm_calls += 1
            if self._memory:
                self._memory.add_assistant_message(answer)
            return AgentResult(
                answer=answer,
                plan=[],
                results={},
                verified=True,
                slm_calls=slm_calls
            )
        
        # Execute
        results = self._executor.execute(plan)
        
        # Verify (with replanning)
        verified = True
        replan_count = 0
        
        while self._verify and replan_count < self._max_replans:
            is_valid, reason = self._verifier.verify(query, plan, results)
            slm_calls += 1
            
            if is_valid:
                break
            
            # Replan with error context
            replan_context = f"Previous attempt failed: {reason}"
            if context:
                replan_context = f"{context}\n\n{replan_context}"
            
            plan = self._planner.plan(query, replan_context)
            slm_calls += 1
            results = self._executor.execute(plan)
            replan_count += 1
        
        if replan_count >= self._max_replans:
            verified = False
        
        # Solve
        answer = self._solver.solve(query, plan, results, context)
        slm_calls += 1
        
        # Add to memory
        if self._memory:
            self._memory.add_assistant_message(answer)
            for step_id, result in results.items():
                tool_name = next(
                    (s["tool"] for s in plan if s["id"] == step_id),
                    "unknown"
                )
                self._memory.add_tool_result(tool_name, result)
        
        return AgentResult(
            answer=answer,
            plan=plan,
            results=results,
            verified=verified,
            slm_calls=slm_calls
        )
    
    def chat(self, message: str) -> str:
        """
        Interactive chat - returns just the answer string.
        
        Args:
            message: User message.
            
        Returns:
            Assistant response.
        """
        result = self.run(message)
        return result.answer
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self._memory:
            self._memory.clear()
    
    def save_memory(self, path: str):
        """Save conversation memory to file."""
        if self._memory:
            self._memory.save(path)
    
    def load_memory(self, path: str):
        """Load conversation memory from file."""
        if self._memory:
            self._memory.load(path)
    
    @property
    def tools(self) -> Dict[str, BaseTool]:
        """Get all available tools."""
        return {t.name: t for t in self._tools}
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model.model_name
    
    def __repr__(self) -> str:
        tool_count = len(self._tools)
        mcp_count = len(self._mcp_servers)
        return f"<Agent model={self.model_name} tools={tool_count} mcp={mcp_count}>"
