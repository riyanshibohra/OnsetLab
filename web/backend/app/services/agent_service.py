"""Agent service - wraps OnsetLab SDK for web use.

Model-driven REWOO/ReAct agent:
1. Router (model-driven) classifies DIRECT vs TOOL
2. REWOO plans + executes for tool tasks
3. If REWOO fails → ReAct fallback with failure context
4. ReAct iteratively corrects (fixes wrong tool names, missing params, etc.)
"""

import re
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for importing onsetlab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from onsetlab.tools import (
    Calculator,
    DateTime,
    UnitConverter,
    TextProcessor,
    RandomGenerator,
    BaseTool,
)
from onsetlab.rewoo.planner import Planner
from onsetlab.rewoo.executor import Executor
from onsetlab.rewoo.solver import Solver
from onsetlab.rewoo.react_fallback import ReactFallback
from onsetlab.router import Router, Strategy
from onsetlab.skills import generate_tool_rules

from .model_service import GroqModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Write-guard: prevents duplicate writes in ReAct loops
# ---------------------------------------------------------------------------
# MCP tool name patterns that mutate state (create/update/delete).
_WRITE_PATTERNS = re.compile(
    r"(write|create|update|push|merge|delete|fork|assign|add_comment)",
    re.IGNORECASE,
)


class _WriteGuardTool:
    """
    Wraps a write/mutate tool so it can only succeed ONCE per ReAct run.

    After the first successful execution the guard caches the result and
    returns it on all subsequent calls, preventing the 3B model from
    creating 5 duplicate GitHub issues because it never says "Final Answer".
    """

    def __init__(self, tool: BaseTool):
        self._tool = tool
        self._succeeded = False
        self._cached = ""

    # ---- duck-type BaseTool ----
    @property
    def name(self):
        return self._tool.name

    @property
    def description(self):
        return self._tool.description

    @property
    def parameters(self):
        return self._tool.parameters

    # Forward _mcp_tool so _is_mcp_tool() still works
    def __getattr__(self, item):
        return getattr(self._tool, item)

    def execute(self, **kwargs):
        if self._succeeded:
            return f"(Already completed successfully) {self._cached}"
        result = self._tool.execute(**kwargs)
        result_str = str(result)
        # If not an error, lock the tool
        if not any(
            m in result_str.lower()
            for m in ["error:", "error -", "failed:", "exception:", "cannot "]
        ):
            self._succeeded = True
            self._cached = result_str[:500]
        return result


def _guard_write_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """Wrap write/mutate MCP tools with a one-shot guard."""
    guarded = []
    for t in tools:
        if _is_mcp_tool(t) and _WRITE_PATTERNS.search(t.name):
            guarded.append(_WriteGuardTool(t))
        else:
            guarded.append(t)
    return guarded

# Tool name to class mapping (all 6 built-in tools)
TOOL_MAP = {
    "Calculator": Calculator,
    "DateTime": DateTime,
    "UnitConverter": UnitConverter,
    "TextProcessor": TextProcessor,
    "RandomGenerator": RandomGenerator,
}

ALL_TOOLS = list(TOOL_MAP.keys())

# Max MCP tools to show to Planner/ReAct (small models can't handle 40+)
MAX_MCP_TOOLS_FOR_PLANNER = 8


# ---------------------------------------------------------------------------
# Tool relevance scoring & filtering
# ---------------------------------------------------------------------------

def _is_mcp_tool(tool: BaseTool) -> bool:
    """Check if a tool is an MCP tool (vs built-in)."""
    return hasattr(tool, '_mcp_tool')


def _expand_query_with_synonyms(query: str) -> str:
    """Expand query with synonyms for better tool matching."""
    synonyms = {
        "prs": "pull requests",
        "pr": "pull request",
        "repos": "repositories",
        "show": "list get",
        "find": "search",
        "look up": "search get",
        "check": "get",
        "who am i": "get me",
        "my profile": "get me",
        "my info": "get me",
        "star": "stars",
        "fork": "fork repository",
        "branch": "branches",
        "tag": "tags",
        "release": "releases",
        "commit": "commits",
        "diff": "get commit",
        "merge": "merge pull request",
    }
    expanded = query.lower()
    for abbr, full in synonyms.items():
        if abbr in expanded:
            expanded += " " + full
    return expanded


def _score_tool_relevance(query: str, tool: BaseTool) -> float:
    """Score how relevant an MCP tool is for a given query."""
    expanded = _expand_query_with_synonyms(query)
    query_words = set(re.findall(r'\b\w{3,}\b', expanded))
    score = 0.0

    # Tool name matching (highest weight)
    name_words = set(tool.name.lower().replace('_', ' ').split())
    name_overlap = query_words & name_words
    score += len(name_overlap) * 5.0

    # Substring match in name (e.g., "issue" in "list_issues")
    name_lower = tool.name.lower().replace('_', ' ')
    for qw in query_words:
        if len(qw) >= 4 and qw in name_lower:
            score += 3.0

    # Description matching (medium weight)
    desc_words = set(re.findall(r'\b\w{3,}\b', tool.description.lower()[:100]))
    desc_overlap = query_words & desc_words
    stop = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was'}
    desc_overlap -= stop
    score += len(desc_overlap) * 1.5

    return score


def _filter_tools_for_query(
    query: str,
    builtin_tools: List[BaseTool],
    mcp_tools: List[BaseTool],
    max_mcp: int = MAX_MCP_TOOLS_FOR_PLANNER,
) -> List[BaseTool]:
    """
    Select the most relevant MCP tools for a query.

    Always keeps all built-in tools. Scores MCP tools by relevance
    and returns the top `max_mcp` ones. This keeps the prompt
    small enough for a 3B model to handle.
    """
    if len(mcp_tools) <= max_mcp:
        return builtin_tools + mcp_tools

    scored = [(tool, _score_tool_relevance(query, tool)) for tool in mcp_tools]
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [tool for tool, _ in scored[:max_mcp]]

    logger.info(
        f"Filtered {len(mcp_tools)} MCP tools → {len(selected)} for query: "
        f"{[(t.name, f'{s:.1f}') for t, s in scored[:max_mcp]]}"
    )

    return builtin_tools + selected


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    """Trace of the agent pipeline for the UI."""
    router_decision: str = ""
    router_reason: str = ""
    tools_total: int = 0
    tools_filtered: int = 0
    tools_selected: List[str] = None
    tool_rules: str = ""
    planner_think: str = ""
    planner_prompt: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""

    def __post_init__(self):
        if self.tools_selected is None:
            self.tools_selected = []


@dataclass
class AgentResult:
    """Result from agent execution."""
    answer: str
    plan: List[Dict[str, Any]]
    results: Dict[str, str]
    strategy: str
    slm_calls: int
    trace: PipelineTrace = None
    parallel_waves: int = 0

    def __post_init__(self):
        if self.trace is None:
            self.trace = PipelineTrace()


# ---------------------------------------------------------------------------
# Agent Service
# ---------------------------------------------------------------------------

class AgentService:
    """
    Hybrid REWOO/ReAct agent service for web playground.

    Mirrors the SDK's Agent class architecture:
    1. Router picks REWOO / REACT / DIRECT
    2. REWOO runs first (plan → execute → solve)
    3. If REWOO fails → ReAct fallback with failure context
    4. ReAct iteratively corrects wrong tool names, params, etc.

    MCP-specific optimizations:
    - Tool filtering: Planner and ReAct only see top-N relevant MCP tools
    - Executor keeps ALL tools so any correctly-named call works
    """

    def __init__(
        self,
        model: GroqModel,
        tools: List[BaseTool],
        context: Optional[str] = None,
        has_mcp_tools: bool = False,
    ):
        self.model = model
        self.tools = tools
        self.context = context
        self._has_mcp_tools = has_mcp_tools

        # Split tools
        self._builtin_tools = [t for t in tools if not _is_mcp_tool(t)]
        self._mcp_tools = [t for t in tools if _is_mcp_tool(t)]

        # Executor always has ALL tools (so any valid tool name works)
        # parallel=True enables concurrent execution of independent steps
        self._executor = Executor(tools, parallel=True)
        self._solver = Solver(model)
        self._router = Router(model, tools)

        # Planner & ReAct are created per-query when MCP tools are present
        # (to filter to the most relevant subset)
        if not has_mcp_tools:
            self._planner = Planner(model, tools)
            self._react = ReactFallback(model, tools, max_iterations=2)
        else:
            self._planner = None  # Created per query
            self._react = None    # Created per query

        logger.info(
            f"AgentService init: {len(self._builtin_tools)} built-in, "
            f"{len(self._mcp_tools)} MCP tools, react_fallback=True"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentResult:
        """Run the agent on a query using the hybrid approach."""
        logger.info(f"=== QUERY: {query!r} ===")

        # Initialize pipeline trace
        self._trace = PipelineTrace(
            tools_total=len(self.tools),
        )

        # Step 1: Route — model decides DIRECT vs TOOL
        routing = self._router.route(query, self.context or "")
        self._trace.router_decision = routing.strategy.value.upper()
        self._trace.router_reason = routing.reason
        logger.info(
            f"Router: strategy={routing.strategy.value}, "
            f"confidence={routing.confidence:.0%}, "
            f"reason={routing.reason}"
        )

        # Step 2: Execute strategy
        if routing.strategy == Strategy.DIRECT:
            return self._execute_direct(query)
        else:
            return self._execute_rewoo(query)

    # ------------------------------------------------------------------
    # DIRECT strategy
    # ------------------------------------------------------------------

    def _execute_direct(self, query: str) -> AgentResult:
        """Handle direct questions (no tools needed)."""
        answer = self._handle_direct_response(query)
        logger.info(f"=== RESULT: strategy=direct, steps=0, slm_calls=1 ===")
        return AgentResult(
            answer=answer, plan=[], results={},
            strategy="direct", slm_calls=1,
            trace=self._trace,
        )

    def _handle_direct_response(self, query: str) -> str:
        """Generate a direct response without tools."""
        # Build tool awareness context so the model knows its capabilities
        tool_names = [t.name for t in self.tools]
        tool_list = ", ".join(tool_names) if tool_names else "none"

        context_parts = []
        if self.context:
            context_parts.append(f"Conversation:\n{self.context}")

        context_str = "\n\n".join(context_parts)

        prompt = (
            f"You are an AI assistant with access to these tools: {tool_list}\n"
            f"{context_str}\n\n"
            f"Answer this question directly and concisely:\n{query}\n\nAnswer:"
        )
        return self.model.generate(prompt, max_tokens=256)

    # ------------------------------------------------------------------
    # REWOO strategy (with ReAct fallback)
    # ------------------------------------------------------------------

    def _execute_rewoo(self, query: str) -> AgentResult:
        """
        Run REWOO strategy. If it fails, fall back to ReAct.

        Failure = empty plan OR all steps errored.
        """
        slm_calls = 0

        # Get query-filtered Planner & record tool filtering in trace
        planner, filtered_tools = self._get_planner_with_trace(query)

        # Record tool rules in trace
        tool_rules = generate_tool_rules(filtered_tools)
        self._trace.tool_rules = tool_rules

        # Plan with reasoning (THINK → PLAN)
        plan_result = planner.plan_with_reasoning(query, self.context)
        plan = plan_result.steps
        self._trace.planner_think = plan_result.think
        slm_calls += 1
        logger.info(f"REWOO think: {plan_result.think}")
        logger.info(f"REWOO plan: {plan}")

        # Execute if we have a plan
        results: Dict[str, str] = {}
        if plan:
            results = self._executor.execute(plan)
            logger.info(f"REWOO execution: {list(results.keys())}")
            for k, v in results.items():
                v_str = str(v)
                logger.info(f"  {k}: {'ERROR' if v_str.startswith('Error') else 'OK'} ({len(v_str)} chars)")

        # Check if REWOO failed
        if self._check_rewoo_failed(plan, results):
            # ---- Attempt 1: Rescue missing params from query ----
            if self._rescue_missing_params(query, plan, results):
                logger.info("Params rescued → re-executing plan")
                results = self._executor.execute(plan)

                # If re-execution succeeds, solve normally
                if not self._check_rewoo_failed(plan, results):
                    plan_steps = self._format_plan_steps_from_rewoo(plan, results)
                    answer = self._solver.solve(
                        query, plan, results, self.context
                    )
                    slm_calls += 1
                    logger.info(
                        f"=== RESULT: strategy=rewoo (rescued), "
                        f"steps={len(plan_steps)}, slm_calls={slm_calls} ==="
                    )
                    return AgentResult(
                        answer=answer, plan=plan_steps,
                        results={k: str(v) for k, v in results.items()},
                        strategy="rewoo", slm_calls=slm_calls,
                        trace=self._trace,
                    )
                logger.info("Re-execution after rescue still failed")

            # ---- Attempt 2: ReAct fallback ----
            logger.info("REWOO failed → falling back to ReAct (1 iter)")
            self._trace.fallback_used = True
            self._trace.fallback_reason = self._summarize_errors(results)
            failure_context = self._build_failure_context(plan, results)
            react_result = self._run_react_with_solver(
                query, failure_context, "rewoo->react",
                max_iterations=1,
            )
            react_result.slm_calls += slm_calls
            react_result.trace = self._trace
            return react_result

        # REWOO succeeded — solve
        plan_steps = self._format_plan_steps_from_rewoo(plan, results)

        answer = self._solver.solve(query, plan, results, self.context)
        slm_calls += 1

        results_str = {k: str(v) for k, v in results.items()}

        logger.info(
            f"=== RESULT: strategy=rewoo, "
            f"steps={len(plan_steps)}, slm_calls={slm_calls} ==="
        )

        return AgentResult(
            answer=answer, plan=plan_steps, results=results_str,
            strategy="rewoo", slm_calls=slm_calls,
            trace=self._trace,
        )

    # ------------------------------------------------------------------
    # ReAct runner (used as REWOO fallback)
    # ------------------------------------------------------------------

    def _run_react_with_solver(
        self,
        query: str,
        context: Optional[str],
        strategy_label: str,
        max_iterations: int = 2,
    ) -> AgentResult:
        """
        Run ReAct and use Solver on results.

        Small models (3B) often don't produce "Final Answer:" in the
        ReAct loop, so they hit max_iterations even when the tool call
        succeeded.  We detect this and use the Solver to synthesize.
        """
        react = self._get_react(query, max_iterations=max_iterations)
        answer, steps, results = react.run(query, context)
        slm_calls = len(steps) + 1

        # Check if ReAct hit max iterations but has successful results
        success_results = {
            k: v for k, v in results.items()
            if not self._is_error_result(str(v))
        }
        if success_results and "couldn't complete" in answer.lower():
            logger.info(
                f"ReAct hit max iterations but has {len(success_results)} "
                f"successful result(s) — using Solver"
            )
            # Build a plan-like structure for the Solver
            solver_plan = [
                s for s in steps
                if not self._is_error_result(str(s.get("result", "")))
            ]
            answer = self._solver.solve(query, solver_plan, success_results, self.context)
            slm_calls += 1

        plan_steps = self._format_plan_steps(steps)
        results_str = {k: str(v) for k, v in results.items()}

        logger.info(
            f"=== RESULT: strategy={strategy_label}, "
            f"steps={len(plan_steps)}, slm_calls={slm_calls} ==="
        )

        return AgentResult(
            answer=answer, plan=plan_steps, results=results_str,
            strategy=strategy_label, slm_calls=slm_calls,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_planner(self, query: str) -> Planner:
        """Get a Planner with filtered tools for this query."""
        if self._planner is not None:
            return self._planner
        filtered = _filter_tools_for_query(
            query, self._builtin_tools, self._mcp_tools
        )
        return Planner(self.model, filtered)

    def _get_planner_with_trace(self, query: str) -> Tuple[Planner, List[BaseTool]]:
        """Get Planner + record tool filtering in trace."""
        if self._planner is not None:
            tools = list(self._planner.tools.values())
            self._trace.tools_filtered = len(tools)
            self._trace.tools_selected = [t.name for t in tools[:10]]
            return self._planner, tools

        filtered = _filter_tools_for_query(
            query, self._builtin_tools, self._mcp_tools
        )
        self._trace.tools_filtered = len(filtered)
        self._trace.tools_selected = [t.name for t in filtered[:10]]
        return Planner(self.model, filtered), filtered

    @staticmethod
    def _summarize_errors(results: Dict[str, str]) -> str:
        """Summarize error results for trace."""
        errors = []
        for k, v in results.items():
            v_str = str(v)
            if v_str.startswith("Error"):
                errors.append(f"{k}: {v_str[:80]}")
        return "; ".join(errors) if errors else "Empty plan"

    def _get_react(
        self, query: str, max_iterations: int = 2,
    ) -> ReactFallback:
        """Get a ReactFallback with filtered + write-guarded tools."""
        if self._react is not None:
            return self._react
        filtered = _filter_tools_for_query(
            query, self._builtin_tools, self._mcp_tools
        )
        guarded = _guard_write_tools(filtered)
        return ReactFallback(
            self.model, guarded, max_iterations=max_iterations,
        )

    def _check_rewoo_failed(
        self, plan: List[Dict], results: Dict[str, str]
    ) -> bool:
        """
        Check if REWOO failed and should fall back to ReAct.

        Returns True if:
        - No plan was generated
        - All plan steps have validation errors
        - All execution results are errors
        """
        if not plan:
            return True

        # All steps have validation errors
        if all("error" in step for step in plan):
            return True

        # All results are errors
        if results:
            all_errors = all(
                self._is_error_result(str(v)) for v in results.values()
            )
            if all_errors:
                return True

        return False

    @staticmethod
    def _is_error_result(result: str) -> bool:
        """Check if a result looks like an error."""
        result_lower = result.lower()
        error_markers = [
            "error:", "error -", "failed:", "exception:",
            "cannot ", "could not ", "unable to ",
            "not found", "permission denied",
            "missing required", "missing param", "invalid ",
            "unauthorized", "forbidden", "bad request",
            "timed out", "timeout",
        ]
        return any(m in result_lower for m in error_markers)

    @staticmethod
    def _extract_missing_param_info(
        results: Dict[str, str],
    ) -> Optional[Tuple[str, List[str], str]]:
        """
        Check if failure is about missing required parameters.

        Returns (tool_name, [missing_param_names], step_id) if so, else None.
        """
        for step_id, result_str in results.items():
            r = str(result_str).lower()
            if "missing required param" in r or "missing param" in r:
                tool_match = re.search(r"tool '(\w+)'", r)
                param_match = re.search(
                    r"missing required param(?:eter)?s?:?\s*(.+)",
                    r, re.IGNORECASE
                )
                tool_name = tool_match.group(1) if tool_match else "the tool"
                if param_match:
                    raw = param_match.group(1).strip("[] '\"")
                    params = [p.strip("' \"") for p in raw.split(",")]
                    return (tool_name, params, step_id)
                return (tool_name, ["required parameter"], step_id)
        return None

    @staticmethod
    def _extract_param_value(
        query: str, param_name: str, context: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Extract a parameter value from a natural language query.

        Handles:
          - title: titled "X", called "X", named "X", quoted strings
          - body: body "X", saying "X", description "X"
          - owner/repo: owner/repo pattern
          - method: infer from verbs (create/update/list/delete)
          - issue_number/pullNumber: #N patterns
          - query: search terms
        """
        q = query
        ql = query.lower()

        if param_name == "title":
            for pat in [
                r'titled\s+["\']([^"\']+)["\']',
                r'titled\s+(\S+)',
                r'called\s+["\']([^"\']+)["\']',
                r'named\s+["\']([^"\']+)["\']',
                r'title\s*[:=]\s*["\']([^"\']+)["\']',
            ]:
                m = re.search(pat, q, re.IGNORECASE)
                if m:
                    return m.group(1)
            # Fallback: last quoted string
            quoted = re.findall(r'["\']([^"\']+)["\']', q)
            if quoted:
                return quoted[-1]

        elif param_name == "body":
            for pat in [
                r'body\s+["\']([^"\']+)["\']',
                r'with body\s+["\']([^"\']+)["\']',
                r'saying\s+["\']([^"\']+)["\']',
                r'description\s+["\']([^"\']+)["\']',
            ]:
                m = re.search(pat, q, re.IGNORECASE)
                if m:
                    return m.group(1)

        elif param_name == "owner":
            m = re.search(r'(\w[\w-]*)/(\w[\w.-]+)', q)
            if m:
                return m.group(1)
            # Try context
            if context:
                m = re.search(r'(\w[\w-]*)/(\w[\w.-]+)', context)
                if m:
                    return m.group(1)

        elif param_name == "repo":
            m = re.search(r'(\w[\w-]*)/(\w[\w.-]+)', q)
            if m:
                return m.group(2)
            if context:
                m = re.search(r'(\w[\w-]*)/(\w[\w.-]+)', context)
                if m:
                    return m.group(2)

        elif param_name == "method":
            if any(w in ql for w in ["create", "new", "add", "make", "open"]):
                return "create"
            elif any(w in ql for w in ["update", "edit", "modify", "change"]):
                return "update"
            elif any(w in ql for w in ["close", "closed"]):
                return "update"
            elif any(w in ql for w in ["list", "show", "get", "find", "fetch"]):
                return "get"

        elif param_name in ("issue_number", "pullNumber", "pull_number"):
            m = re.search(r'#(\d+)', q)
            if m:
                return int(m.group(1))
            m = re.search(r'(?:issue|pr|pull request)\s+(\d+)', q, re.IGNORECASE)
            if m:
                return int(m.group(1))

        elif param_name == "query":
            quoted = re.findall(r'["\']([^"\']+)["\']', q)
            if quoted:
                return quoted[0]
            for pat in [r'search (?:for )?(.+)', r'find (.+)', r'look (?:up |for )(.+)']:
                m = re.search(pat, q, re.IGNORECASE)
                if m:
                    return m.group(1).strip()

        return None

    def _rescue_missing_params(
        self,
        query: str,
        plan: List[Dict],
        results: Dict[str, str],
    ) -> bool:
        """
        Try to extract missing parameter values from the query/context
        and patch the plan. Returns True if any params were rescued.
        """
        info = self._extract_missing_param_info(results)
        if not info:
            return False

        tool_name, missing_params, step_id = info

        # Find the matching step
        target_step = None
        for step in plan:
            if step.get("id") == step_id:
                target_step = step
                break
        if not target_step:
            return False

        rescued_any = False
        for param_name in missing_params:
            value = self._extract_param_value(query, param_name, self.context)
            if value is not None:
                target_step["params"][param_name] = value
                rescued_any = True
                logger.info(f"Rescued param {param_name}={value!r} from query")

        if rescued_any:
            # Check if all missing params were found
            still_missing = [
                p for p in missing_params if p not in target_step["params"]
            ]
            if not still_missing:
                # Clear the error so Executor will re-run this step
                target_step.pop("error", None)
                logger.info(
                    f"All missing params rescued for {tool_name} — "
                    f"will re-execute"
                )
            else:
                logger.info(
                    f"Rescued some params but still missing: {still_missing}"
                )

        return rescued_any

    def _build_failure_context(
        self, plan: List[Dict], results: Dict[str, str]
    ) -> str:
        """Build failure context so ReAct knows what REWOO got wrong."""
        if not plan or not results:
            return ""

        parts = []
        for step in plan:
            step_id = step.get("id", "?")
            tool = step.get("tool", "?")
            params = step.get("params", {})
            result = results.get(step_id, "no result")

            parts.append(
                f"PREVIOUS ATTEMPT FAILED - FIX IT:\n"
                f"Tool used: {tool}\n"
                f"Parameters: {params}\n"
                f"Error: {result}\n\n"
                f"Fix the parameters or use the correct tool name."
            )

        return "\n".join(parts)

    def _format_plan_steps_from_rewoo(
        self, plan: List[Dict], results: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Format REWOO plan + results into the response format."""
        steps = []
        for step in plan:
            step_id = step.get("id", "")
            result_str = str(results.get(step_id, ""))
            steps.append({
                "id": step_id,
                "tool": step.get("tool", ""),
                "params": step.get("params", {}),
                "result": result_str,
                "status": "error" if self._is_error_result(result_str) else "done",
            })
        return steps

    def _format_plan_steps(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format ReAct steps into the response format."""
        formatted = []
        for step in steps:
            formatted.append({
                "id": step.get("id", ""),
                "tool": step.get("tool", ""),
                "params": step.get("params", {}),
                "result": step.get("result", ""),
                "status": "error" if "Error" in str(step.get("result", "")) else "done",
            })
        return formatted


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_agent(
    model_name: str,
    tool_names: List[str],
    context: Optional[str] = None,
    mcp_tools: Optional[List[BaseTool]] = None,
) -> AgentService:
    """Factory function to create an agent."""
    model = GroqModel(model_name)

    tools: List[BaseTool] = []
    for name in tool_names:
        tool_class = TOOL_MAP.get(name)
        if tool_class:
            tools.append(tool_class())

    if not tools:
        tools = [Calculator(), DateTime()]

    has_mcp = bool(mcp_tools)
    if mcp_tools:
        tools.extend(mcp_tools)

    return AgentService(model, tools, context, has_mcp_tools=has_mcp)
