"""
OnsetLab Skills — auto-generated tool-calling rules from tool schemas.

Instead of hardcoding rules per API (GitHub, Slack, ...), we parse the
tool's JSON schema (properties, required, types, enums) and generate
compact, SLM-friendly rules automatically.  This works for ANY MCP server.

Optional HINTS provide domain knowledge for well-known APIs.

Usage:
    from onsetlab.skills import generate_tool_rules

    rules = generate_tool_rules(tools)   # Auto-generated from schemas
    plan = planner.plan(query, rules)    # SLM gets focused instructions
"""

from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Optional hints for well-known APIs — augment, NOT replace, auto-generated
# ---------------------------------------------------------------------------

HINTS: Dict[str, str] = {
    "github": (
        "- Split owner/repo: owner=\"user\", repo=\"name\" (SEPARATE params)\n"
        "- To find the latest/most recent issue → use list_issues, NOT issue_read\n"
        "- issue_read & issue_write both need method as the FIRST param"
    ),
    "slack": (
        "- Channel IDs look like \"C01ABCDEF\" — use list_channels first to find them\n"
        "- For sending: channel and text are both REQUIRED"
    ),
    "notion": (
        "- Page/database IDs are UUIDs like \"abc123-def456\"\n"
        "- Use search first to find page IDs before creating/updating"
    ),
}


# ---------------------------------------------------------------------------
# Auto-generated tool rules from schemas
# ---------------------------------------------------------------------------

def generate_tool_rules(tools, max_tools: int = 12) -> str:
    """
    Generate compact tool-calling rules from actual tool schemas.

    Reads each tool's ``parameters`` property (which MCP tools populate
    from their JSON schema) and produces a compact, SLM-friendly string
    that tells the model exactly what parameters are required, their
    types, and any enum constraints.

    Args:
        tools:     List of BaseTool instances (built-in or MCP).
        max_tools: Cap the number of tools to keep the prompt short.

    Returns:
        A multi-line string ready to inject into the Planner prompt.
    """
    if not tools:
        return ""

    lines: List[str] = []

    for tool in tools[:max_tools]:
        params = _get_params(tool)
        if not params:
            lines.append(f"- {tool.name}() — {_short_desc(tool)}")
            continue

        req_parts: List[str] = []
        opt_parts: List[str] = []

        for p_name, p_info in params.items():
            if not isinstance(p_info, dict):
                continue

            is_req = p_info.get("required", False)
            hint = _format_param_hint(p_name, p_info)

            if is_req:
                req_parts.append(hint)
            else:
                opt_parts.append(hint)

        # Build signature: show required params inline
        sig = ", ".join(req_parts) if req_parts else ""
        line = f"- {tool.name}({sig})"

        # Short description
        line += f" — {_short_desc(tool)}"

        # Mark required
        if req_parts:
            line += f"  [ALL REQUIRED]"

        lines.append(line)

    # Detect and append hints for known APIs
    hints = _detect_hints(tools)
    if hints:
        lines.append("")
        lines.append("Hints:")
        lines.append(hints)

    return "\n".join(lines)


def generate_examples(tools, max_examples: int = 3) -> str:
    """
    Generate concrete call examples from tool schemas.

    Args:
        tools:        List of BaseTool instances.
        max_examples: Maximum number of examples to generate.

    Returns:
        A compact examples string, e.g.:
        ``list_issues(owner="user", repo="myrepo")``
    """
    examples: List[str] = []

    for tool in tools[:max_examples]:
        params = _get_params(tool)
        if not params:
            examples.append(f"{tool.name}()")
            continue

        parts: List[str] = []
        for p_name, p_info in params.items():
            if not isinstance(p_info, dict):
                continue
            if not p_info.get("required", False):
                continue
            parts.append(_format_example_param(p_name, p_info))

        examples.append(f"{tool.name}({', '.join(parts)})")

    return ", ".join(examples) if examples else ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_params(tool) -> Dict[str, Any]:
    """Extract parameters dict from a tool, handling both schema styles."""
    raw = tool.parameters
    if "properties" in raw:
        # JSON Schema style → flatten to our format
        props = raw.get("properties", {})
        required = set(raw.get("required", []))
        out: Dict[str, Any] = {}
        for p_name, p_info in props.items():
            entry = dict(p_info)
            entry["required"] = p_name in required
            out[p_name] = entry
        return out
    return raw


def _format_param_hint(name: str, info: Dict[str, Any]) -> str:
    """Format a single parameter for the rules string."""
    p_type = info.get("type", "string")

    if "enum" in info:
        vals = info["enum"]
        if len(vals) <= 4:
            return f'{name}=' + "|".join(f'"{v}"' for v in vals)
        return f'{name}="{vals[0]}" (or {len(vals)-1} others)'

    if p_type == "integer":
        return f"{name}=N"
    if p_type == "number":
        return f"{name}=N"
    if p_type == "boolean":
        return f"{name}=true|false"
    if p_type == "array":
        return f"{name}=[...]"
    if p_type == "object":
        return f'{name}={{...}}'

    return f'{name}="..."'


def _format_example_param(name: str, info: Dict[str, Any]) -> str:
    """Format a single parameter for an example call."""
    if "enum" in info:
        return f'{name}="{info["enum"][0]}"'

    p_type = info.get("type", "string")
    if p_type == "integer":
        return f"{name}=1"
    if p_type == "number":
        return f"{name}=1.0"
    if p_type == "boolean":
        return f"{name}=true"
    if p_type == "array":
        return f'{name}=["item"]'

    return f'{name}="example"'


def _short_desc(tool) -> str:
    """Truncate a tool description to keep prompts short."""
    desc = tool.description
    if len(desc) > 60:
        return desc[:57] + "..."
    return desc


def _detect_hints(tools) -> str:
    """
    Check if any well-known API hints should be appended.

    Looks at tool names for patterns like 'issue', 'slack', 'notion'.
    """
    names_lower = " ".join(getattr(t, "name", "").lower() for t in tools)

    matched: List[str] = []
    for api_key, hint_text in HINTS.items():
        if api_key in names_lower:
            matched.append(hint_text)

    return "\n".join(matched)


# ---------------------------------------------------------------------------
# Legacy compat — thin wrappers so old imports don't break
# ---------------------------------------------------------------------------

def detect_skill(tools) -> Optional[str]:
    """Detect API type from tools (legacy compat)."""
    names = " ".join(getattr(t, "name", "").lower() for t in tools)
    for api_key in HINTS:
        if api_key in names:
            return api_key
    return None


def get_skill_context(skill_key: Optional[str]) -> str:
    """Get hint text for a skill key (legacy compat)."""
    if skill_key and skill_key in HINTS:
        return HINTS[skill_key]
    return ""


def get_skill_for_query(query: str, tools) -> Optional[str]:
    """Detect skill from query + tools (legacy compat)."""
    return detect_skill(tools)


# Public API
__all__ = [
    "generate_tool_rules",
    "generate_examples",
    "HINTS",
    # Legacy compat
    "detect_skill",
    "get_skill_context",
    "get_skill_for_query",
]
