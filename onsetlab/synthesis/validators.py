"""
Training Data Validators
=========================
Format validation (correct tool names, types) and semantic validation
(does the tool call actually match the user's query?). Consolidating
quality checks here makes it easy to add semantic checks—the #1
improvement for training data quality (Shaw Talebi, Microsoft research).
"""

import re
from typing import Any, Callable, Optional


# -----------------------------------------------------------------------------
# Placeholder / junk detection
# -----------------------------------------------------------------------------

def has_placeholder(text: str) -> bool:
    """Check for placeholder or template patterns that should not appear in training data."""
    patterns = [
        r'\{\{[^{}]+\}\}',        # {{date}}
        r'<[A-Z][A-Z_]+>',        # <NAME>
        r'\[[A-Z][A-Z_]*\]',      # [DATE]
        r'PLACEHOLDER',
        r'YOUR_\w+_HERE',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


# -----------------------------------------------------------------------------
# Format validation and param type fixing
# -----------------------------------------------------------------------------

def _get_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return tool.get('name', '')
    return getattr(tool, 'name', '')


def _get_params(tool: Any) -> dict:
    if isinstance(tool, dict):
        return tool.get('parameters', {})
    return getattr(tool, 'parameters', {})


def _get_required(tool: Any) -> list:
    if isinstance(tool, dict):
        return tool.get('required_params', [])
    return getattr(tool, 'required_params', [])


def fix_param_types(
    tool_name: str,
    params: dict,
    tools: list,
) -> None:
    """
    Coerce parameter values to correct types based on tool schema (in-place).
    Handles common LLM mistakes: string numbers, single string instead of array, etc.
    """
    tool_schema = None
    for t in tools:
        if _get_name(t) == tool_name:
            tool_schema = t
            break
    if not tool_schema:
        return

    schema_params = _get_params(tool_schema)
    required_params = set(_get_required(tool_schema))

    # Remove None for optional params
    for param_name in [k for k, v in params.items() if v is None and k not in required_params]:
        del params[param_name]

    for param_name, value in list(params.items()):
        if param_name not in schema_params:
            continue
        param_info = schema_params[param_name]
        if not isinstance(param_info, dict):
            continue
        expected_type = param_info.get('type', 'string')

        if value is None:
            if expected_type == 'array':
                params[param_name] = []
            elif expected_type in ('number', 'integer'):
                params[param_name] = 0
            elif expected_type == 'boolean':
                params[param_name] = False
            elif expected_type == 'string':
                params[param_name] = ""
            continue

        if expected_type == 'array':
            if isinstance(value, str):
                params[param_name] = [v.strip() for v in value.split(',')] if ',' in value else ([value] if value else [])
            elif not isinstance(value, list):
                params[param_name] = [value] if value else []

        elif expected_type in ('number', 'integer'):
            if isinstance(value, str):
                try:
                    params[param_name] = float(value) if '.' in value else int(value)
                except ValueError:
                    params[param_name] = 0
            elif isinstance(value, bool):
                params[param_name] = 1 if value else 0

        elif expected_type == 'boolean':
            if isinstance(value, str):
                params[param_name] = value.lower() in ('true', 'yes', '1')
            elif isinstance(value, (int, float)):
                params[param_name] = bool(value)

        elif expected_type == 'string' and not isinstance(value, str) and value is not None:
            params[param_name] = str(value)

        enum_values = param_info.get('enum')
        if enum_values and isinstance(params.get(param_name), str):
            enum_map = {str(v).lower(): v for v in enum_values}
            if value.lower() in enum_map:
                params[param_name] = enum_map[value.lower()]


def validate_format(
    ex: dict,
    tool_names: set,
    tool_map: dict,
    tools: list,
) -> tuple[bool, dict]:
    """
    Validate format: real tool name, valid structure, fix param types.
    Returns (is_valid, example). Modifies ex in place (tool name normalization, param fixes).
    """
    if has_placeholder(str(ex)):
        return False, ex

    # Edge cases without tool (clarification, refusal, casual) — only need response
    if ex.get('tool') is None and ex.get('response'):
        return True, ex

    tool = ex.get('tool')
    if not tool:
        return True, ex

    actual_tool = None
    if tool in tool_names:
        actual_tool = tool
    else:
        normalized = tool.lower().replace("-", "_")
        if normalized in tool_map:
            ex['tool'] = tool_map[normalized]
            actual_tool = tool_map[normalized]

    if not actual_tool:
        return False, ex

    params = ex.get('parameters', ex.get('params', {}))
    if params:
        fix_param_types(actual_tool, params, tools)
        if 'params' in ex and 'parameters' not in ex:
            ex['parameters'] = ex.pop('params', {})

    return True, ex


# -----------------------------------------------------------------------------
# Semantic validation (tool call matches user intent + param values from query)
# -----------------------------------------------------------------------------

# Action verbs: query intent → must match tool name/description
_ACTION_VERBS = {
    "create": ["create", "add", "new", "open", "write", "post", "make"],
    "list": ["list", "show", "get all", "find", "search", "fetch", "see", "what"],
    "read": ["get", "read", "fetch", "retrieve", "show me", "details", "recall"],
    "update": ["update", "edit", "change", "modify"],
    "delete": ["delete", "remove"],
    "send": ["send", "post", "message", "notify", "tell"],
    "comment": ["comment", "reply", "respond"],
    "memory": ["remember", "save", "recall", "stored", "memory", "favorite"],
}
# Params whose values are system/enum (not user-provided from query)
_SYSTEM_PARAMS = frozenset(["id", "status", "state", "type", "limit", "page"])


def _get_query_actions(query_lower: str) -> list:
    """Return all canonical actions whose keywords appear in query (e.g. 'list open issues' -> ['list', 'create'])."""
    return [action for action, keywords in _ACTION_VERBS.items() if any(kw in query_lower for kw in keywords)]


def _action_matches_tool(query_action: str, tool_lower: str, tool_desc: str) -> bool:
    """True if tool name or description reflects the same action as the query."""
    combined = f"{tool_lower} {tool_desc}"
    return query_action in combined


def _param_values_from_query(query_lower: str, parameters: dict) -> bool:
    """
    Require that substantive string param values appear in the query (anti-hallucination).
    Skips system params (id, status, state, type, limit, page).
    """
    for param_name, param_value in parameters.items():
        if param_name in _SYSTEM_PARAMS:
            continue
        if not isinstance(param_value, str) or len(param_value) <= 3:
            continue
        val_lower = param_value.lower()
        # Value should appear in query (case-insensitive)
        if " " in param_value or "/" in param_value:
            # e.g. "facebook/react" or "Deploy complete" — require each part in query
            parts = re.split(r"[\s/]+", param_value)
            if any(len(p) > 2 and p.lower() not in query_lower for p in parts):
                return False  # REJECT: hallucinated value
        elif val_lower not in query_lower:
            return False  # REJECT: model hallucinated this value
    return True


def validate_semantic_match(
    query: str,
    tool_name: str,
    parameters: dict,
    tools: list,
) -> bool:
    """
    Check if the chosen tool and params make sense for the user query.
    Requires BOTH (1) word overlap / intent alignment and (2) param values from query.

    - If query has a clear action verb (create, list, send, ...), tool must have the SAME action.
    - Otherwise allow only if there is word overlap.
    - Substantive string param values must appear in the query (catches hallucinated values).
    """
    if not query or not tool_name:
        return True  # Skip semantic check for edge cases

    query_lower = query.lower().strip()
    tool_lower = tool_name.lower().replace("-", "_").replace(" ", "_")

    tool_desc = ""
    for t in tools:
        if _get_name(t) == tool_name:
            tool_desc = (t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")).lower()
            break

    query_words = set(re.findall(r"\b[a-z]{2,}\b", query_lower))
    combined = f"{tool_lower} {tool_desc}"
    combined_words = set(re.findall(r"\b[a-z]{2,}\b", combined))
    overlap = query_words & combined_words

    # 1) Intent alignment: if query has clear action verbs, at least one must match the tool
    query_actions = _get_query_actions(query_lower)
    if query_actions:
        if not any(_action_matches_tool(a, tool_lower, tool_desc) for a in query_actions):
            return False  # REJECT: wrong action (e.g. "send message" vs issue_create)
        # Action matches; still require some word overlap
        if len(overlap) == 0:
            return False
    else:
        # No clear action verb: allow only if there's word overlap
        if len(overlap) == 0:
            return False

    # 2) Param values must come from query (no hallucinated values)
    if not _param_values_from_query(query_lower, parameters or {}):
        return False

    return True


def validate_example(
    ex: dict,
    tool_names: set,
    tool_map: dict,
    tools: list,
    *,
    semantic: bool = True,
) -> bool:
    """
    Full validation: format + optional semantic check.
    Returns True if the example should be kept. Modifies ex in place (normalization, param fixes).
    """
    valid, ex = validate_format(ex, tool_names, tool_map, tools)
    if not valid:
        return False

    # Semantic: only for tool-call examples
    if semantic and ex.get('tool') and ex.get('query'):
        if not validate_semantic_match(
            ex.get("query", ""),
            ex.get("tool", ""),
            ex.get("parameters", ex.get("params", {})),
            tools,
        ):
            return False

    return True
