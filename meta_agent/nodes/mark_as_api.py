"""
Mark as API Node
================
Marks a service for API implementation when no good MCP server exists.
Extracts detailed API endpoint information for tools.py generation.
"""

import json
import anthropic

from meta_agent.state import MetaAgentState, APIServerFallback, APIEndpoint
from meta_agent.tools.tavily_search import tavily_search


API_DISCOVERY_PROMPT = """You are an expert at analyzing REST API documentation and extracting endpoint details.

Given a service name and search results, provide detailed API information that can be used to generate Python API client code.

For each endpoint, provide:
1. A function name (snake_case, descriptive)
2. HTTP method (GET, POST, PUT, DELETE, PATCH)
3. URL path (with {placeholders} for path parameters)
4. Description of what it does
5. Parameters (query params, path params)
6. Request body schema (for POST/PUT/PATCH)

Common authentication patterns:
- bearer: "Authorization: Bearer {token}"
- api_key: "X-API-Key: {key}" or as query param
- basic: "Authorization: Basic {base64(user:pass)}"

Respond with a JSON object in this exact format:
{
    "base_url": "https://api.example.com/v1",
    "api_docs_url": "https://docs.example.com/api",
    "auth_type": "bearer" | "api_key" | "basic" | "oauth",
    "auth_header": "Authorization: Bearer {token}",
    "env_var": "SERVICE_API_KEY",
    "endpoints": [
        {
            "name": "list_items",
            "method": "GET",
            "path": "/items",
            "description": "List all items with optional filtering",
            "parameters": {
                "limit": {"type": "integer", "description": "Max results to return"},
                "offset": {"type": "integer", "description": "Pagination offset"},
                "status": {"type": "string", "description": "Filter by status"}
            },
            "required_params": [],
            "request_body": null,
            "response_schema": {
                "type": "array",
                "items": {"type": "object"}
            }
        },
        {
            "name": "create_item",
            "method": "POST",
            "path": "/items",
            "description": "Create a new item",
            "parameters": {},
            "required_params": [],
            "request_body": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Item name"},
                    "value": {"type": "number", "description": "Item value"}
                },
                "required": ["name"]
            },
            "response_schema": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}}
            }
        },
        {
            "name": "get_item",
            "method": "GET",
            "path": "/items/{item_id}",
            "description": "Get a specific item by ID",
            "parameters": {
                "item_id": {"type": "string", "description": "The item ID"}
            },
            "required_params": ["item_id"],
            "request_body": null,
            "response_schema": {"type": "object"}
        }
    ]
}

Be thorough - include the most commonly used endpoints for the service.
Use real API patterns for well-known services (Slack, Twilio, SendGrid, etc.)."""


def mark_as_api(state: MetaAgentState) -> dict:
    """
    Mark current service for API implementation with detailed endpoint info.
    
    When no good MCP server is found:
    1. Search for official API documentation
    2. Extract detailed endpoint information
    3. Create tool-like schemas for API endpoints
    4. Add to api_servers list
    5. Increment current_service_index
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with new api_servers entry containing detailed endpoints
    """
    current_service = state.get("current_service")
    api_key = state["anthropic_api_key"]
    tavily_api_key = state["tavily_api_key"]
    current_index = state["current_service_index"]
    
    # Get reason from evaluation
    reasoning = state.get("_eval_reasoning", "No suitable MCP server found")
    
    print(f"\n🔧 Marking for API implementation: {current_service}")
    print(f"   Reason: {reasoning}")
    
    # Search for API documentation
    service_name = current_service.replace("_", " ").title()
    
    try:
        # Search for official API docs
        search_results = tavily_search(
            f"{service_name} REST API documentation endpoints reference",
            api_key=tavily_api_key,
            max_results=5
        )
        
        # Use LLM to extract detailed API info
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"""{API_DISCOVERY_PROMPT}

Service: {service_name}

Search Results:
{search_results}

Extract detailed API endpoint information for this service.
Focus on the most commonly used endpoints for typical integrations.
Respond with JSON."""}
            ],
        )
        
        response_text = response.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        
        # Extract values
        base_url = result.get("base_url", f"https://api.{current_service.replace('_', '')}.com/v1")
        api_docs_url = result.get("api_docs_url")
        auth_type = result.get("auth_type", "bearer")
        auth_header = result.get("auth_header", "Authorization: Bearer {token}")
        env_var = result.get("env_var", f"{current_service.upper()}_API_KEY")
        
        # Parse endpoints
        endpoints: list[APIEndpoint] = []
        for ep in result.get("endpoints", []):
            endpoint: APIEndpoint = {
                "name": ep.get("name", "unknown"),
                "method": ep.get("method", "GET"),
                "path": ep.get("path", "/"),
                "description": ep.get("description", ""),
                "parameters": ep.get("parameters", {}),
                "required_params": ep.get("required_params", []),
                "request_body": ep.get("request_body"),
                "response_schema": ep.get("response_schema"),
            }
            endpoints.append(endpoint)
        
        print(f"   ✅ Found API docs: {api_docs_url}")
        print(f"   🔗 Base URL: {base_url}")
        print(f"   🔐 Auth: {auth_type}")
        print(f"   📋 Endpoints discovered: {len(endpoints)}")
        
        for ep in endpoints[:5]:
            print(f"      • {ep['method']} {ep['path']}: {ep['name']}")
        if len(endpoints) > 5:
            print(f"      ... and {len(endpoints) - 5} more")
                
    except Exception as e:
        print(f"   ⚠️ Could not extract detailed API info: {e}")
        
        # Create fallback with basic info
        base_url = f"https://api.{current_service.replace('_', '')}.com/v1"
        api_docs_url = None
        auth_type = "bearer"
        auth_header = "Authorization: Bearer {token}"
        env_var = f"{current_service.upper()}_API_KEY"
        
        # Create basic CRUD endpoints as fallback
        endpoints = [
            {
                "name": f"list_{current_service}s",
                "method": "GET",
                "path": f"/{current_service}s",
                "description": f"List all {current_service.replace('_', ' ')}s",
                "parameters": {
                    "limit": {"type": "integer", "description": "Max results"},
                    "offset": {"type": "integer", "description": "Pagination offset"}
                },
                "required_params": [],
                "request_body": None,
                "response_schema": {"type": "array"},
            },
            {
                "name": f"get_{current_service}",
                "method": "GET",
                "path": f"/{current_service}s/{{id}}",
                "description": f"Get a specific {current_service.replace('_', ' ')} by ID",
                "parameters": {
                    "id": {"type": "string", "description": "Resource ID"}
                },
                "required_params": ["id"],
                "request_body": None,
                "response_schema": {"type": "object"},
            },
            {
                "name": f"create_{current_service}",
                "method": "POST",
                "path": f"/{current_service}s",
                "description": f"Create a new {current_service.replace('_', ' ')}",
                "parameters": {},
                "required_params": [],
                "request_body": {"type": "object"},
                "response_schema": {"type": "object"},
            },
        ]
    
    # Create APIServerFallback entry with detailed info
    api_fallback: APIServerFallback = {
        "service": current_service,
        "reason": reasoning,
        "api_docs_url": api_docs_url,
        "base_url": base_url,
        "auth_type": auth_type,
        "auth_header": auth_header,
        "env_var": env_var,
        "endpoints": endpoints,
    }
    
    # Also create tool schemas from endpoints (for consistency with MCP tools)
    tool_schemas = []
    for ep in endpoints:
        # Combine path params and query params
        all_params = dict(ep.get("parameters", {}))
        
        # Add body params if present
        if ep.get("request_body") and isinstance(ep["request_body"], dict):
            body_props = ep["request_body"].get("properties", {})
            for key, val in body_props.items():
                all_params[key] = val
        
        tool_schema = {
            "name": ep["name"],
            "description": f"[API] {ep['description']} ({ep['method']} {ep['path']})",
            "inputSchema": {
                "type": "object",
                "properties": all_params,
                "required": ep.get("required_params", [])
            }
        }
        tool_schemas.append(tool_schema)
    
    print(f"   ✅ Created {len(tool_schemas)} tool schemas for API endpoints")
    
    return {
        "api_servers": [api_fallback],
        "tool_schemas": tool_schemas,
        "current_service_index": current_index + 1,
    }
