"""
Extract Schemas Node
====================
Fetches MCP server README and extracts tool schemas.
Includes self-reflection to validate tools match the problem statement.
"""

import json
import anthropic

from meta_agent.state import MetaAgentState, MCPServerDiscovery
from meta_agent.tools.github_tools import fetch_github_file_sync, extract_package_json
from meta_agent.tools.npm_tools import (
    get_npm_package_info, 
    validate_mcp_package_health,
)


VALIDATE_TOOLS_PROMPT = """You are an expert at evaluating if discovered tools match their SERVICE PURPOSE.

Given:
1. A problem statement that may involve MULTIPLE services
2. The CURRENT SERVICE being validated (e.g., "github", "slack")
3. Tools extracted for that specific service

Determine if the tools are relevant FOR THAT SERVICE'S ROLE in the problem.

## IMPORTANT:
- Check if tools help with that SERVICE's specific functionality
- Do NOT penalize one service's MCP for lacking another service's tools
- Each MCP server only needs to provide tools for ITS service

## RULES:
1. Focus ONLY on whether these tools serve their service's purpose
2. At least 2-3 core tools should match the service's role
3. Generic/low-level tools don't count - need actual service API tools

## RESPONSE FORMAT (JSON):
{
    "is_relevant": true | false,
    "matching_tools": ["tool1", "tool2"],
    "missing_capabilities": ["what tools for THIS SERVICE are needed but missing"],
    "reasoning": "Brief explanation focused on the service's role"
}

Respond with valid JSON."""


def detect_server_type(github_url: str, is_npm_package: bool) -> tuple[str, str]:
    """
    Detect the server type and Docker image based on repository contents.
    
    Returns:
        Tuple of (server_type, docker_image)
        server_type: "npm", "docker", "go", "python", "binary"
        docker_image: Docker image URL if applicable
    """
    if is_npm_package:
        return "npm", ""
    
    # Check for Go (go.mod)
    try:
        go_mod = fetch_github_file_sync(github_url, "go.mod")
        if go_mod and "module " in go_mod:
            # Try to find Docker image from README or Dockerfile
            docker_image = ""
            # Parse owner/repo from GitHub URL
            if "github.com" in github_url:
                parts = github_url.split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1].split("?")[0].split("#")[0]
                    # Common pattern: ghcr.io/owner/repo
                    docker_image = f"ghcr.io/{owner}/{repo}"
            return "docker", docker_image  # Go binary, use Docker wrapper
    except:
        pass
    
    # Check for Python (pyproject.toml or setup.py)
    try:
        pyproject = fetch_github_file_sync(github_url, "pyproject.toml")
        if pyproject and "[project]" in pyproject:
            docker_image = ""
            if "github.com" in github_url:
                parts = github_url.split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1].split("?")[0].split("#")[0]
                    docker_image = f"ghcr.io/{owner}/{repo}"
            return "docker", docker_image  # Python, use Docker wrapper
    except:
        pass
    
    try:
        setup_py = fetch_github_file_sync(github_url, "setup.py")
        if setup_py and "setup(" in setup_py:
            docker_image = ""
            if "github.com" in github_url:
                parts = github_url.split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1].split("?")[0].split("#")[0]
                    docker_image = f"ghcr.io/{owner}/{repo}"
            return "docker", docker_image
    except:
        pass
    
    # Check for Dockerfile (indicates Docker distribution)
    try:
        dockerfile = fetch_github_file_sync(github_url, "Dockerfile")
        if dockerfile and "FROM " in dockerfile:
            docker_image = ""
            if "github.com" in github_url:
                parts = github_url.split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1].split("?")[0].split("#")[0]
                    docker_image = f"ghcr.io/{owner}/{repo}"
            return "docker", docker_image
    except:
        pass
    
    # Default to NPM (most MCP servers are npm packages)
    return "npm", ""


def fetch_readme_from_github(github_url: str) -> str:
    """
    Fetch README from GitHub URL, handling subdirectory URLs.
    Tries multiple README filename variants.
    """
    for filename in ["README.md", "readme.md", "Readme.md", "README"]:
        try:
            content = fetch_github_file_sync(github_url, filename)
            if content and len(content) > 100:
                return content
        except Exception:
            continue
    return ""


def fetch_source_files_from_github(github_url: str) -> str:
    """
    Fetch source files that might contain tool definitions.
    MCP servers typically define tools in TypeScript/JavaScript files.
    """
    source_files = [
        "index.ts",
        "src/index.ts",
        "tools.ts",
        "src/tools.ts",
        "index.js",
        "src/index.js",
    ]
    
    combined_content = []
    for filename in source_files:
        try:
            content = fetch_github_file_sync(github_url, filename)
            if content and len(content) > 50:
                combined_content.append(f"=== {filename} ===\n{content}")
        except Exception:
            continue
    
    return "\n\n".join(combined_content) if combined_content else ""


def validate_tools_relevance(
    problem_statement: str,
    tools: list,
    current_service: str,
    api_key: str
) -> tuple[bool, str]:
    """
    Validate if extracted tools are relevant for THIS SERVICE'S role in the problem.
    
    Args:
        problem_statement: Full problem description
        tools: Extracted tools for current service
        current_service: The service being validated (e.g., "github", "slack")
        api_key: OpenAI API key
    
    Returns:
        Tuple of (is_relevant, reasoning)
    """
    if not tools:
        return False, "No tools extracted"
    
    tool_names = [t.get("name", "") for t in tools]
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"""{VALIDATE_TOOLS_PROMPT}

Problem Statement:
{problem_statement}

Current Service Being Validated: {current_service}

Extracted Tools for {current_service} ({len(tools)} total):
{json.dumps(tool_names, indent=2)}

Are these {current_service} tools relevant for the {current_service}-related parts of the problem?
Respond with valid JSON."""}
            ],
        )
        
        response_text = response.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        is_relevant = result.get("is_relevant", False)
        reasoning = result.get("reasoning", "")
        matching = result.get("matching_tools", [])
        missing = result.get("missing_capabilities", [])
        
        if missing:
            reasoning += f" Missing: {', '.join(missing[:3])}"
        
        return is_relevant, reasoning
        
    except Exception as e:
        # On error, assume relevant to avoid blocking
        return True, f"Validation error: {e}"


EXTRACT_TOOLS_PROMPT = """You are an expert at extracting MCP (Model Context Protocol) tool definitions.

You will receive either:
1. README documentation describing the MCP server
2. TypeScript/JavaScript source code with tool definitions
3. Both

For TypeScript MCP servers, tools are typically defined using patterns like:
- server.setRequestHandler(ListToolsRequestSchema, ...) with tool definitions
- { name: "tool_name", description: "...", inputSchema: {...} }
- Functions decorated or registered as MCP tools

Extract ALL tools with their schemas. For each tool:
1. name: The exact tool name (e.g., "create_issue", "list_repositories", "send_message")
2. description: What the tool does
3. parameters: Object with parameter definitions (type and description for each)
4. required_params: List of required parameter names

## CRITICAL: Extract ALL Environment Variables!

Be AGGRESSIVE about finding env vars. Look in:
- README "Environment Variables", "Setup", "Configuration" sections
- Code patterns: process.env.*, os.environ, env.get()
- .env.example, .env.sample files
- Docker/compose configurations
- Any mention of required tokens, keys, or IDs

Extract ANY env var that looks like a credential or required config:
- *_TOKEN, *_API_KEY, *_SECRET
- *_ID (team IDs, workspace IDs, org IDs - these are often REQUIRED!)
- *_KEY, *_CREDENTIAL, *_PASSWORD

IMPORTANT: Many MCP servers require MULTIPLE env vars (e.g., both a token AND an ID).
If you see mentions of multiple required credentials, include them ALL.

Respond with JSON in this exact format:
{
    "tools": [
        {
            "name": "action_name",
            "description": "Brief description of what this tool does",
            "parameters": {
                "param1": {"type": "string", "description": "Parameter description"},
                "param2": {"type": "string", "description": "Description", "enum": ["VALUE1", "VALUE2"]}
            },
            "required_params": ["param1"]
        }
    ],
    "auth_type": "oauth" | "token" | "api_key" | "none",
    "env_vars": ["SERVICE_TOKEN", "SERVICE_ID"],
    "setup_notes": "Any setup notes"
}

## PARAMETER ENUM VALUES:
- If a parameter has a fixed set of valid values (enum), include them in UPPERCASE
- Example: "state": {"type": "string", "enum": ["OPEN", "CLOSED", "ALL"], "description": "Issue state"}
- Common enums: issue states, order statuses, event types, sort directions

## IMPORTANT RULES:
- Extract the ACTUAL tools defined for THIS specific MCP server
- Include ALL required environment variables (not just the first one!)
- Some MCP servers require multiple env vars - capture them all
- Be thorough - extract ALL tools mentioned in the source code or docs"""


def try_extract_from_candidate(
    candidate: dict,
    current_service: str,
    problem_statement: str,
    api_key: str
) -> tuple[list, dict, bool, str]:
    """
    Try to extract tools from a single candidate.
    
    Returns:
        Tuple of (tools, mcp_discovery, is_valid, reason)
    """
    package_name = candidate.get("package_name")
    github_url = candidate.get("github_url")
    auth_type = candidate.get("auth_type", "unknown")
    env_var = candidate.get("env_var")
    confidence = candidate.get("confidence", 0.5)
    
    # Detect if this is a GitHub repo (owner/repo) vs NPM package (@scope/name)
    is_github_repo = (
        package_name and 
        "/" in package_name and 
        not package_name.startswith("@")
    )
    is_npm_package = package_name and package_name.startswith("@")
    
    print(f"   📦 Trying: {package_name}")
    
    # STEP 1: Validate package health - ONLY for NPM packages
    if is_npm_package:
        health = validate_mcp_package_health(package_name)
        
        if not health.get("healthy"):
            issues = health.get("issues", [])
            print(f"      ❌ Package unhealthy: {'; '.join(issues)}")
            
            # Check for the specific missing SDK issue
            if not health.get("has_mcp_sdk"):
                print(f"      ⚠️ Missing @modelcontextprotocol/sdk - will fail at runtime!")
                return [], {}, False, "Package missing MCP SDK dependency"
        else:
            print(f"      ✅ Package health: score={health.get('score', 0)}")
    elif is_github_repo:
        # For GitHub repos, skip NPM health check - just use the repo directly
        print(f"      📂 GitHub repo detected, skipping NPM check")
        # Convert to github URL if not already set
        if not github_url:
            github_url = f"https://github.com/{package_name}"
    
    # Get actual GitHub URL
    npm_info = None
    actual_github_url = github_url
    npm_package_name = package_name if is_npm_package else None
    
    # Only look up NPM info for NPM packages
    if is_npm_package:
        npm_info = get_npm_package_info(package_name)
        if npm_info.get("exists"):
            if npm_info.get("deprecated") and npm_info.get("new_repo_url"):
                actual_github_url = npm_info["new_repo_url"]
                print(f"      🔗 NEW REPO (deprecated): {actual_github_url}")
            elif npm_info.get("repository"):
                actual_github_url = npm_info["repository"]
    elif is_github_repo:
        # For GitHub repos, construct URL and try to find NPM package name from package.json
        actual_github_url = f"https://github.com/{package_name}"
        print(f"      🔗 Using GitHub repo: {actual_github_url}")
        
        # Try to find NPM package name from package.json
        try:
            pkg_json = extract_package_json(actual_github_url)
            if pkg_json and pkg_json.get("name"):
                npm_package_name = pkg_json["name"]
                print(f"      📦 Found NPM package: {npm_package_name}")
                
                # Validate this NPM package health
                health = validate_mcp_package_health(npm_package_name)
                if not health.get("healthy"):
                    issues = health.get("issues", [])
                    print(f"      ⚠️ Package issues: {'; '.join(issues)}")
                    if not health.get("has_mcp_sdk"):
                        print(f"      ⚠️ Missing @modelcontextprotocol/sdk - may fail at runtime!")
                        # Don't fail immediately for GitHub repos, just warn
        except Exception as e:
            print(f"      ⚠️ Could not get package.json: {e}")
    
    if not actual_github_url:
        return [], {}, False, "No GitHub URL"
    
    # Fetch docs
    readme_content = fetch_readme_from_github(actual_github_url)
    source_content = fetch_source_files_from_github(actual_github_url)
    
    combined_content = ""
    if readme_content:
        combined_content += f"=== README ===\n{readme_content}\n\n"
    if source_content:
        combined_content += f"=== SOURCE FILES ===\n{source_content}\n\n"
    
    if not combined_content:
        return [], {}, False, "No documentation found"
    
    print(f"      ✅ Fetched docs ({len(combined_content)} chars)")
    
    # Extract tools - limit content to avoid truncation issues
    # 15K chars is safer for Claude to process without JSON issues
    content_limit = 15000
    truncated_content = combined_content[:content_limit]
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"""{EXTRACT_TOOLS_PROMPT}

MCP Server: {package_name or current_service}
Service Type: {current_service}

Documentation and Source Code:
{truncated_content}

Extract all tool definitions for this {current_service} MCP server.
Respond with valid JSON only. No markdown, no explanation, just the JSON object."""}
            ],
        )
        
        response_text = response.content[0].text.strip()
        
        # Extract JSON from response - handle various formats
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            # Find the JSON block
            parts = response_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{"):
                    response_text = part
                    break
        
        # Clean up any trailing text after JSON
        response_text = response_text.strip()
        
        # Find the end of the JSON object
        brace_count = 0
        json_end = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response_text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
        
        if json_end > 0:
            response_text = response_text[:json_end]
        
        result = json.loads(response_text)
        tools = result.get("tools", [])
        env_vars = result.get("env_vars", [])
        
    except json.JSONDecodeError as e:
        print(f"      ⚠️ JSON parse error: {e}")
        return [], {}, False, f"Extraction failed: JSON parse error"
    except Exception as e:
        return [], {}, False, f"Extraction failed: {e}"
    
    if not tools:
        return [], {}, False, "No tools extracted"
    
    print(f"      ✅ Extracted {len(tools)} tools")
    
    # Validate tools against problem statement (service-specific)
    is_relevant, reason = validate_tools_relevance(
        problem_statement, tools, current_service, api_key
    )
    
    if not is_relevant:
        print(f"      ❌ Tools not relevant: {reason}")
        return tools, {}, False, reason
    
    print(f"      ✅ Tools validated as relevant")
    
    # Create MCP discovery - capture ALL env vars
    all_env_vars = []
    
    # Add env vars from evaluation
    if env_var:
        all_env_vars.append(env_var)
    
    # Add env vars from extraction
    if env_vars:
        for ev in env_vars:
            if ev and ev not in all_env_vars:
                all_env_vars.append(ev)
    
    if all_env_vars:
        print(f"      📋 Required env vars: {', '.join(all_env_vars)}")
    
    # Use the discovered NPM package name if available, otherwise the original
    final_package_name = npm_package_name or package_name or f"unknown-{current_service}"
    
    # Detect server type (npm vs docker vs go etc.)
    server_type, docker_image = detect_server_type(actual_github_url, is_npm_package)
    
    if server_type != "npm":
        print(f"      🐳 Detected {server_type} server (Docker: {docker_image or 'TBD'})")
    
    mcp_discovery = {
        "service": current_service,
        "package": final_package_name,
        "server_type": server_type,  # "npm", "docker", "go", "python", "binary"
        "auth_type": auth_type,
        "env_vars": all_env_vars,  # Now a LIST of all required env vars
        "tools": tools,
        "setup_url": actual_github_url,
        "confidence": confidence,
        "docker_image": docker_image if server_type != "npm" else None,
        "repo_url": actual_github_url,
    }
    
    return tools, mcp_discovery, True, "Success"


def extract_schemas(state: MetaAgentState) -> dict:
    """
    Extract tool schemas from the discovered MCP server.
    
    Implements RETRY LOGIC:
    1. Try each candidate from evaluate_mcp
    2. After extracting, validate tools match problem statement
    3. If not relevant, try next candidate
    4. Return best result or mark for API fallback
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with new mcp_servers and tool_schemas entries
    """
    current_service = state.get("current_service")
    problem_statement = state.get("problem_statement", "")
    api_key = state["anthropic_api_key"]
    current_index = state["current_service_index"]
    
    # Get ALL candidates from evaluation
    candidates = state.get("_eval_candidates", [])
    
    # Fallback to single candidate if no list
    if not candidates:
        package_name = state.get("_eval_package_name")
        if package_name:
            candidates = [{
                "package_name": package_name,
                "github_url": state.get("_eval_github_url"),
                "auth_type": state.get("_eval_auth_type", "unknown"),
                "env_var": state.get("_eval_env_var"),
                "confidence": state.get("_eval_confidence", 0.5),
            }]
    
    print(f"📦 Extracting schemas for: {current_service}")
    print(f"   🔄 {len(candidates)} candidate(s) to try")
    
    # =================================================================
    # TRY EACH CANDIDATE WITH SELF-REFLECTION
    # =================================================================
    best_tools = []
    best_mcp_discovery = None
    
    for i, candidate in enumerate(candidates):
        print(f"\n   --- Attempt {i+1}/{len(candidates)} ---")
        
        tools, mcp_discovery, is_valid, reason = try_extract_from_candidate(
            candidate=candidate,
            current_service=current_service,
            problem_statement=problem_statement,
            api_key=api_key
        )
        
        if is_valid and tools:
            # Found a good candidate!
            best_tools = tools
            best_mcp_discovery = mcp_discovery
            print(f"   ✅ SUCCESS with candidate {i+1}")
            break
        else:
            print(f"   ⏭️ Trying next candidate... ({reason})")
    
    # If no candidate worked
    if not best_mcp_discovery:
        print(f"   ❌ No working MCP server found for {current_service}")
        print(f"   ⚠️ Will fall back to API implementation")
        
        # Return empty to trigger mark_as_api
        return {
            "result_quality": "no_mcp",  # Override to trigger API fallback
            "current_service_index": current_index + 1,
        }
    
    # Convert tools to tool_schemas format
    tool_schemas = []
    for tool in best_tools:
        tool_schema = {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "inputSchema": {
                "type": "object",
                "properties": tool.get("parameters", {}),
                "required": tool.get("required_params", [])
            }
        }
        tool_schemas.append(tool_schema)
    
    print(f"\n   ✅ Added MCP server: {best_mcp_discovery.get('package')}")
    print(f"   📋 {len(tool_schemas)} validated tools")
    
    return {
        "mcp_servers": [best_mcp_discovery],
        "tool_schemas": tool_schemas,
        "current_service_index": current_index + 1,
    }
