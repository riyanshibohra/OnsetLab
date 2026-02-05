"""
Meta-Agent FastAPI Server
=========================
REST API for the OnsetLab meta-agent. Handles:
- Dynamic MCP server discovery from official registry
- Loading services/tools from curated registry (fallback)
- Generating notebooks from selected tools
"""

import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Get registry path
REGISTRY_PATH = Path(__file__).parent.parent / "registry"

# Import discovery modules
from meta_agent.nodes.discover_servers import (
    fetch_all_servers,
    search_registry_for_service,
    extract_server_config
)
from meta_agent.nodes.verify_server import verify_single_server
from meta_agent.utils.gist_upload import upload_to_gist_async

app = FastAPI(
    title="OnsetLab Meta-Agent API",
    description="Build custom AI agents from tool registries",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class ServiceInfo(BaseModel):
    id: str
    name: str
    description: str
    tool_count: int


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict
    required_params: list[str]


class ServiceWithTools(BaseModel):
    id: str
    name: str
    description: str
    tools: list[ToolInfo]
    package: dict
    auth: dict


# Discovery Models (NEW)
class DiscoverRequest(BaseModel):
    services: list[str]  # Service names to discover (e.g., ["github", "slack"])


class DiscoveredServer(BaseModel):
    service: str
    name: str
    description: str
    version: str
    install_type: Optional[str] = None  # "npm" | "docker" | "remote"
    install_command: Optional[str] = None
    remote_url: Optional[str] = None
    env_vars: list[str] = []
    verified: bool = False
    score: int = 0
    tools_found: int = 0


class DiscoverResponse(BaseModel):
    success: bool
    servers: list[DiscoveredServer] = []
    not_found: list[str] = []
    error: Optional[str] = None


class GenerateRequest(BaseModel):
    problem_statement: str
    selected_services: list[str]
    selected_tools: dict[str, list[str]]  # service_id -> [tool_names]
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None  # For skill generation
    upload_to_gist: bool = False  # Optional: upload notebook to GitHub Gist
    github_token: Optional[str] = None  # Optional: GitHub token for Gist upload


class GenerateResponse(BaseModel):
    success: bool
    notebook_json: Optional[str] = None
    notebook_url: Optional[str] = None
    full_skill: Optional[str] = None      # Generated skill (for preview)
    condensed_rules: Optional[str] = None  # System prompt (for preview)
    error: Optional[str] = None
    gist_url: Optional[str] = None  # URL to GitHub Gist (if uploaded)
    colab_url: Optional[str] = None  # Direct Colab link (if uploaded)


# =============================================================================
# Registry Loading
# =============================================================================

def load_registry_file(service_id: str) -> dict:
    """Load a service's registry JSON file."""
    file_path = REGISTRY_PATH / f"{service_id}.json"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")
    
    with open(file_path, "r") as f:
        return json.load(f)


def get_all_services() -> list[dict]:
    """Get all available services from registry."""
    services = []
    
    for file_path in REGISTRY_PATH.glob("*.json"):
        # Skip hidden/internal files
        if file_path.stem.startswith("_"):
            continue
            
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            service_id = file_path.stem
            
            services.append({
                "id": service_id,
                "name": data.get("name", service_id.title()),
                "description": data.get("description", ""),
                "tool_count": len(data.get("tools", [])),
            })
        except Exception:
            continue
    
    return sorted(services, key=lambda s: s["name"])


# =============================================================================
# Notebook Generation (Standalone)
# =============================================================================

def create_notebook_cell(cell_type: str, source: list[str], execution_count: int = None) -> dict:
    """Create a Jupyter notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell


def generate_notebook_standalone(
    problem_statement: str,
    mcp_servers: list[dict],
    tool_schemas: list[dict],
    full_skill: str = None,
    condensed_rules: str = None,
) -> str:
    """Generate a Colab notebook from the given configuration."""
    
    cells = []
    
    # Cell 1: Title
    cells.append(create_notebook_cell("markdown", [
        "# 🚀 OnsetLab Agent Builder\n",
        "\n",
        "This notebook was auto-generated by OnsetLab.\n",
        "\n",
        f"**Problem Statement:** {problem_statement}\n",
        "\n",
        "## What this notebook does:\n",
        f"1. Configures **{len(mcp_servers)}** MCP servers\n",
        f"2. Registers **{len(tool_schemas)}** tools\n",
        "3. Generates synthetic training data\n",
        "4. Fine-tunes Qwen 2.5 3B (~15 min)\n",
        "5. Packages your agent for local deployment\n",
        "\n",
        "---"
    ]))
    
    # Cell 2: Installation
    cells.append(create_notebook_cell("markdown", [
        "## 1️⃣ Install Dependencies"
    ]))
    
    cells.append(create_notebook_cell("code", [
        "# Check GPU\n",
        "!nvidia-smi --query-gpu=name --format=csv,noheader\n",
        "\n",
        "# Install Unsloth\n",
        '!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n',
        "!pip install -q --no-deps xformers trl peft accelerate bitsandbytes\n",
        "!pip install -q openai anthropic httpx\n",
        "\n",
        "print('✅ Dependencies installed!')"
    ]))
    
    # Cell 3: Upload SDK
    cells.append(create_notebook_cell("markdown", [
        "## 2️⃣ Upload OnsetLab SDK\n",
        "\n",
        "Upload `onsetlab.zip` when prompted:"
    ]))
    
    cells.append(create_notebook_cell("code", [
        "from google.colab import files\n",
        "uploaded = files.upload()  # Upload onsetlab.zip\n",
        "!unzip -o onsetlab.zip\n",
        "print('✅ SDK extracted!')"
    ]))
    
    # Cell 4: API Key
    cells.append(create_notebook_cell("markdown", [
        "## 3️⃣ Enter Your LLM API Key\n",
        "\n",
        "For training data generation:"
    ]))
    
    cells.append(create_notebook_cell("code", [
        "import os\n",
        "\n",
        "# Enter ONE of these:\n",
        "os.environ['ANTHROPIC_API_KEY'] = ''  # @param {type:\"string\"}\n",
        "os.environ['OPENAI_API_KEY'] = ''  # @param {type:\"string\"}\n",
        "\n",
        "if os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENAI_API_KEY'):\n",
        "    print('✅ API key configured!')\n",
        "else:\n",
        "    print('⚠️ Please enter an API key above')"
    ]))
    
    # Cell 5: Tool Schemas
    cells.append(create_notebook_cell("markdown", [
        f"## 4️⃣ Tool Schemas ({len(tool_schemas)} tools)"
    ]))
    
    tools_code = [
        "from onsetlab import ToolSchema\n",
        "\n",
        "tools = [\n",
    ]
    
    for tool in tool_schemas:
        tools_code.append(f"    ToolSchema(\n")
        tools_code.append(f"        name=\"{tool.get('name', 'unknown')}\",\n")
        
        desc = tool.get('description', '').replace('"', '\\"').replace('\n', ' ')[:100]
        tools_code.append(f"        description=\"{desc}\",\n")
        
        raw_params = tool.get('parameters', {})
        clean_params = {}
        for param_name, param_def in raw_params.items():
            if isinstance(param_def, dict):
                param_type = param_def.get('type', 'string')
                clean_param = {
                    'type': param_type,
                    'description': param_def.get('description', '')[:80],
                }
                # Preserve nested properties for object types
                if param_type == 'object' and 'properties' in param_def:
                    clean_param['properties'] = param_def['properties']
                clean_params[param_name] = clean_param
        
        params_str = repr(clean_params) if clean_params else "{}"
        tools_code.append(f"        parameters={params_str},\n")
        
        required = tool.get('required_params', [])
        tools_code.append(f"        required_params={required},\n")
        tools_code.append(f"    ),\n")
    
    tools_code.append("]\n")
    tools_code.append(f"\nprint(f'✅ Loaded {{len(tools)}} tools')")
    
    cells.append(create_notebook_cell("code", tools_code))
    
    # Cell 6: MCP Servers
    cells.append(create_notebook_cell("markdown", [
        f"## 5️⃣ MCP Server Configurations ({len(mcp_servers)} servers)"
    ]))
    
    servers_code = [
        "from onsetlab import MCPServerConfig\n",
        "\n",
        "mcp_servers = [\n",
    ]
    
    for server in mcp_servers:
        service = server.get("service", "unknown")
        package = server.get("package", {})
        auth = server.get("auth", {})
        selected_tools = server.get("tools", [])
        
        pkg_type = package.get("type", "npm") if isinstance(package, dict) else "npm"
        
        servers_code.append(f"    MCPServerConfig(\n")
        
        if pkg_type == "docker":
            # Docker servers (e.g., GitHub)
            docker_image = package.get("image", "")
            servers_code.append(f'        package="{docker_image}",\n')
            servers_code.append(f'        server_type="docker",\n')
            servers_code.append(f'        docker_image="{docker_image}",\n')
        elif pkg_type == "binary":
            # Binary servers via npx (e.g., Slack)
            pkg_name = package.get("name", "")
            command = package.get("command", "npx")
            args = package.get("args", [])
            args_str = ", ".join(f'"{a}"' for a in args)
            servers_code.append(f'        package="{pkg_name}",\n')
            servers_code.append(f'        server_type="binary",\n')
            servers_code.append(f'        command="{command}",\n')
            servers_code.append(f'        args=[{args_str}],\n')
        else:
            # NPM servers (Notion, Google Calendar, Tavily, Filesystem, etc.)
            pkg_name = package.get("name", "")
            command = package.get("command", "npx")
            args = package.get("args", [])
            servers_code.append(f'        package="{pkg_name}",\n')
            # Include command/args for npm servers too
            if command and args:
                args_str = ", ".join(f'"{a}"' for a in args)
                servers_code.append(f'        command="{command}",\n')
                servers_code.append(f'        args=[{args_str}],\n')
        
        # Environment variables
        env_vars = auth.get("env_vars", [])
        if env_vars:
            env_vars_str = ", ".join(f'"{ev}"' for ev in env_vars)
            servers_code.append(f'        env_vars=[{env_vars_str}],\n')
        
        # Tool names
        tool_names = [t.get("name", "") for t in selected_tools if t.get("name")]
        if tool_names:
            tool_names_str = ", ".join(f'"{t}"' for t in tool_names)
            servers_code.append(f'        tools=[{tool_names_str}],\n')
        
        servers_code.append(f"    ),\n")
    
    servers_code.append("]\n")
    servers_code.append(f"\nprint(f'✅ Configured {{len(mcp_servers)}} MCP servers')")
    
    cells.append(create_notebook_cell("code", servers_code))
    
    # NOTE: Skill generation is now handled internally by AgentBuilder
    # No need to show skill/system_prompt cells to the user
    section_num = 6
    
    # Cell 6: Build
    cells.append(create_notebook_cell("markdown", [
        f"## {section_num}️⃣ Build Your Agent\n",
        "\n",
        "This will generate training data and fine-tune the model (~15-20 min):"
    ]))
    
    escaped_problem = problem_statement.replace('"', '\\"').replace('\n', '\\n')
    
    build_code = [
        "from onsetlab import AgentBuilder, BuildConfig\n",
        "import os\n",
        "\n",
        "config = BuildConfig(\n",
        "    num_examples=None,\n",
        "    base_model='phi-3.5-fc',  # Phi-3.5-mini-instruct-hermes-fc (3.8B, best for function calling)\n",
        "    epochs=None,\n",
        "    agent_name='my_agent',\n",
        "    output_dir='./agent_build',\n",
        "    runtime='both',\n",
        ")\n",
        "\n",
        "builder = AgentBuilder(\n",
        f"    problem_statement=\"\"\"{escaped_problem}\"\"\",\n",
        "    tools=tools,\n",
        "    mcp_servers=mcp_servers,\n",
        "    api_key=os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENAI_API_KEY'),\n",
        "    config=config,\n",
        ")\n",
        "\n",
        "# Build the agent (skill generation happens automatically inside!)\n",
        "agent = builder.build()"
    ]
    
    cells.append(create_notebook_cell("code", build_code))
    
    # Cell 8: Download
    section_num += 1
    cells.append(create_notebook_cell("markdown", [
        f"## {section_num}️⃣ Download Your Agent"
    ]))
    
    cells.append(create_notebook_cell("code", [
        "zip_path = agent.export('./my_agent.zip')\n",
        "\n",
        "from google.colab import files\n",
        "files.download(zip_path)\n",
        "\n",
        "print('🎉 Agent exported!')"
    ]))
    
    # Cell 9: Next steps
    cells.append(create_notebook_cell("markdown", [
        "## 🎉 Next Steps\n",
        "\n",
        "```bash\n",
        "unzip my_agent.zip && cd my_agent\n",
        "pip install -r requirements.txt\n",
        "cp .env.example .env  # Add your service tokens\n",
        "ollama create my_agent -f Modelfile\n",
        "python agent.py --interactive\n",
        "```\n",
        "\n",
        "---\n",
        "Built with ❤️ by OnsetLab"
    ]))
    
    # Create notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU"
        },
        "cells": cells
    }
    
    return json.dumps(notebook, indent=2)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "OnsetLab Meta-Agent API"}


@app.get("/api/services", response_model=list[ServiceInfo])
async def list_services():
    """List all available services from the registry."""
    return get_all_services()


@app.get("/api/services/{service_id}", response_model=ServiceWithTools)
async def get_service(service_id: str):
    """Get a service with all its tools."""
    data = load_registry_file(service_id)
    
    tools = []
    for tool in data.get("tools", []):
        tools.append(ToolInfo(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            parameters=tool.get("parameters", {}),
            required_params=tool.get("required_params", []),
        ))
    
    return ServiceWithTools(
        id=service_id,
        name=data.get("name", service_id.title()),
        description=data.get("description", ""),
        tools=tools,
        package=data.get("package", {}),
        auth=data.get("auth", {}),
    )


@app.get("/api/services/{service_id}/tools", response_model=list[ToolInfo])
async def get_service_tools(service_id: str):
    """Get tools for a specific service."""
    data = load_registry_file(service_id)
    
    tools = []
    for tool in data.get("tools", []):
        tools.append(ToolInfo(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            parameters=tool.get("parameters", {}),
            required_params=tool.get("required_params", []),
        ))
    
    return tools


# =============================================================================
# Discovery Endpoints (NEW - searches official MCP Registry)
# =============================================================================

# Cache for registry data (avoid fetching on every request)
_registry_cache = None
_registry_cache_time = None

def get_registry_cache():
    """Get cached registry data, refreshing if older than 5 minutes."""
    import time
    global _registry_cache, _registry_cache_time
    
    # Refresh cache every 5 minutes
    if _registry_cache is None or (time.time() - (_registry_cache_time or 0)) > 300:
        print("📡 Fetching MCP Registry (cache refresh)...")
        _registry_cache = fetch_all_servers()
        _registry_cache_time = time.time()
        print(f"   ✅ Cached {len(_registry_cache)} servers")
    
    return _registry_cache


@app.post("/api/discover", response_model=DiscoverResponse)
async def discover_servers(request: DiscoverRequest):
    """
    Discover MCP servers from the official registry.
    
    Takes a list of service names and searches the MCP Registry for matching servers.
    Returns verified server configurations.
    
    Example:
        POST /api/discover
        {"services": ["github", "slack", "linear"]}
    """
    try:
        # Get cached registry data
        all_servers = get_registry_cache()
        
        if not all_servers:
            return DiscoverResponse(
                success=False,
                error="Failed to fetch MCP Registry"
            )
        
        discovered = []
        not_found = []
        
        for service in request.services:
            print(f"🔍 Searching for: {service}")
            
            # Search registry
            matches = search_registry_for_service(service, all_servers)
            
            if matches:
                # Get best match
                best = matches[0]
                config = extract_server_config(best)
                
                # Verify the server
                verification = verify_single_server(config)
                
                # Build install info
                install_type = None
                install_command = None
                install_info = config.get("install", {})
                
                if install_info:
                    install_type = install_info.get("type")
                    install_command = install_info.get("command")
                
                discovered.append(DiscoveredServer(
                    service=service,
                    name=config.get("name", ""),
                    description=config.get("description", "")[:200],
                    version=config.get("version", ""),
                    install_type=install_type,
                    install_command=install_command,
                    remote_url=config.get("remote_url"),
                    env_vars=[e.get("name", "") for e in config.get("env_vars", [])],
                    verified=verification.get("verified", False),
                    score=verification.get("score", 0),
                    tools_found=len(verification.get("extracted_tools", [])),
                ))
                
                print(f"   ✅ Found: {config.get('name')} (score: {verification.get('score', 0)})")
            else:
                not_found.append(service)
                print(f"   ❌ Not found: {service}")
        
        return DiscoverResponse(
            success=True,
            servers=discovered,
            not_found=not_found,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DiscoverResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_notebook(request: GenerateRequest):
    """
    Generate a Colab notebook for the selected tools.
    
    Note: Skill generation now happens inside the SDK's AgentBuilder.build()
    This keeps the notebook simple and the SDK smart.
    """
    try:
        # Load selected services and tools
        mcp_servers = []
        tool_schemas = []
        
        for service_id, tool_names in request.selected_tools.items():
            data = load_registry_file(service_id)
            
            # Filter tools
            selected_tools = []
            for tool in data.get("tools", []):
                if tool.get("name") in tool_names:
                    tool["_service"] = service_id
                    selected_tools.append(tool)
                    tool_schemas.append(tool)
            
            # Build MCP server config
            if selected_tools:
                mcp_servers.append({
                    "service": service_id,
                    "name": data.get("name", service_id),
                    "package": data.get("package", {}),
                    "auth": data.get("auth", {}),
                    "tools": selected_tools,
                })
        
        # Generate notebook (SDK handles skill generation internally)
        notebook_json = generate_notebook_standalone(
            problem_statement=request.problem_statement,
            mcp_servers=mcp_servers,
            tool_schemas=tool_schemas,
        )
        
        # Upload to Gist if requested
        gist_url = None
        colab_url = None
        if request.upload_to_gist:
            try:
                gist_result = await upload_to_gist_async(
                    notebook_json=notebook_json,
                    filename="onsetlab_agent_builder.ipynb",
                    description=f"OnsetLab Agent: {request.problem_statement[:100]}",
                    github_token=request.github_token,
                    public=True,
                )
                gist_url = gist_result["gist_url"]
                colab_url = gist_result["colab_url"]
            except Exception as gist_error:
                print(f"⚠️  Gist upload failed: {gist_error}")
                # Don't fail the whole request if gist upload fails
        
        return GenerateResponse(
            success=True,
            notebook_json=notebook_json,
            gist_url=gist_url,
            colab_url=colab_url,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return GenerateResponse(
            success=False,
            error=str(e),
        )


# =============================================================================
# Gist Upload Endpoint
# =============================================================================

class UploadGistRequest(BaseModel):
    notebook_json: str
    github_token: Optional[str] = None


class UploadGistResponse(BaseModel):
    success: bool
    gist_url: Optional[str] = None
    colab_url: Optional[str] = None
    error: Optional[str] = None


@app.post("/api/upload-gist", response_model=UploadGistResponse)
async def upload_gist(request: UploadGistRequest):
    """Upload a notebook to GitHub Gist and return Colab URL."""
    try:
        gist_result = await upload_to_gist_async(
            notebook_json=request.notebook_json,
            filename="onsetlab_agent_builder.ipynb",
            description="OnsetLab Agent Builder Notebook",
            github_token=request.github_token,
            public=True,
        )
        return UploadGistResponse(
            success=True,
            gist_url=gist_result["gist_url"],
            colab_url=gist_result["colab_url"],
        )
    except Exception as e:
        return UploadGistResponse(
            success=False,
            error=str(e),
        )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting OnsetLab Meta-Agent API...")
    print("   Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
