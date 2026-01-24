"""
Agent Packager
==============
Generates runnable agent packages with all necessary files.

Supports two runtime options:
1. Ollama (GGUF) - Simple CLI: `ollama run my-agent`
2. Python - Direct script: `python agent.py`
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RuntimeType(Enum):
    """Supported runtime types."""
    OLLAMA = "ollama"      # GGUF model with Ollama
    PYTHON = "python"      # Python script with transformers
    BOTH = "both"          # Generate both options


@dataclass
class PackageConfig:
    """Configuration for agent packaging."""
    runtime: RuntimeType = RuntimeType.BOTH
    agent_name: str = "onset-agent"  # Branded default name
    output_dir: str = "./agent_package"
    include_training_data: bool = False  # Include train.jsonl for debugging/sharing
    include_readme: bool = True


class AgentPackager:
    """
    Packages a trained agent for deployment.
    
    Supports two tool backends:
    1. MCP Servers - For services with MCP protocol support
    2. API Servers - For services with direct REST APIs (no MCP)
    
    Generates:
    - For Ollama: Modelfile, config.yaml
    - For Python: agent.py, tools.py (API wrappers), requirements.txt
    - Common: README.md, system_prompt.txt, mcp_config.json
    """
    
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        tools: list,
        mcp_servers=None,  # Can be list or dict of MCPServerConfig
        api_servers: list = None,  # List of APIServerConfig
        model_path: str = None,
        config: PackageConfig = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = tools
        # Normalize mcp_servers to list (handles both dict and list input)
        self.mcp_servers = self._normalize_mcp_servers(mcp_servers)
        self.api_servers = api_servers or []  # NEW: API-based services
        self.model_path = model_path
        self.config = config or PackageConfig(agent_name=agent_name)
    
    def _normalize_mcp_servers(self, mcp_servers) -> list:
        """Convert mcp_servers to a list regardless of input format."""
        if mcp_servers is None:
            return []
        if isinstance(mcp_servers, dict):
            # Dict format: {"github": MCPServerConfig(...), ...}
            return list(mcp_servers.values())
        if isinstance(mcp_servers, list):
            return mcp_servers
        return []
    
    def package(self) -> str:
        """
        Create the agent package.
        
        Returns path to the package directory.
        """
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📦 Packaging agent to: {output_dir}")
        
        try:
            # Copy the GGUF model file (most important!)
            self._copy_model(output_dir)
            
            # Common files
            self._write_system_prompt(output_dir)
            self._write_mcp_config(output_dir)
            self._write_env_example(output_dir)
            self._write_setup_script(output_dir)
            self._write_gitignore(output_dir)
            
            # Runtime-specific files
            if self.config.runtime in (RuntimeType.OLLAMA, RuntimeType.BOTH):
                self._write_ollama_files(output_dir)
            
            if self.config.runtime in (RuntimeType.PYTHON, RuntimeType.BOTH):
                self._write_python_files(output_dir)
            
            # README
            if self.config.include_readme:
                self._write_readme(output_dir)
            
            print(f"✅ Agent packaged successfully!")
            return output_dir
            
        except Exception as e:
            print(f"   ❌ Error during packaging: {e}")
            raise
    
    def _copy_model(self, output_dir: str):
        """Copy the GGUF model file to the package."""
        import shutil
        import glob
        
        # Search for GGUF file in multiple locations (Unsloth can save anywhere)
        search_locations = []
        
        # 1. Provided model path
        if self.model_path:
            search_locations.append(os.path.join(self.model_path, "*.gguf"))
            search_locations.append(os.path.join(self.model_path, "**", "*.gguf"))
            if self.model_path.endswith('.gguf'):
                search_locations.append(self.model_path)
        
        # 2. Common Unsloth output locations
        search_locations.extend([
            "./*.gguf",                    # Current directory
            "./**/*.gguf",                 # Recursive from current
            "./agent_build/model/gguf/*.gguf",
            "./agent_build/model/gguf/**/*.gguf",
            "/content/*.gguf",             # Colab root
            "/content/**/*.gguf",          # Colab recursive
        ])
        
        gguf_file = None
        all_found = []
        
        for pattern in search_locations:
            try:
                files = glob.glob(pattern, recursive=True)
                all_found.extend(files)
            except:
                continue
        
        # Remove duplicates and filter for Q4_K_M preference
        all_found = list(set(all_found))
        
        if all_found:
            # Prefer Q4_K_M quantization
            gguf_file = next(
                (f for f in all_found if 'Q4_K_M' in f or 'q4_k_m' in f.lower()),
                all_found[0]
            )
        
        if gguf_file and os.path.exists(gguf_file):
            dest = os.path.join(output_dir, "model.gguf")
            shutil.copy(gguf_file, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"   📦 model.gguf ({size_mb:.0f} MB) - from {gguf_file}")
        else:
            print(f"   ⚠️ No GGUF file found!")
            print(f"      Searched: {self.model_path}, ./agent_build/model/gguf/, /content/")
            print(f"      Found files: {all_found[:5] if all_found else 'none'}")
    
    def _write_system_prompt(self, output_dir: str):
        """Write system prompt file."""
        path = os.path.join(output_dir, "system_prompt.txt")
        with open(path, "w") as f:
            f.write(self.system_prompt)
        print(f"   📄 system_prompt.txt")
    
    def _write_mcp_config(self, output_dir: str):
        """
        Write configuration for both MCP servers and API servers.
        
        The agent uses this to know:
        1. Which MCP servers to connect to
        2. Which API servers are available
        3. Which tools are whitelisted (trained on)
        """
        config = {
            "mcpServers": {},
            "apiServers": {},  # NEW: API-based services
            "allowed_tools": [],
            "mcp_tools": [],   # Tools that use MCP
            "api_tools": []    # Tools that use direct API
        }
        
        # Build MCP server configs
        for server in self.mcp_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            
            name = server_dict.get("name", "server")
            command = server_dict.get("command", "npx")
            args = server_dict.get("args", [])
            env_var = server_dict.get("env_var", "API_KEY")
            server_tools = server_dict.get("tools", [])
            
            config["mcpServers"][name] = {
                "command": command,
                "args": args,
                "env": {
                    env_var: f"${{{env_var}}}"
                },
                "tools": server_tools
            }
            
            # Track MCP tools
            config["mcp_tools"].extend(server_tools)
        
        # Build API server configs
        for server in self.api_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            
            name = server_dict.get("name", "api")
            base_url = server_dict.get("base_url", "")
            auth_env_var = server_dict.get("auth_env_var", "API_KEY")
            server_tools = [
                t.get("name") if isinstance(t, dict) else t.name 
                for t in server_dict.get("tools", [])
            ]
            
            config["apiServers"][name] = {
                "base_url": base_url,
                "auth_env_var": auth_env_var,
                "tools": server_tools
            }
            
            # Track API tools
            config["api_tools"].extend(server_tools)
        
        # Collect all allowed tools (only tools we trained on)
        for tool in self.tools:
            tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
            config["allowed_tools"].append(tool_dict.get("name", ""))
        
        path = os.path.join(output_dir, "mcp_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"   📄 mcp_config.json (MCP: {len(config['mcp_tools'])}, API: {len(config['api_tools'])} tools)")
    
    def _write_env_example(self, output_dir: str):
        """
        Write .env.example file with all required credentials.
        
        GENERIC: Uses MCPServerConfig and APIServerConfig data directly.
        """
        env_content = f"""# {self.agent_name} - Environment Variables
# ============================================
# Copy this file to .env and fill in your credentials:
#   cp .env.example .env
#
# Then edit .env with your actual values.
# NEVER commit .env to version control!
# ============================================

"""
        # MCP Servers
        if self.mcp_servers:
            env_content += "# ==========================================\n"
            env_content += "# MCP SERVERS\n"
            env_content += "# ==========================================\n\n"
            
            for server in self.mcp_servers:
                server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
                package = server_dict.get("package", "unknown")
                env_var = server_dict.get("env_var", "API_KEY")
                auth_type = server_dict.get("auth_type", "token")
                description = server_dict.get("description", "")
                setup_url = server_dict.get("setup_url", "")
                example_value = server_dict.get("example_value", "")
                
                env_content += f"# {package}\n"
                if description:
                    env_content += f"# {description}\n"
                
                # Add helpful hints based on auth type
                if auth_type == "token":
                    env_content += f"# Get your token from the provider's settings page\n"
                elif auth_type == "oauth":
                    env_content += f"# Path to OAuth credentials JSON file\n"
                elif auth_type == "connection_string":
                    env_content += f"# Format: protocol://user:password@host:port/database\n"
                elif auth_type == "cookie":
                    env_content += f"# Extract from browser cookies (see provider docs)\n"
                
                if setup_url:
                    env_content += f"# Setup guide: {setup_url}\n"
                
                if example_value:
                    env_content += f'{env_var}={example_value}\n\n'
                else:
                    env_content += f'{env_var}=your_{env_var.lower()}_here\n\n'
        
        # API Servers
        if self.api_servers:
            env_content += "# ==========================================\n"
            env_content += "# API SERVERS (Direct REST API)\n"
            env_content += "# ==========================================\n\n"
            
            for server in self.api_servers:
                server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
                name = server_dict.get("name", "api")
                env_var = server_dict.get("auth_env_var", "API_KEY")
                auth_type = server_dict.get("auth_type", "bearer")
                description = server_dict.get("description", "")
                setup_url = server_dict.get("setup_url", "")
                example_value = server_dict.get("example_value", "")
                
                env_content += f"# {name.upper()} API\n"
                if description:
                    env_content += f"# {description}\n"
                
                # Add helpful hints based on auth type
                if auth_type == "bearer":
                    env_content += f"# Bearer token for Authorization header\n"
                elif auth_type == "api_key":
                    env_content += f"# API key (added to header)\n"
                elif auth_type == "basic":
                    env_content += f"# Format: username:password (will be base64 encoded)\n"
                
                if setup_url:
                    env_content += f"# Setup guide: {setup_url}\n"
                
                if example_value:
                    env_content += f'{env_var}={example_value}\n\n'
                else:
                    env_content += f'{env_var}=your_{env_var.lower()}_here\n\n'
        
        path = os.path.join(output_dir, ".env.example")
        with open(path, "w") as f:
            f.write(env_content)
        print(f"   📄 .env.example")
    
    def _write_setup_script(self, output_dir: str):
        """Write setup.sh script to validate credentials."""
        env_vars = []
        for server in self.mcp_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            env_vars.append(server_dict.get("env_var", "API_KEY"))
        
        env_checks = "\n".join([
            f'check_env "{var}"' for var in env_vars
        ])
        
        setup_script = f'''#!/bin/bash
# {self.agent_name} - Setup Script
# ================================
# This script validates that all required credentials are set.

set -e

echo "🔧 {self.agent_name} Setup"
echo "========================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo ""
    echo "To get started:"
    echo "  1. cp .env.example .env"
    echo "  2. Edit .env with your credentials"
    echo "  3. Run this script again"
    echo ""
    exit 1
fi

# Load .env
export $(grep -v '^#' .env | xargs)

# Function to check env var
check_env() {{
    local var_name=$1
    local var_value=${{!var_name}}
    
    if [ -z "$var_value" ] || [[ "$var_value" == *"xxxx"* ]] || [[ "$var_value" == *"your_"* ]]; then
        echo "❌ $var_name - Not set or contains placeholder"
        return 1
    else
        echo "✅ $var_name - Set"
        return 0
    fi
}}

echo "Checking required environment variables..."
echo ""

ERRORS=0

{env_checks}

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "❌ Some credentials are missing. Please update .env"
    exit 1
fi

echo ""
echo "✅ All credentials configured!"
echo ""
echo "Next steps:"
echo "  - For Ollama: ollama create {self.agent_name} -f Modelfile"
echo "  - For Python: pip install -r requirements.txt && python agent.py -i"
'''
        
        path = os.path.join(output_dir, "setup.sh")
        with open(path, "w") as f:
            f.write(setup_script)
        os.chmod(path, 0o755)  # Make executable
        print(f"   📄 setup.sh")
    
    def _write_gitignore(self, output_dir: str):
        """Write .gitignore to protect credentials."""
        gitignore = """# Credentials - NEVER COMMIT!
.env
*.env

# Model files (too large for git)
*.gguf
*.bin
*.safetensors
model/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
*.swp
"""
        
        path = os.path.join(output_dir, ".gitignore")
        with open(path, "w") as f:
            f.write(gitignore)
        print(f"   📄 .gitignore")
    
    def _write_ollama_files(self, output_dir: str):
        """Write Ollama-specific files."""
        # Modelfile - always use ./model.gguf since _copy_model() copies it there
        # This is relative to the package directory, not the original training location
        gguf_path = "./model.gguf"
        modelfile = f'''# Ollama Modelfile for {self.agent_name}
# 
# Usage:
#   ollama create {self.agent_name} -f Modelfile
#   ollama run {self.agent_name}

FROM {gguf_path}

SYSTEM """{self.system_prompt}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "</tool_call>"
'''
        
        path = os.path.join(output_dir, "Modelfile")
        with open(path, "w") as f:
            f.write(modelfile)
        print(f"   📄 Modelfile (Ollama)")
    
    def _write_python_files(self, output_dir: str):
        """Write Python runtime files with MCP + API-based tool execution."""
        # Write tools.json (all tool schemas for reference)
        tools_json = json.dumps(
            [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.tools],
            indent=2
        )
        tools_path = os.path.join(output_dir, "tools.json")
        with open(tools_path, "w") as f:
            f.write(tools_json)
        print(f"   📄 tools.json")
        
        # Write api_tools.py if there are API servers
        if self.api_servers:
            self._write_api_tools(output_dir)
        
        # Load agent template
        template_path = os.path.join(os.path.dirname(__file__), "agent_template.py")
        with open(template_path, "r") as f:
            agent_py = f.read()
        
        # Replace placeholders
        agent_py = agent_py.replace("{agent_name}", self.agent_name)
        
        # Add API tools import if needed
        if self.api_servers:
            api_import = "from api_tools import API_TOOLS, call_api_tool"
            agent_py = agent_py.replace(
                "# {api_tools_import}",
                api_import
            )
            agent_py = agent_py.replace("{has_api_tools}", "True")
        else:
            agent_py = agent_py.replace("# {api_tools_import}", "")
            agent_py = agent_py.replace("{has_api_tools}", "False")
        
        # Write agent.py
        agent_path = os.path.join(output_dir, "agent.py")
        with open(agent_path, "w") as f:
            f.write(agent_py)
        os.chmod(agent_path, 0o755)
        print(f"   📄 agent.py (MCP + API)")
        
        # Write requirements.txt
        requirements = '''# Requirements for running the agent
llama-cpp-python>=0.2.0
python-dotenv>=1.0.0
requests>=2.28.0
'''
        req_path = os.path.join(output_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write(requirements)
        print(f"   📄 requirements.txt")
    
    def _write_api_tools(self, output_dir: str):
        """
        Generate api_tools.py with wrapper functions for direct API calls.
        
        For each APIServerConfig, generates:
        1. Auth setup for that service
        2. Function for each API endpoint
        3. Registration in API_TOOLS dict
        """
        code = f'''"""
API Tools - Auto-generated by OnsetLab
======================================
Direct API wrappers for services without MCP servers.

DO NOT EDIT - regenerate by rebuilding the agent.
"""

import os
import json
import requests
from typing import Any, Dict, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

'''
        
        # Generate code for each API server
        for server in self.api_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            server_name = server_dict.get('name', 'api')
            base_url = server_dict.get('base_url', '')
            auth_type = server_dict.get('auth_type', 'bearer')
            auth_env_var = server_dict.get('auth_env_var', 'API_KEY')
            auth_header = server_dict.get('auth_header', 'Authorization')
            tools = server_dict.get('tools', [])
            
            # Server config section
            code += f'''# ============================================
# {server_name.upper()} API
# ============================================

{server_name.upper()}_BASE_URL = "{base_url}"
{server_name.upper()}_TOKEN = os.getenv("{auth_env_var}", "")

def _get_{server_name}_headers() -> Dict[str, str]:
    """Get auth headers for {server_name}."""
'''
            
            # Auth header generation based on type
            if auth_type == "bearer":
                code += f'''    return {{
        "Authorization": f"Bearer {{{server_name.upper()}_TOKEN}}",
        "Content-Type": "application/json"
    }}
'''
            elif auth_type == "api_key":
                code += f'''    return {{
        "{auth_header}": {server_name.upper()}_TOKEN,
        "Content-Type": "application/json"
    }}
'''
            elif auth_type == "basic":
                code += f'''    import base64
    auth = base64.b64encode({server_name.upper()}_TOKEN.encode()).decode()
    return {{
        "Authorization": f"Basic {{auth}}",
        "Content-Type": "application/json"
    }}
'''
            else:  # header or custom
                code += f'''    return {{
        "{auth_header}": {server_name.upper()}_TOKEN,
        "Content-Type": "application/json"
    }}
'''
            
            code += "\n"
            
            # Generate function for each tool
            for tool in tools:
                tool_dict = tool if isinstance(tool, dict) else tool.to_dict()
                tool_name = tool_dict.get('name', 'unknown')
                method = tool_dict.get('method', 'GET').upper()
                path = tool_dict.get('path', '/')
                description = tool_dict.get('description', '')
                params = tool_dict.get('parameters', {})
                required = tool_dict.get('required_params', [])
                request_body = tool_dict.get('request_body_schema', {})
                
                # Build function signature
                param_list = []
                for pname, pinfo in params.items():
                    ptype = pinfo.get('type', 'str')
                    py_type = {'string': 'str', 'integer': 'int', 'boolean': 'bool', 'number': 'float'}.get(ptype, 'Any')
                    if pname in required:
                        param_list.append(f"{pname}: {py_type}")
                    else:
                        default = pinfo.get('default', 'None')
                        if py_type == 'str' and default != 'None':
                            default = f'"{default}"'
                        param_list.append(f"{pname}: Optional[{py_type}] = {default}")
                
                params_str = ", ".join(param_list) if param_list else ""
                
                code += f'''
def {tool_name}({params_str}) -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated API wrapper.
    """
    url = f"{{{server_name.upper()}_BASE_URL}}{path}"
    headers = _get_{server_name}_headers()
    
'''
                
                # Handle path parameters (e.g., /repos/{owner}/{repo})
                path_params = [p.strip('{}') for p in path.split('/') if '{' in p]
                if path_params:
                    code += f'''    # Substitute path parameters
'''
                    for pp in path_params:
                        code += f'''    url = url.replace("{{{{{pp}}}}}", str({pp}))
'''
                    code += "\n"
                
                # Build request based on method
                if method == "GET":
                    # Build query params
                    query_params = [p for p in params.keys() if p not in path_params]
                    if query_params:
                        code += f'''    params = {{}}
'''
                        for qp in query_params:
                            code += f'''    if {qp} is not None:
        params["{qp}"] = {qp}
'''
                        code += f'''
    response = requests.get(url, headers=headers, params=params)
'''
                    else:
                        code += f'''    response = requests.get(url, headers=headers)
'''
                
                elif method in ("POST", "PUT", "PATCH"):
                    # Build request body
                    body_params = list(request_body.keys()) if request_body else [p for p in params.keys() if p not in path_params]
                    if body_params:
                        code += f'''    body = {{}}
'''
                        for bp in body_params:
                            code += f'''    if {bp} is not None:
        body["{bp}"] = {bp}
'''
                        code += f'''
    response = requests.{method.lower()}(url, headers=headers, json=body)
'''
                    else:
                        code += f'''    response = requests.{method.lower()}(url, headers=headers)
'''
                
                elif method == "DELETE":
                    code += f'''    response = requests.delete(url, headers=headers)
'''
                
                # Handle response
                code += f'''
    if response.ok:
        try:
            return response.json()
        except:
            return {{"status": "success", "message": response.text}}
    else:
        return {{"error": response.status_code, "message": response.text}}

'''
        
        # Build the API_TOOLS registry
        code += '''
# ============================================
# API TOOLS REGISTRY
# ============================================

API_TOOLS = {
'''
        for server in self.api_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            for tool in server_dict.get('tools', []):
                tool_dict = tool if isinstance(tool, dict) else tool.to_dict()
                tool_name = tool_dict.get('name', 'unknown')
                code += f'    "{tool_name}": {tool_name},\n'
        
        code += '''}


def call_api_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call an API tool by name with parameters.
    
    Args:
        name: Tool name (must be in API_TOOLS)
        params: Dictionary of parameters
        
    Returns:
        Tool result as dictionary
    """
    if name not in API_TOOLS:
        return {"error": f"Unknown API tool: {name}"}
    
    try:
        return API_TOOLS[name](**params)
    except Exception as e:
        return {"error": str(e)}
'''
        
        # Write the file
        path = os.path.join(output_dir, "api_tools.py")
        with open(path, "w") as f:
            f.write(code)
        print(f"   📄 api_tools.py ({len(self.api_servers)} services, {sum(len(s.tools) if hasattr(s, 'tools') else len(s.get('tools', [])) for s in self.api_servers)} tools)")
    
    def _write_readme(self, output_dir: str):
        """Write README with usage instructions."""
        runtime = self.config.runtime
        
        readme = f'''# {self.agent_name}

AI Agent built with [OnsetLab](https://github.com/your-repo/onsetlab)

## 🚀 Quick Start

### Step 1: Setup Credentials

Before running the agent, you need to configure your MCP server credentials:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or use your preferred editor

# Validate your setup
./setup.sh
```

### Step 2: Run the Agent

'''
        
        if runtime in (RuntimeType.OLLAMA, RuntimeType.BOTH):
            readme += '''#### Option A: Ollama (Recommended)

```bash
# Load environment variables
source .env

# Create the model in Ollama
ollama create {name} -f Modelfile

# Run the agent
ollama run {name}
```

'''.format(name=self.agent_name)
        
        if runtime in (RuntimeType.PYTHON, RuntimeType.BOTH):
            readme += '''#### Option B: Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactively
python agent.py --interactive

# Or with a single query
python agent.py "What's on my calendar today?"
```

'''
        
        readme += '''---

## 🔑 Required Credentials

'''
        for server in self.mcp_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            package = server_dict.get('package', 'unknown')
            env_var = server_dict.get('env_var', 'API_KEY')
            auth_type = server_dict.get('auth_type', 'unknown')
            description = server_dict.get('description', '')
            setup_url = server_dict.get('setup_url', '')  # Use from config!
            
            readme += f"### {package}\n\n"
            if description:
                readme += f"{description}\n\n"
            readme += f"- **Environment Variable**: `{env_var}`\n"
            readme += f"- **Auth Type**: {auth_type}\n"
            
            # Add setup hint from config (no hardcoding!)
            if setup_url:
                readme += f"- **Setup Guide**: [{setup_url}]({setup_url})\n"
            
            readme += "\n"
        
        readme += '''---

## 🔧 Available Tools

'''
        for tool in self.tools:
            tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
            readme += f"- **{tool_dict.get('name', 'unknown')}**: {tool_dict.get('description', '')}\n"
        
        readme += '''
---

## 📁 Files

| File | Description |
|------|-------------|
| `.env.example` | Template for credentials (copy to `.env`) |
| `.env` | Your actual credentials (DO NOT COMMIT!) |
| `setup.sh` | Validates your credential setup |
| `system_prompt.txt` | The agent's system prompt |
| `mcp_config.json` | MCP server configuration |
'''
        
        if runtime in (RuntimeType.OLLAMA, RuntimeType.BOTH):
            readme += "| `Modelfile` | Ollama model definition |\n"
        
        if runtime in (RuntimeType.PYTHON, RuntimeType.BOTH):
            readme += "| `agent.py` | Python runtime script |\n"
            readme += "| `requirements.txt` | Python dependencies |\n"
        
        readme += '''
---

## ⚠️ Security Notes

- **NEVER commit `.env` to version control**
- Add `.env` to your `.gitignore`
- Rotate credentials if accidentally exposed
- Use environment-specific credentials for dev/prod

---

Built with [OnsetLab](https://github.com/your-repo/onsetlab)
'''
        
        path = os.path.join(output_dir, "README.md")
        with open(path, "w") as f:
            f.write(readme)
        print(f"   📄 README.md")
