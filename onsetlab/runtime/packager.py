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
    agent_name: str = "my_agent"
    output_dir: str = "./agent_package"
    include_training_data: bool = False  # Include training data in package
    include_readme: bool = True


class AgentPackager:
    """
    Packages a trained agent for deployment.
    
    Generates:
    - For Ollama: Modelfile, config.yaml
    - For Python: agent.py, requirements.txt
    - Common: README.md, system_prompt.txt, mcp_config.json
    """
    
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        tools: list,
        mcp_servers,  # Can be list or dict
        model_path: str = None,
        config: PackageConfig = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = tools
        # Normalize mcp_servers to list (handles both dict and list input)
        self.mcp_servers = self._normalize_mcp_servers(mcp_servers)
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
        """Write MCP server configuration with tool whitelist."""
        config = {
            "mcpServers": {},
            "allowed_tools": []
        }
        
        # Package name mapping (shorthand -> actual npm package)
        PACKAGE_MAP = {
            "github-mcp-server": "@modelcontextprotocol/server-github",
            "slack-mcp-server": "slack-mcp-server",  # korotovsky's package
            "filesystem-mcp-server": "@modelcontextprotocol/server-filesystem",
        }
        
        # Tool patterns to identify which server they belong to
        GITHUB_TOOLS = {"issue", "pull_request", "repository", "file_contents", "commit", "branch", "fork", "star"}
        SLACK_TOOLS = {"conversations", "channels", "chat", "message", "users", "search_messages"}
        
        # Categorize tools by server
        tool_lists = {name: [] for name in ["github", "slack", "other"]}
        for tool in self.tools:
            tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
            tool_name = tool_dict.get("name", "").lower()
            
            if any(kw in tool_name for kw in GITHUB_TOOLS):
                tool_lists["github"].append(tool_dict.get("name", ""))
            elif any(kw in tool_name for kw in SLACK_TOOLS):
                tool_lists["slack"].append(tool_dict.get("name", ""))
            else:
                tool_lists["other"].append(tool_dict.get("name", ""))
        
        # Build server configs
        for server in self.mcp_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            package = server_dict.get("package", "")
            
            # Get actual npm package name
            actual_package = PACKAGE_MAP.get(package, package)
            
            # Determine server name
            name = package.split("/")[-1].replace("-mcp-server", "").replace("-mcp", "").replace("server-", "")
            if "github" in package.lower():
                name = "github"
            elif "slack" in package.lower():
                name = "slack"
            
            # Get tools for this specific server
            server_tools = tool_lists.get(name, [])
            if not server_tools:
                # Fallback: include "other" tools
                server_tools = tool_lists.get("other", [])
            
            config["mcpServers"][name] = {
                "command": "npx",
                "args": ["-y", actual_package],
                "env": {
                    server_dict.get("env_var", "API_KEY"): f"${{{server_dict.get('env_var', 'API_KEY')}}}"
                },
                "tools": server_tools
            }
        
        # Collect all allowed tools (only tools we trained on)
        for tool in self.tools:
            tool_dict = tool.to_dict() if hasattr(tool, 'to_dict') else tool
            config["allowed_tools"].append(tool_dict.get("name", ""))
        
        path = os.path.join(output_dir, "mcp_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"   📄 mcp_config.json")
    
    def _write_env_example(self, output_dir: str):
        """Write .env.example file with all required credentials."""
        env_content = f"""# {self.agent_name} - Environment Variables
# ============================================
# Copy this file to .env and fill in your credentials:
#   cp .env.example .env
#
# Then edit .env with your actual values.
# NEVER commit .env to version control!
# ============================================

"""
        for server in self.mcp_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            package = server_dict.get("package", "unknown")
            env_var = server_dict.get("env_var", "API_KEY")
            auth_type = server_dict.get("auth_type", "token")
            description = server_dict.get("description", "")
            setup_url = server_dict.get("setup_url", "")
            
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
            
            if setup_url:
                env_content += f"# Setup guide: {setup_url}\n"
            
            # Add example value
            if "github" in package.lower():
                env_content += f'{env_var}=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n'
            elif "google" in package.lower() or "oauth" in auth_type:
                env_content += f'{env_var}=/path/to/credentials.json\n\n'
            elif "postgres" in package.lower():
                env_content += f'{env_var}=postgresql://user:password@localhost:5432/mydb\n\n'
            elif "slack" in package.lower():
                env_content += f'{env_var}=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx\n\n'
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
        # Modelfile
        gguf_path = self.model_path or "./model.gguf"
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
        """Write Python runtime files with MCP-based tool execution."""
        # Write tools.json
        tools_json = json.dumps(
            [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.tools],
            indent=2
        )
        tools_path = os.path.join(output_dir, "tools.json")
        with open(tools_path, "w") as f:
            f.write(tools_json)
        print(f"   📄 tools.json")
        
        # Load agent template
        template_path = os.path.join(os.path.dirname(__file__), "agent_template.py")
        with open(template_path, "r") as f:
            agent_py = f.read()
        
        # Replace placeholders
        agent_py = agent_py.replace("{agent_name}", self.agent_name)
        
        # Write agent.py
        agent_path = os.path.join(output_dir, "agent.py")
        with open(agent_path, "w") as f:
            f.write(agent_py)
        os.chmod(agent_path, 0o755)
        print(f"   📄 agent.py (MCP-based)")
        
        # Write requirements.txt
        requirements = '''# Requirements for running the agent
llama-cpp-python>=0.2.0
python-dotenv>=1.0.0
'''
        req_path = os.path.join(output_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write(requirements)
        print(f"   📄 requirements.txt")
    
    def _write_python_files_OLD_UNUSED(self, output_dir: str):
        """OLD METHOD - NOT USED. Kept for reference."""
        agent_py = f'''#!/usr/bin/env python3
"""
{self.agent_name} - AI Agent with Real Tool Execution
======================================================
Auto-generated by OnsetLab

Usage:
    python agent.py --interactive
    python agent.py "Show me open issues in facebook/react"
"""

import argparse
import json
import os
import re
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
AGENT_NAME = "{self.agent_name}"
MODEL_PATH = "./model"
SYSTEM_PROMPT = """{self.system_prompt}"""

TOOLS = {tools_json}


# ============================================================
# TOOL EXECUTORS - Real API calls to GitHub, Slack, etc.
# ============================================================

import requests

def execute_tool(tool_name: str, params: dict) -> str:
    """Execute a tool and return the result."""
    
    # GitHub Tools
    if tool_name == "list_issues":
        return github_list_issues(params)
    elif tool_name == "get_issue":
        return github_get_issue(params)
    elif tool_name == "create_issue":
        return github_create_issue(params)
    elif tool_name == "add_issue_comment":
        return github_add_comment(params)
    elif tool_name == "search_repositories":
        return github_search_repos(params)
    elif tool_name == "get_file_contents":
        return github_get_file(params)
    elif tool_name == "create_pull_request":
        return github_create_pr(params)
    elif tool_name == "list_commits":
        return github_list_commits(params)
    
    # Slack Tools
    elif tool_name == "post_message":
        return slack_post_message(params)
    elif tool_name == "list_channels":
        return slack_list_channels(params)
    
    else:
        return f"Unknown tool: {{tool_name}}"


# ============================================================
# GitHub API Functions
# ============================================================

def github_headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return {{"error": "GITHUB_TOKEN not set in .env"}}
    return {{"Authorization": f"token {{token}}", "Accept": "application/vnd.github.v3+json"}}

def github_list_issues(params):
    owner = params.get("owner")
    repo = params.get("repo")
    state = params.get("state", "open")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/issues"
    resp = requests.get(url, headers=github_headers(), params={{"state": state, "per_page": 10}})
    
    if resp.status_code != 200:
        return f"Error: {{resp.status_code}} - {{resp.text[:200]}}"
    
    issues = resp.json()
    result = f"Found {{len(issues)}} issues in {{owner}}/{{repo}}:\\n"
    for i in issues[:5]:
        result += f"  #{{i['number']}} {{i['title']}} [{{i['state']}}]\\n"
    return result

def github_get_issue(params):
    owner = params.get("owner")
    repo = params.get("repo")
    issue_number = params.get("issue_number")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/issues/{{issue_number}}"
    resp = requests.get(url, headers=github_headers())
    
    if resp.status_code != 200:
        return f"Error: {{resp.status_code}}"
    
    issue = resp.json()
    return f"Issue #{{issue['number']}}: {{issue['title']}}\\nState: {{issue['state']}}\\nBody: {{issue['body'][:300] if issue['body'] else 'No description'}}"

def github_create_issue(params):
    owner = params.get("owner")
    repo = params.get("repo")
    title = params.get("title")
    body = params.get("body", "")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/issues"
    resp = requests.post(url, headers=github_headers(), json={{"title": title, "body": body}})
    
    if resp.status_code != 201:
        return f"Error creating issue: {{resp.status_code}} - {{resp.text[:200]}}"
    
    issue = resp.json()
    return f"Created issue #{{issue['number']}}: {{issue['html_url']}}"

def github_add_comment(params):
    owner = params.get("owner")
    repo = params.get("repo")
    issue_number = params.get("issue_number")
    body = params.get("body")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/issues/{{issue_number}}/comments"
    resp = requests.post(url, headers=github_headers(), json={{"body": body}})
    
    if resp.status_code != 201:
        return f"Error: {{resp.status_code}}"
    
    return f"Comment added to issue #{{issue_number}}"

def github_search_repos(params):
    query = params.get("query")
    sort = params.get("sort", "stars")
    
    url = "https://api.github.com/search/repositories"
    resp = requests.get(url, headers=github_headers(), params={{"q": query, "sort": sort, "per_page": 5}})
    
    if resp.status_code != 200:
        return f"Error: {{resp.status_code}}"
    
    repos = resp.json().get("items", [])
    result = f"Found {{len(repos)}} repositories:\\n"
    for r in repos:
        result += f"  ⭐ {{r['stargazers_count']:,}} - {{r['full_name']}}: {{r['description'][:50] if r['description'] else ''}}...\\n"
    return result

def github_get_file(params):
    owner = params.get("owner")
    repo = params.get("repo")
    path = params.get("path")
    branch = params.get("branch", "main")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/contents/{{path}}"
    resp = requests.get(url, headers=github_headers(), params={{"ref": branch}})
    
    if resp.status_code != 200:
        return f"Error: {{resp.status_code}}"
    
    import base64
    content = base64.b64decode(resp.json().get("content", "")).decode("utf-8")
    return f"File {{path}}:\\n{{content[:500]}}..."

def github_create_pr(params):
    owner = params.get("owner")
    repo = params.get("repo")
    title = params.get("title")
    body = params.get("body", "")
    head = params.get("head")
    base = params.get("base", "main")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/pulls"
    resp = requests.post(url, headers=github_headers(), json={{
        "title": title, "body": body, "head": head, "base": base
    }})
    
    if resp.status_code != 201:
        return f"Error: {{resp.status_code}} - {{resp.text[:200]}}"
    
    pr = resp.json()
    return f"Created PR #{{pr['number']}}: {{pr['html_url']}}"

def github_list_commits(params):
    owner = params.get("owner")
    repo = params.get("repo")
    sha = params.get("sha", "main")
    
    url = f"https://api.github.com/repos/{{owner}}/{{repo}}/commits"
    resp = requests.get(url, headers=github_headers(), params={{"sha": sha, "per_page": 5}})
    
    if resp.status_code != 200:
        return f"Error: {{resp.status_code}}"
    
    commits = resp.json()
    result = "Recent commits:\\n"
    for c in commits:
        msg = c["commit"]["message"].split("\\n")[0][:50]
        result += f"  {{c['sha'][:7]}} - {{msg}}\\n"
    return result


# ============================================================
# Slack API Functions
# ============================================================

def slack_headers():
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return {{"error": "SLACK_BOT_TOKEN not set in .env"}}
    return {{"Authorization": f"Bearer {{token}}", "Content-Type": "application/json"}}

def slack_post_message(params):
    channel = params.get("channel", "").lstrip("#")
    text = params.get("text")
    
    url = "https://slack.com/api/chat.postMessage"
    resp = requests.post(url, headers=slack_headers(), json={{"channel": channel, "text": text}})
    
    data = resp.json()
    if not data.get("ok"):
        return f"Error: {{data.get('error', 'Unknown error')}}"
    
    return f"Message posted to #{{channel}}"

def slack_list_channels(params):
    url = "https://slack.com/api/conversations.list"
    resp = requests.get(url, headers=slack_headers(), params={{"types": "public_channel", "limit": 20}})
    
    data = resp.json()
    if not data.get("ok"):
        return f"Error: {{data.get('error', 'Unknown error')}}"
    
    channels = data.get("channels", [])
    result = f"Found {{len(channels)}} channels:\\n"
    for c in channels[:10]:
        result += f"  #{{c['name']}}\\n"
    return result


# ============================================================
# Model Loading & Generation
# ============================================================

def load_model():
    """Load the GGUF model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed.")
        print("Run: pip install llama-cpp-python")
        sys.exit(1)
    
    # Find GGUF file
    gguf_file = None
    model_dir = "./model"
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith(".gguf"):
                gguf_file = os.path.join(model_dir, f)
                break
    
    if not gguf_file:
        print("Error: No .gguf file found in ./model/")
        sys.exit(1)
    
    print(f"Loading {{gguf_file}}...")
    model = Llama(
        model_path=gguf_file,
        n_ctx=4096,
        n_gpu_layers=-1,  # Use GPU if available
        verbose=False,
    )
    return model


def generate_response(model, messages: list) -> str:
    """Generate a response using the model."""
    # Format messages for Qwen chat template
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|im_start|>system\\n{{content}}<|im_end|>\\n"
        elif role == "user":
            prompt += f"<|im_start|>user\\n{{content}}<|im_end|>\\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\\n{{content}}<|im_end|>\\n"
        elif role == "tool":
            prompt += f"<|im_start|>tool\\n{{content}}<|im_end|>\\n"
    
    prompt += "<|im_start|>assistant\\n"
    
    output = model(prompt, max_tokens=512, temperature=0.3, stop=["<|im_end|>", "</tool_call>"])
    response = output["choices"][0]["text"]
    
    # If stopped at </tool_call>, add it back
    if "<tool_call>" in response and "</tool_call>" not in response:
        response += "</tool_call>"
    
    return response.strip()


def parse_tool_call(response: str) -> dict:
    """Parse tool call from response."""
    if "<tool_call>" in response:
        match = re.search(r"<tool_call>\\s*(\\{{.*?\\}})\\s*", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ============================================================
# Main Agent Loop
# ============================================================

def run_agent(model, query: str) -> str:
    """Run the full agentic loop with tool execution."""
    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{"role": "user", "content": query}},
    ]
    
    max_iterations = 5
    for i in range(max_iterations):
        response = generate_response(model, messages)
        tool_call = parse_tool_call(response)
        
        if tool_call:
            tool_name = tool_call.get("tool")
            params = tool_call.get("parameters", {{}})
            
            print(f"\\n🔧 Calling {{tool_name}}...")
            result = execute_tool(tool_name, params)
            print(f"📋 Result: {{result[:200]}}...")
            
            # Add to conversation
            messages.append({{"role": "assistant", "content": response}})
            messages.append({{"role": "tool", "content": result}})
        else:
            # No tool call - final response
            return response
    
    return "Max iterations reached."


def interactive_mode(model):
    """Interactive chat with real tool execution."""
    print(f"\\n🤖 {{AGENT_NAME}} Ready!")
    print("Tools will be executed with REAL APIs (GitHub, Slack)")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            query = input("\\nYou: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue
            
            response = run_agent(model, query)
            print(f"\\n🤖 Agent: {{response}}")
                
        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description=f"{{AGENT_NAME}} - AI Agent")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    # Check credentials
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️  GITHUB_TOKEN not set in .env")
    if not os.getenv("SLACK_BOT_TOKEN"):
        print("⚠️  SLACK_BOT_TOKEN not set in .env")
    
    model = load_model()
    
    if args.interactive or not args.query:
        interactive_mode(model)
    else:
        response = run_agent(model, args.query)
        print(response)


if __name__ == "__main__":
    main()
'''
        
        path = os.path.join(output_dir, "agent.py")
        with open(path, "w") as f:
            f.write(agent_py)
        os.chmod(path, 0o755)  # Make executable
        print(f"   📄 agent.py (Python)")
        
        # Also write tools.json for reference
        path = os.path.join(output_dir, "tools.json")
        with open(path, "w") as f:
            f.write(tools_json)
        print(f"   📄 tools.json")
        
        # requirements.txt
        requirements = '''# Requirements for running the agent
llama-cpp-python>=0.2.0
requests>=2.28.0
python-dotenv>=1.0.0
'''
        
        path = os.path.join(output_dir, "requirements.txt")
        with open(path, "w") as f:
            f.write(requirements)
        print(f"   📄 requirements.txt")
    
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
            
            readme += f"### {package}\n\n"
            if description:
                readme += f"{description}\n\n"
            readme += f"- **Environment Variable**: `{env_var}`\n"
            readme += f"- **Auth Type**: {auth_type}\n"
            
            # Add setup hints
            if "github" in package.lower():
                readme += f"- **Get Token**: [GitHub Settings → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens)\n"
            elif "google" in package.lower():
                readme += f"- **Setup Guide**: [Google Cloud Console](https://console.cloud.google.com/apis/credentials)\n"
            elif "postgres" in package.lower():
                readme += f"- **Format**: `postgresql://user:password@host:5432/database`\n"
            elif "slack" in package.lower():
                readme += f"- **Get Token**: [Slack API Apps](https://api.slack.com/apps)\n"
            
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
