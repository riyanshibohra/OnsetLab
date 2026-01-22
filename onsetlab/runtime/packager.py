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
    
    def _write_system_prompt(self, output_dir: str):
        """Write system prompt file."""
        path = os.path.join(output_dir, "system_prompt.txt")
        with open(path, "w") as f:
            f.write(self.system_prompt)
        print(f"   📄 system_prompt.txt")
    
    def _write_mcp_config(self, output_dir: str):
        """Write MCP server configuration."""
        config = {
            "mcpServers": {}
        }
        
        for server in self.mcp_servers:
            server_dict = server.to_dict() if hasattr(server, 'to_dict') else server
            name = server_dict.get("package", "").split("/")[-1].replace("-mcp", "")
            config["mcpServers"][name] = {
                "command": "npx",
                "args": ["-y", server_dict.get("package", "")],
                "env": {
                    server_dict.get("env_var", "API_KEY"): f"${{{server_dict.get('env_var', 'API_KEY')}}}"
                }
            }
        
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
        """Write Python runtime files."""
        # agent.py
        tools_json = json.dumps(
            [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.tools],
            indent=2
        )
        
        agent_py = f'''#!/usr/bin/env python3
"""
{self.agent_name} - AI Agent
============================
Auto-generated by OnsetLab

Usage:
    python agent.py "What's on my calendar today?"
    python agent.py --interactive
"""

import argparse
import json
import os
import sys

# Configuration
AGENT_NAME = "{self.agent_name}"
MODEL_PATH = "{self.model_path or './model'}"
SYSTEM_PROMPT = """{self.system_prompt}"""

TOOLS = {tools_json}


def load_model():
    """Load the fine-tuned model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("Error: transformers not installed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print(f"Loading model from {{MODEL_PATH}}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    return model, tokenizer


def generate_response(model, tokenizer, query: str) -> str:
    """Generate a response for the given query."""
    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{"role": "user", "content": query}},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant" in response.lower():
        parts = response.split("assistant")
        response = parts[-1].strip()
    
    return response


def parse_tool_call(response: str) -> dict:
    """Parse tool call from response if present."""
    if "<tool_call>" in response:
        start = response.find("<tool_call>") + len("<tool_call>")
        end = response.find("</tool_call>")
        if end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass
    return None


def interactive_mode(model, tokenizer):
    """Run in interactive mode."""
    print(f"\\n🤖 {{AGENT_NAME}} Ready!")
    print("Type your message (or 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            query = input("\\nYou: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue
            
            response = generate_response(model, tokenizer, query)
            tool_call = parse_tool_call(response)
            
            if tool_call:
                print(f"\\n🔧 Tool Call: {{tool_call['name']}}")
                print(f"   Args: {{json.dumps(tool_call.get('arguments', {{}}), indent=2)}}")
            else:
                print(f"\\nAgent: {{response}}")
                
        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description=f"{{AGENT_NAME}} - AI Agent")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    model, tokenizer = load_model()
    
    if args.interactive or not args.query:
        interactive_mode(model, tokenizer)
    else:
        response = generate_response(model, tokenizer, args.query)
        tool_call = parse_tool_call(response)
        
        if tool_call:
            print(json.dumps(tool_call, indent=2))
        else:
            print(response)


if __name__ == "__main__":
    main()
'''
        
        path = os.path.join(output_dir, "agent.py")
        with open(path, "w") as f:
            f.write(agent_py)
        os.chmod(path, 0o755)  # Make executable
        print(f"   📄 agent.py (Python)")
        
        # requirements.txt
        requirements = '''# Requirements for running the agent
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
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
