"""
OnsetLab Meta-Agent - Quick Gradio UI
=====================================
A simple web interface for testing the meta-agent.

Run with:
    python gradio_ui.py
"""

import gradio as gr
import httpx
import json
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# API endpoint
API_URL = "http://localhost:8000/api/generate-agent"


def generate_agent(
    problem_statement: str,
    anthropic_api_key: str,
    tavily_api_key: str,
):
    """Call the meta-agent API and return results."""
    
    if not problem_statement.strip():
        return "❌ Please enter a problem statement", "", "", ""
    
    if not anthropic_api_key.strip():
        return "❌ Anthropic API key is required", "", "", ""
    
    if not tavily_api_key.strip():
        return "❌ Tavily API key is required", "", "", ""
    
    try:
        # Call the API
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                API_URL,
                json={
                    "problem_statement": problem_statement,
                    "anthropic_api_key": anthropic_api_key,
                    "tavily_api_key": tavily_api_key,
                    "upload_to_gist": False,
                }
            )
        
        if response.status_code != 200:
            return f"❌ API Error: {response.text}", "", "", ""
        
        result = response.json()
        
        # Format summary
        mcp_count = len(result.get("mcp_servers", []))
        api_count = len(result.get("api_servers", []))
        tool_count = result.get("tool_count", 0)
        
        summary = f"""## ✅ Agent Generated!

### Discovery Results
- **MCP Servers**: {mcp_count}
- **API Fallbacks**: {api_count}
- **Total Tools**: {tool_count}

### MCP Servers (with protocol support)
"""
        for server in result.get("mcp_servers", []):
            summary += f"- **{server['service']}**: `{server['package']}` ({len(server.get('tools', []))} tools)\n"
        
        if result.get("api_servers"):
            summary += "\n### API Servers (fallback - generated code)\n"
            for api in result.get("api_servers", []):
                summary += f"- **{api['service']}**: {api.get('base_url', 'N/A')} ({len(api.get('endpoints', []))} endpoints)\n"
        
        if result.get("errors"):
            summary += "\n### ⚠️ Warnings\n"
            for error in result["errors"]:
                summary += f"- {error}\n"
        
        # Tools list
        tools_md = "## Discovered Tools\n\n"
        for i, server in enumerate(result.get("mcp_servers", [])):
            tools_md += f"### {server['service']} (MCP)\n"
            for tool in server.get("tools", [])[:10]:
                tools_md += f"- `{tool.get('name', '?')}`: {tool.get('description', '')[:60]}\n"
            if len(server.get("tools", [])) > 10:
                tools_md += f"- ... and {len(server['tools']) - 10} more\n"
            tools_md += "\n"
        
        for api in result.get("api_servers", []):
            tools_md += f"### {api['service']} (API)\n"
            for ep in api.get("endpoints", [])[:10]:
                tools_md += f"- `{ep.get('name', '?')}`: {ep.get('method', 'GET')} {ep.get('path', '/')}\n"
            tools_md += "\n"
        
        # Token guides
        guides_md = "## Setup Guides\n\n"
        for guide in result.get("token_guides", []):
            guides_md += f"### {guide['service']}\n"
            guides_md += f"**Auth Type**: {guide['auth_type']}\n"
            guides_md += f"**Env Variable**: `{guide['env_var']}`\n\n"
            for step in guide.get("steps", []):
                guides_md += f"{step}\n"
            guides_md += "\n---\n\n"
        
        # Notebook JSON (pretty)
        notebook_json = result.get("colab_notebook", "")
        if notebook_json:
            try:
                notebook_obj = json.loads(notebook_json)
                notebook_display = f"**Notebook generated!** ({len(notebook_obj.get('cells', []))} cells)\n\nDownload the JSON below to open in Colab."
            except:
                notebook_display = "Notebook generated (invalid JSON)"
        else:
            notebook_display = "No notebook generated"
        
        return summary, tools_md, guides_md, notebook_json
        
    except httpx.ConnectError:
        return "❌ Cannot connect to API. Make sure the server is running:\n\n```\npython -m meta_agent.api.server\n```", "", "", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


# Create Gradio interface
with gr.Blocks(title="OnsetLab Meta-Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 OnsetLab Meta-Agent
    
    Describe your agent and we'll discover the right MCP servers and tools for you!
    
    **How it works:**
    1. Enter what you want your agent to do
    2. We search for MCP servers (preferred) or fall back to REST APIs
    3. Get a Colab notebook to train and deploy your agent
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            problem_input = gr.Textbox(
                label="What should your agent do?",
                placeholder="I need an agent that manages GitHub issues and sends Slack notifications when bugs are reported",
                lines=3,
            )
            
            with gr.Row():
                anthropic_key = gr.Textbox(
                    label="Anthropic API Key",
                    type="password",
                    value=os.getenv("ANTHROPIC_API_KEY", ""),
                    placeholder="sk-ant-...",
                )
                tavily_key = gr.Textbox(
                    label="Tavily API Key",
                    type="password",
                    value=os.getenv("TAVILY_API_KEY", ""),
                    placeholder="tvly-...",
                )
            
            generate_btn = gr.Button("🚀 Generate Agent", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            summary_output = gr.Markdown(label="Summary")
    
    with gr.Tabs():
        with gr.Tab("🔧 Tools Discovered"):
            tools_output = gr.Markdown()
        
        with gr.Tab("📚 Setup Guides"):
            guides_output = gr.Markdown()
        
        with gr.Tab("📓 Notebook JSON"):
            notebook_output = gr.Code(language="json", label="Colab Notebook (copy and save as .ipynb)")
    
    # Connect button
    generate_btn.click(
        fn=generate_agent,
        inputs=[problem_input, anthropic_key, tavily_key],
        outputs=[summary_output, tools_output, guides_output, notebook_output],
    )
    
    # Example problems
    gr.Examples(
        examples=[
            ["I need an agent that manages GitHub issues and sends Slack notifications"],
            ["Build an agent that checks my Google Calendar and sends email reminders"],
            ["Create an assistant that can search Notion and post to Discord"],
            ["I want an agent that monitors Linear tasks and updates Trello boards"],
        ],
        inputs=[problem_input],
    )


if __name__ == "__main__":
    print("🚀 Starting Gradio UI...")
    print("📝 Make sure the API server is running: python -m meta_agent.api.server")
    demo.launch(share=False, server_port=7860)
