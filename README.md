# OnsetLab

Make your local SLMs do actual work.

## What

Agent framework that connects small language models (via Ollama) to MCP tools. REWOO architecture with ReAct fallback makes 3-4B models actually usable for tool calling.

## Install

```bash
pip install onsetlab
```

Requires [Ollama](https://ollama.ai) running locally.

## Usage

```python
from onsetlab import Agent, MCPServer

agent = Agent("phi3.5")
agent.add_mcp_server(MCPServer.from_registry("filesystem", path="."))

result = agent.run("List all Python files in this directory")
print(result.answer)
```

## MCP Servers

Built-in registry: `filesystem`, `github`, `slack`, `notion`, `google_calendar`, `tavily`

```python
# With authentication
agent.add_mcp_server(MCPServer.from_registry("github", token="ghp_..."))
```

## Architecture

```
User Query → Planner (1 SLM call) → Executor → Verifier → Solver (1 SLM call)
                                        ↓
                               ReAct fallback if needed
```

## Limitations

- Works best with <20 tools (SLMs struggle with large tool sets)
- Tested with phi3.5, llama3.2:3b
- MCP servers must be installed separately

## License

Apache 2.0
