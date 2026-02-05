# OnsetLab

Make your local SLMs do actual work.

## What

Agent framework for small language models (3B parameters) that achieves GPT-4 level tool use on your laptop. Uses REWOO architecture (plan → execute → verify) with ReAct fallback for robustness.

- **No API keys** - runs entirely on Ollama
- **No fine-tuning** - works out of the box with any Ollama model
- **MCP support** - connect to any MCP-compatible server

## Install

```bash
pip install onsetlab
```

Requires [Ollama](https://ollama.ai) running locally.

## Quick Start

```python
from onsetlab import Agent

agent = Agent("phi3.5")
result = agent.run("What time is it in Tokyo?")
print(result.answer)
```

## Adding Tools

Choose only the tools you need:

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime, UnitConverter, TextProcessor, RandomGenerator

agent = Agent(
    model="phi3.5",
    tools=[Calculator(), DateTime()]
)

result = agent.run("What's 15% tip on $84.50?")
print(result.answer)  # "The 15% tip on $84.50 is $12.68"
```

**Available built-in tools:**
- `Calculator` - math operations
- `DateTime` - current time, timezones, date math
- `UnitConverter` - convert between units
- `TextProcessor` - word count, search, transform text
- `RandomGenerator` - random numbers, strings, choices

## MCP Servers

Connect to external services via Model Context Protocol:

```python
from onsetlab import Agent, MCPServer

agent = Agent("phi3.5")

# Add filesystem access
agent.add_mcp_server(MCPServer.from_registry("filesystem", path="/my/project"))

# Add GitHub integration
agent.add_mcp_server(MCPServer.from_registry("github", token="ghp_..."))

result = agent.run("List all Python files in the project")
print(result.answer)
```

**Registry servers:** `filesystem`, `github`, `slack`, `notion`, `google_calendar`, `tavily`

Or configure any MCP server manually:

```python
server = MCPServer(
    name="my-server",
    command="npx",
    args=["-y", "@some/mcp-server"],
    env={"API_KEY": "..."}
)
agent.add_mcp_server(server)
```

## Architecture

```
Task → Planner → Executor → Verifier → Solver → Answer
         ↓          ↓
      REWOO     (ReAct fallback if needed)
```

1. **Planner** - creates step-by-step execution plan
2. **Verifier** - validates plan before execution
3. **Executor** - runs tools with dependency resolution
4. **Solver** - synthesizes final answer from results
5. **ReAct fallback** - recovers from planning failures

## Options

```python
agent = Agent(
    model="phi3.5",           # any Ollama model
    tools=[...],              # built-in tools
    mcp_servers=[...],        # MCP servers
    memory=True,              # conversation memory
    verify=True,              # pre-execution verification
    react_fallback=True,      # fallback on REWOO failure
    debug=False               # verbose logging
)
```

## License

Apache 2.0
