<div align="center">

# OnsetLab

### Tool-calling AI agents that run locally.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/onsetlab.svg)](https://pypi.org/project/onsetlab/)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-000000.svg)](https://ollama.com)

[Quick Start](#quick-start) · [Architecture](#architecture) · [MCP Servers](#mcp-servers) · [CLI](#cli) · [Models](#tested-models) · [Docs](https://onsetlab.app/docs)

</div>

---

Local models are fast, free, and private. But ask one to call a tool and it falls apart. Wrong function names, broken parameters, infinite loops.

**The models are capable. The framework wasn't.**

OnsetLab makes 3B-7B models do reliable tool calling through a hybrid REWOO/ReAct architecture. The framework handles planning, execution, and error recovery. The model only does what it's good at: one step at a time.

---

## Quick Start

```bash
pip install onsetlab
```

Requires [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull phi3.5
```

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime

agent = Agent("phi3.5", tools=[Calculator(), DateTime()])

result = agent.run("What's 15% tip on $84.50?")
print(result.answer)
```

The agent routes the query, builds an execution plan, calls the right tool, and returns the answer. No prompt engineering required.

---

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/riyanshibohra/OnsetLab/main/assets/architecture.png" alt="OnsetLab Architecture" width="600" />
</p>

The **Router** classifies queries as tool-needed or direct-answer using the model itself. The **Planner** generates structured `THINK -> PLAN` steps with auto-generated tool rules from JSON schemas. The **Executor** resolves dependencies and runs tools in order. If planning fails, the **ReAct Fallback** switches to iterative `Thought -> Action -> Observation` loops to recover.

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `Calculator` | Math expressions, percentages, sqrt/sin/log |
| `DateTime` | Current time, timezones, date math, day of week |
| `UnitConverter` | Length, weight, temperature, volume, speed, data |
| `TextProcessor` | Word count, find/replace, case transforms, pattern extraction |
| `RandomGenerator` | Random numbers, UUIDs, passwords, dice rolls, coin flips |

> More tools will be added over time.

## MCP Servers

Connect any [MCP-compatible](https://modelcontextprotocol.io) server to give your agent access to external tools like GitHub, Slack, Notion, and more.

```python
from onsetlab import Agent, MCPServer

server = MCPServer.from_registry("filesystem", extra_args=["/path/to/dir"])

agent = Agent("phi3.5")
agent.add_mcp_server(server)

result = agent.run("List all Python files in the directory")
print(result.answer)

agent.disconnect_mcp_servers()
```

Any MCP server available via npm works too. See the [docs](https://onsetlab.app/docs#mcp-servers) for examples.

**Built-in registry:** `filesystem` · `github` · `slack` · `notion` · `google_calendar` · `tavily`

---

## CLI

```bash
python -m onsetlab                                          # interactive chat
python -m onsetlab --model qwen2.5:7b                       # specify model
python -m onsetlab benchmark --model phi3.5 --verbose        # validate a model
python -m onsetlab benchmark --compare phi3.5,qwen2.5:7b    # compare models
python -m onsetlab export --format docker -o ./my-agent      # export as Docker
python -m onsetlab export --format config -o agent.yaml      # export as YAML
```

Export formats: **YAML** (portable config), **Docker** (Dockerfile + compose + Ollama), **vLLM** (GPU-accelerated), **Script** (standalone .py file). See [Export & Deploy docs](https://onsetlab.app/docs#export-deploy) for details.

---

## Tested Models

| Model | Size | RAM | Notes |
|-------|------|-----|-------|
| `phi3.5` | 3.8B | 4GB+ | Default. Good balance of speed and quality |
| `qwen2.5:3b` | 3B | 4GB+ | Fast, good for simple tasks |
| `qwen2.5:7b` | 7B | 8GB+ | Strong tool calling |
| `qwen3-a3b` | MoE, 3B active | 16GB+ | Best tool calling accuracy |
| `llama3.2:3b` | 3B | 4GB+ | General purpose |

Works with any Ollama model. Run `python -m onsetlab benchmark --model your-model` to verify.

---

<div align="center">

[Website](https://onsetlab.app) · [Playground](https://onsetlab.app/playground) · [Documentation](https://onsetlab.app/docs) · [PyPI](https://pypi.org/project/onsetlab/)

Apache 2.0

</div>
