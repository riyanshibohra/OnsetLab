<div align="center">

# OnsetLab

### Tool-calling AI agents that run locally.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/onsetlab.svg)](https://pypi.org/project/onsetlab/)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-000000.svg)](https://ollama.com)

[Install](#install) · [Quick Start](#quick-start) · [Architecture](#architecture) · [MCP Servers](#mcp-servers) · [CLI](#cli) · [Export & Deploy](#export--deploy)

</div>

---

## The Problem

Building AI agents today means paying per API call, sending every request to someone else's server, and hoping the network stays up.

Local models are fast, free, and private. But ask one to call a tool and it falls apart. Wrong function names, broken parameters, infinite loops.

**The models are capable. The framework wasn't.**

## The Solution

OnsetLab makes 3B-7B models do reliable tool calling through a hybrid REWOO/ReAct architecture. The framework handles planning, execution, and error recovery. The model only does what it's good at: one step at a time.

```
pip install onsetlab  →  connect Ollama  →  add tools  →  run
```

No API keys. No fine-tuning. No cloud.

---

## Install

```bash
pip install onsetlab
```

Requires [Ollama](https://ollama.com) running locally:

```bash
ollama pull phi3.5
```

## Quick Start

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime

agent = Agent("phi3.5", tools=[Calculator(), DateTime()])

result = agent.run("What's 15% tip on $84.50?")
print(result.answer)
```

The agent routes the query, builds an execution plan, calls the calculator, and returns the answer. No prompt engineering required.

---

## Architecture

```mermaid
flowchart TD
    Q["Query"] --> R["Router"]

    R -->|"tools needed"| P["Planner"]
    R -->|"no tools"| D["Direct Answer"]

    P --> E["Executor"]
    E --> S["Solver"]

    P -. "plan fails" .-> RE["ReAct Fallback"]

    D --> A["Answer"]
    S --> A
    RE --> A

    style Q fill:#4a6670,stroke:#4a6670,color:#fff
    style R fill:#fff,stroke:#4a6670,color:#2d3b40
    style P fill:#e8f0fe,stroke:#7aa2f7,color:#3b5998
    style E fill:#e8f0fe,stroke:#7aa2f7,color:#3b5998
    style S fill:#e8f0fe,stroke:#7aa2f7,color:#3b5998
    style D fill:#edf7ef,stroke:#9ece6a,color:#2d6a2e
    style RE fill:#fdf4e7,stroke:#e0af68,color:#8a6914
    style A fill:#4a6670,stroke:#4a6670,color:#fff
```

### Router

The model itself decides the strategy. No regex, no keyword matching. The SLM reads the query and available tools, then classifies: `REWOO` (needs tools) or `DIRECT` (answer from knowledge). Trivial greetings are caught before the model is even called.

### Planner

Generates a structured `THINK → PLAN` output. Each plan step specifies a tool, parameters, and dependencies on previous steps. Tool rules are auto-generated from JSON schemas, so the model sees exactly what each tool can do.

### Executor

Resolves dependencies between steps and runs tool calls in order. If step 2 depends on step 1's output, the result is substituted automatically.

### ReAct Fallback

If REWOO planning fails (bad format, wrong tool, missing params), the agent switches to iterative `Thought → Action → Observation` loops. Catches edge cases that structured planning misses.

---

## MCP Servers

Connect any [MCP-compatible](https://modelcontextprotocol.io) server to give your agent access to external tools.

**From the built-in registry:**

```python
from onsetlab import Agent, MCPServer

server = MCPServer.from_registry("filesystem", extra_args=["/path/to/dir"])

agent = Agent("phi3.5")
agent.add_mcp_server(server)

result = agent.run("List all Python files in the directory")
print(result.answer)

agent.disconnect_mcp_servers()
```

**Any MCP server from npm:**

```python
# Web search (requires API key)
server = MCPServer(
    name="tavily",
    command="npx",
    args=["-y", "tavily-mcp@latest"],
    env={"TAVILY_API_KEY": "..."}
)

# Fetch web pages (no key needed)
server = MCPServer(
    name="fetch",
    command="npx",
    args=["-y", "@tokenizin/mcp-npx-fetch"],
)

agent.add_mcp_server(server)
```

**Built-in registry:** `filesystem` · `github` · `slack` · `notion` · `google_calendar` · `tavily`

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

## CLI

OnsetLab includes a command-line interface for interactive chat, benchmarking, and exporting.

```bash
# Interactive chat
python -m onsetlab
python -m onsetlab --model qwen2.5:7b

# Benchmark a model
python -m onsetlab benchmark --model phi3.5
python -m onsetlab benchmark --compare phi3.5,qwen2.5:7b --verbose

# Export agent
python -m onsetlab export --format docker -o ./my-agent
python -m onsetlab export --format config -o agent.yaml
python -m onsetlab export --format binary -o agent.py
```

## Export & Deploy

Export your agent in multiple formats for deployment:

| Format | What you get | Requirements |
|--------|-------------|--------------|
| **YAML** | Portable config file | None |
| **Docker** | Dockerfile + compose with Ollama sidecar | Docker |
| **vLLM** | Docker setup with GPU-accelerated inference | NVIDIA GPU + Docker |
| **Script** | Standalone Python file | Ollama |

```python
# Export from code
agent.export("config", "my_agent.yaml")
agent.export("docker", "./deploy/")
agent.export("binary", "agent.py")
```

## Benchmark

Validate that a model handles tool calling correctly before deploying.

```bash
python -m onsetlab benchmark --model phi3.5 --verbose
```

```python
from onsetlab import Benchmark

result = Benchmark.run(model="phi3.5", verbose=True)
result.print_summary()
```

Tests tool selection (does the model pick the right tool?) and parameter extraction (does it fill in the right values?) across the built-in tools.

## Tested Models

| Model | Size | RAM needed | Notes |
|-------|------|-----------|-------|
| `phi3.5` | 3.8B | 4GB+ | Default. Good balance of speed and quality |
| `qwen2.5:3b` | 3B | 4GB+ | Fast, good for simple tasks |
| `qwen2.5:7b` | 7B | 8GB+ | Strong tool calling |
| `qwen3-a3b` | MoE (3B active / 30B total) | 16GB+ | Best tool calling accuracy |
| `llama3.2:3b` | 3B | 4GB+ | General purpose |

Works with any model available through Ollama. Use `python -m onsetlab benchmark --model your-model` to verify compatibility.

## Configuration

```python
agent = Agent(
    model="phi3.5",            # any Ollama model
    tools=[...],               # built-in tools
    memory=True,               # conversation memory
    verify=True,               # pre-execution plan verification
    routing=True,              # model-driven strategy selection
    react_fallback=True,       # automatic fallback on planning failure
    debug=False,               # verbose logging
)
```

---

## License

Apache 2.0
