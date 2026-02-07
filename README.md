# OnsetLab

### Make small models do actual work.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/onsetlab.svg)](https://pypi.org/project/onsetlab/)
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-000000.svg)](https://ollama.com)

[Install](#install) · [Quick Start](#quick-start) · [MCP Servers](#mcp-servers) · [Architecture](#architecture)

---

## The Problem

Agent frameworks send every request to an API. They cost money per call and break when the network goes down.

Small language models (SLMs) can run locally for free. They're fast, private, and surprisingly capable. But they can't do tool calling reliably. They hallucinate tool names, mess up parameters, and don't know when to stop.

## The Solution

OnsetLab makes 3B-7B models do reliable tool calling through a hybrid REWOO/ReAct architecture. The framework handles planning, execution, and error recovery. The model only does what it's actually good at: one step at a time.

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
ollama pull qwen2.5:7b
```

## Quick Start

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime

agent = Agent("qwen2.5:7b", tools=[Calculator(), DateTime()])

result = agent.run("What's 15% tip on $84.50?")
print(result.answer)
```

The agent routes the query, builds an execution plan, calls the calculator, and returns the answer. No prompt engineering required.

---

## Architecture

```
                              ┌─────────┐
                              │  Query  │
                              └────┬────┘
                                   │
                            ┌──────▼──────┐
                            │   Router    │
                            │  (model)    │
                            └───┬─────┬───┘
                                │     │
               no tools needed  │     │  tools needed
                                │     │
                         ┌──────▼┐   ┌▼──────────────────────────────┐
                         │DIRECT │   │         REWOO Pipeline        │
                         │       │   │                                │
                         │answer │   │  Planner → Executor → Solver  │
                         │from   │   │                                │
                         │model  │   │  on failure:                   │
                         │       │   │  ReAct Fallback                │
                         │       │   │  think → act → observe         │
                         └───┬───┘   └──────────────┬────────────────┘
                             │                      │
                             └──────────┬───────────┘
                                        ▼
                                  ┌──────────┐
                                  │  Answer  │
                                  └──────────┘
```

### Router

The model itself decides the strategy. No regex, no keyword matching. The SLM reads the query and available tools, then classifies: `REWOO` (needs tools) or `DIRECT` (answer from knowledge). Trivial greetings are caught before the model is even called.

### Planner

Generates a structured `THINK → PLAN` output. Each plan step specifies a tool, parameters, and dependencies on previous steps. Tool rules are auto-generated from JSON schemas, so the model sees exactly what each tool can do and how to call it.

### Executor

Resolves dependencies between steps and runs tool calls in order. If step 2 depends on step 1's output, the result is substituted automatically.

### ReAct Fallback

If REWOO planning fails (bad format, wrong tool, missing params), the agent switches to iterative `Thought → Action → Observation` loops. This catches edge cases that structured planning misses, and recovers without restarting.

---

## MCP Servers

Connect any [MCP-compatible](https://modelcontextprotocol.io) server to give your agent access to external tools.

**From the built-in registry:**

```python
from onsetlab import Agent, MCPServer

server = MCPServer.from_registry("filesystem", extra_args=["/path/to/dir"])

agent = Agent("qwen2.5:7b")
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

```python
from onsetlab.tools import Calculator, DateTime, UnitConverter, TextProcessor, RandomGenerator
```

## Tested Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen2.5:7b` | 7B | Best results for tool calling |
| `qwen2.5:3b` | 3B | Fast, good for simple tasks |
| `phi3.5` | 3.8B | Solid balance of speed and quality |
| `llama3.2:3b` | 3B | General purpose |

Works with any model available through Ollama.

## Configuration

```python
agent = Agent(
    model="qwen2.5:7b",       # any Ollama model
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
