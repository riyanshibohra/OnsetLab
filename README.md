# OnsetLab

### Tool-calling agents that run on your laptop.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/onsetlab.svg)](https://pypi.org/project/onsetlab/)

[Install](#install) · [Quick Start](#quick-start) · [MCP Servers](#mcp-servers) · [Architecture](#architecture)

---

## The Problem

Agent frameworks assume GPT-4. They send every request to an API, cost money per call, and break when the network is down. Small language models can run locally for free — but they can't do tool calling reliably. They hallucinate tool names, mess up parameters, and don't know when to stop.

## The Solution

OnsetLab makes 3B–7B models do real tool calling. A hybrid REWOO/ReAct architecture handles the planning, and the model only has to do what it's good at — one step at a time.

```
pip install onsetlab → connect Ollama → connect tools → run
```

No API keys. No fine-tuning. No cloud.

---

## How It Works

| Step | What Happens |
|:----:|-------------|
| **1** | Your query hits the **Router** — the model itself decides: do I need tools, or can I answer directly? |
| **2** | If tools are needed, the **Planner** generates a step-by-step execution plan with explicit reasoning |
| **3** | The **Verifier** checks the plan for errors before anything runs |
| **4** | The **Executor** runs each tool call with dependency resolution |
| **5** | The **Solver** reads all tool outputs and writes a final answer |
| **6** | If planning failed? **ReAct fallback** kicks in — iterative think → act → observe until solved |

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

That's it. The agent routes the query, builds a plan, calls the calculator, and returns the answer.

---

## MCP Servers

Connect any [MCP-compatible](https://modelcontextprotocol.io) server — filesystem, GitHub, Slack, search, databases, anything.

```python
from onsetlab import Agent, MCPServer

# From the built-in registry
server = MCPServer.from_registry("filesystem", extra_args=["/path/to/dir"])

agent = Agent("qwen2.5:7b")
agent.add_mcp_server(server)

result = agent.run("List all Python files in the directory")
print(result.answer)

agent.disconnect_mcp_servers()
```

**Any npm MCP server works:**

```python
# Web search (needs API key)
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
```

**Built-in registry:** `filesystem` · `github` · `slack` · `notion` · `google_calendar` · `tavily`

---

## Architecture

```
                        ┌─────────┐
                        │  Query  │
                        └────┬────┘
                             │
                       ┌─────▼─────┐
                       │   Router   │  ← model classifies: tools needed?
                       └─────┬─────┘
                    ┌────────┼────────┐
                    ▼                 ▼
              ┌───────────┐    ┌──────────┐
              │   DIRECT   │    │   REWOO   │
              │ (no tools) │    │  Pipeline │
              └───────────┘    └─────┬─────┘
                                     │
                    ┌────────────────┬┴───────────────┐
                    ▼                ▼                 ▼
              ┌──────────┐   ┌───────────┐   ┌────────────┐
              │ Planner   │ → │ Executor  │ → │   Solver   │
              │ THINK/PLAN│   │ run tools │   │ synthesize │
              └──────────┘   └───────────┘   └────────────┘
                    │                                 │
                    │  plan failed?                    │
                    ▼                                 ▼
              ┌──────────────┐                  ┌──────────┐
              │ ReAct        │ ───────────────→ │  Answer  │
              │ Fallback     │  think→act→observe└──────────┘
              └──────────────┘
```

**Router** — The model itself decides the strategy. No regex, no keyword matching. The SLM reads the query and available tools, then classifies: `REWOO` (needs tools) or `DIRECT` (answer from knowledge). Trivial greetings are caught before the model is even called.

**Planner** — Generates a structured `THINK → PLAN` output. Each plan step specifies a tool, parameters, and dependencies on previous steps. Tool rules are auto-generated from JSON schemas — the model sees exactly what each tool can do.

**Executor** — Resolves dependencies between steps and executes tool calls. If step 2 depends on step 1's output, it substitutes the result automatically.

**ReAct Fallback** — If REWOO planning fails (bad format, wrong tool, missing params), the agent switches to iterative `Thought → Action → Observation` loops. This catches edge cases that structured planning misses.

---

## Built-in Tools

| Tool | What it does |
|------|-------------|
| `Calculator` | Math expressions, percentages, sqrt/sin/log |
| `DateTime` | Current time, timezones, date math, day of week |
| `UnitConverter` | Length, weight, temperature, volume, speed, data |
| `TextProcessor` | Word count, find/replace, case transform, patterns |
| `RandomGenerator` | Random numbers, UUIDs, passwords, dice, coin flips |

## Tested Models

| Model | Size | Notes |
|-------|------|-------|
| `qwen2.5:7b` | 7B | Best for tool calling |
| `qwen2.5:3b` | 3B | Fast, good for simple tasks |
| `phi3.5` | 3.8B | Solid balance |
| `llama3.2:3b` | 3B | General purpose |

---

## Configuration

```python
agent = Agent(
    model="qwen2.5:7b",       # any Ollama model
    tools=[...],               # built-in tools
    memory=True,               # conversation memory
    verify=True,               # pre-execution plan verification
    routing=True,              # model-driven strategy selection
    react_fallback=True,       # fallback on REWOO failure
    debug=False,               # verbose logging
)
```

---

## License

Apache 2.0
