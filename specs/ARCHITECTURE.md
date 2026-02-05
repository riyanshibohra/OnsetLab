# OnsetLab Architecture

> **"Reliable SLM agents. Plan once, execute fast, verify always."**

## Overview

OnsetLab is a Python library for building reliable tool-calling agents with Small Language Models (SLMs). Unlike ReAct-based frameworks, OnsetLab uses **REWOO** (Reasoning Without Observation) — plan all steps upfront, execute in batch, verify results.

## Why OnsetLab?

| Problem with ReAct | OnsetLab Solution |
|-------------------|-------------------|
| 5-10 SLM calls per task | **2-3 SLM calls** (REWOO) |
| Sequential execution | **Parallel tool execution** |
| SLM loses context mid-reasoning | **Full plan upfront** |
| Hard to debug | **See entire plan before execution** |
| No verification | **Built-in verify step** |

---

## Core Approach: REWOO + Verify

```
┌─────────────────────────────────────────────────────────────────┐
│                    User: "What % of Japan lives in Tokyo?"      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: PLAN (1 SLM call)                                      │
│                                                                 │
│  #E1 = WebSearch("Tokyo population 2024")                       │
│  #E2 = WebSearch("Japan population 2024")                       │
│  #E3 = Calculator(#E1 / #E2 * 100)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: EXECUTE (batch, parallel where possible)               │
│                                                                 │
│  #E1 = 13,960,000        ┐                                      │
│  #E2 = 125,700,000       ├── executed in parallel               │
│  #E3 = 11.1              ┘ (depends on #E1, #E2)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: VERIFY (1 SLM call)                                    │
│                                                                 │
│  ✓ Tokyo pop (13.9M) - reasonable                               │
│  ✓ Japan pop (125.7M) - reasonable                              │
│  ✓ Percentage (11.1%) - math checks out                         │
│  Result: VALID                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: ANSWER (1 SLM call)                                    │
│                                                                 │
│  "Tokyo's population of ~14 million is about 11.1% of           │
│   Japan's total population of 126 million."                     │
└─────────────────────────────────────────────────────────────────┘

Total: 3 SLM calls (vs 6-10 for ReAct)
```

### If Verification Fails

```
VERIFY → INVALID (e.g., search returned error)
    │
    ▼
REPLAN with error context → EXECUTE → VERIFY → ANSWER
```

---

## Three Pillars

```
┌─────────────────────────────────────────────────────────────────┐
│                         OnsetLab                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐   │
│  │  BENCHMARK  │   │   REWOO+VERIFY   │   │     PACKAGE     │   │
│  │             │   │                  │   │                 │   │
│  │ Which SLM   │   │ Plan → Execute   │   │ Docker, .exe,   │   │
│  │ works best? │──▶│ → Verify →       │──▶│ share with      │   │
│  │ We tested.  │   │ Answer           │   │ anyone.         │   │
│  │             │   │                  │   │                 │   │
│  └─────────────┘   └──────────────────┘   └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, WebSearch

agent = Agent(
    model="phi3.5",
    tools=[Calculator(), WebSearch()],
    memory=True,  # Remember conversation history
)

result = agent.run("What's 15% tip on $84.50?")
print(result)  # "The 15% tip on $84.50 would be $12.68"
```

---

## Components

### 1. Agent

The main interface. Uses REWOO strategy with verification.

```python
class Agent:
    def __init__(
        self,
        model: str = "phi3.5",           # SLM to use
        tools: list = None,               # Built-in tools
        mcp_servers: list = None,         # MCP server configs
        memory: bool = True,              # Conversation history
        verify: bool = True,              # Verification step
        max_replans: int = 2,             # Retries on failure
    ):
        ...
    
    def run(self, query: str) -> AgentResult:
        """
        1. Plan all tool calls
        2. Execute in batch (parallel where possible)
        3. Verify results
        4. Generate answer
        """
    
    def chat(self, message: str) -> str:
        """Interactive chat with memory."""
    
    def save_memory(self, path: str):
        """Save conversation history."""
    
    def load_memory(self, path: str):
        """Load conversation history."""
```

### 2. Model Backends

```python
# Local (recommended)
agent = Agent(model="phi3.5")           # Uses Ollama
agent = Agent(model="qwen2.5-3b")       # Uses Ollama
agent = Agent(model="./model.gguf")     # Uses llama-cpp-python

# Cloud (for demo/fallback)
agent = Agent(model="groq:llama-3.1-8b")
```

### 3. Built-in Tools

| Tool | Description | API Key? |
|------|-------------|----------|
| `Calculator` | Math, unit conversion | No |
| `WebSearch` | DuckDuckGo search | No |
| `DateTime` | Current time, timezones | No |
| `FileOps` | Read/write local files | No |
| `Shell` | Execute commands (sandboxed) | No |

### 4. MCP Integration

```python
from onsetlab import Agent, MCPServer

github = MCPServer(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "..."}
)

agent = Agent(model="phi3.5", mcp_servers=[github])
agent.run("List open issues in myorg/myrepo")
```

### 5. Memory

```python
agent = Agent(model="phi3.5", memory=True)

# Conversation 1
agent.run("Find open issues in riyanshibohra/onsetlab")
# → Returns list of issues

# Conversation 2 (remembers context)
agent.run("Close the first one")
# → Knows which issues were listed, closes #1

# Save for later
agent.save_memory("./session.json")

# Next session
agent = Agent(model="phi3.5", memory=True)
agent.load_memory("./session.json")
```

---

## Benchmarking

OnsetLab includes benchmarking to find the best SLM for your use case.

```bash
# CLI
onsetlab benchmark phi3.5 qwen2.5-3b llama3.2-3b

# Output:
# ┌─────────────────┬──────────┬─────────┬────────────┐
# │ Model           │ Accuracy │ Latency │ Best For   │
# ├─────────────────┼──────────┼─────────┼────────────┤
# │ phi3.5          │ 87%      │ 1.2s    │ General    │
# │ qwen2.5-3b      │ 91%      │ 1.5s    │ Math/Code  │
# │ llama3.2-3b     │ 82%      │ 1.1s    │ Speed      │
# └─────────────────┴──────────┴─────────┴────────────┘
```

```python
# Programmatic
from onsetlab.benchmark import run_benchmark

results = run_benchmark(
    models=["phi3.5", "qwen2.5-3b"],
    tools=[Calculator(), WebSearch()],
    dataset="tool-calling-basic",  # Built-in test set
)
```

---

## Packaging

Build your agent, package it, share with anyone.

```bash
# Package as Docker
onsetlab package ./my_agent --format=docker
# → my_agent.tar (docker load & run)

# Package as executable
onsetlab package ./my_agent --format=exe --platform=macos
# → my_agent.app (double-click to run)

# Package for Ollama
onsetlab package ./my_agent --format=ollama
# → Modelfile + instructions
```

```python
# Programmatic
from onsetlab.package import package_agent

package_agent(
    agent_dir="./my_agent",
    format="docker",
    include_model=True,  # Bundle the SLM
)
```

---

## Library Structure

```
onsetlab/
├── __init__.py           # Public API: Agent, MCPServer, tools
├── agent.py              # Agent class with REWOO strategy
├── rewoo/
│   ├── __init__.py
│   ├── planner.py        # Generate execution plan
│   ├── executor.py       # Batch execute tools
│   ├── verifier.py       # Verify results
│   └── solver.py         # Synthesize final answer
├── model/
│   ├── __init__.py
│   ├── base.py           # BaseModel interface
│   ├── ollama.py         # Ollama backend
│   ├── gguf.py           # llama-cpp-python backend
│   └── cloud.py          # Groq/Together backend
├── tools/
│   ├── __init__.py
│   ├── base.py           # BaseTool interface
│   ├── calculator.py
│   ├── websearch.py
│   ├── datetime.py
│   ├── fileops.py
│   └── shell.py
├── mcp/
│   ├── __init__.py
│   ├── server.py         # MCPServer class
│   ├── client.py         # MCP protocol client
│   └── registry.py       # Pre-configured MCP servers
├── memory/
│   ├── __init__.py
│   └── conversation.py   # Conversation history
├── benchmark/
│   ├── __init__.py
│   ├── runner.py         # Benchmark runner
│   ├── datasets.py       # Test datasets
│   └── metrics.py        # Accuracy, latency metrics
├── package/
│   ├── __init__.py
│   ├── docker.py         # Docker packaging
│   ├── executable.py     # PyInstaller packaging
│   └── ollama.py         # Ollama Modelfile
└── cli.py                # CLI: onsetlab benchmark, package, etc.
```

---

## What Gets Removed (from old codebase)

| Path | Reason |
|------|--------|
| `onsetlab/synthesis/` | Training data generation (no fine-tuning) |
| `onsetlab/training/` | Fine-tuning code (no fine-tuning) |
| `onsetlab/builder.py` | Old orchestrator (replaced by Agent) |
| `meta_agent/` | LangGraph agent (replaced) |
| `colab_*.ipynb` | Colab notebooks (no fine-tuning) |
| `generated_notebook.ipynb` | Generated notebook (no fine-tuning) |

**Keep:** `meta_agent/registry/` → Move to `onsetlab/mcp/registry/`

---

## Comparison with Alternatives

| | effGen | smolagents | OnsetLab |
|---|--------|------------|----------|
| Strategy | ReAct | ReAct | **REWOO** |
| SLM calls/task | 5-10 | 5-10 | **2-3** |
| Verification | No | No | **Yes** |
| Benchmarking | No | No | **Yes** |
| Packaging | No | No | **Yes** |
| Memory | Yes | Yes | Yes |
| MCP | Yes | Yes | Yes |

---

## Implementation Plan

### Phase 1: Core (MVP)
- [ ] Agent class with REWOO strategy
- [ ] Planner, Executor, Verifier, Solver
- [ ] Ollama backend
- [ ] Built-in tools: Calculator, DateTime
- [ ] Conversation memory

### Phase 2: Tools & MCP
- [ ] WebSearch, FileOps, Shell tools
- [ ] MCP client
- [ ] MCP registry (GitHub, Slack, etc.)

### Phase 3: Benchmark & Package
- [ ] Benchmark runner
- [ ] Test datasets
- [ ] Docker packaging
- [ ] Executable packaging (PyInstaller)

### Phase 4: Polish & Launch
- [ ] CLI (`onsetlab benchmark`, `onsetlab package`)
- [ ] PyPI publish
- [ ] Website with live demo
- [ ] Documentation

---

## Success Metrics

1. **Reliable:** Verification catches errors before returning
2. **Fast:** 2-3 SLM calls vs 5-10 for ReAct
3. **Simple:** <20 lines to get started
4. **Distributable:** Package and share with non-developers
