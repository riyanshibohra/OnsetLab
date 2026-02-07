# OnsetLab — What We Are Building

> **The open-source SDK for building MCP tool-calling agents.**
> Works with any model. Optimized for small/local models. One command to package and deploy.

---

## The Problem

LLMs are great at tool calling. SLMs (3-8B) are not. They pick the wrong tool, hallucinate parameters, forget required fields, and get confused by large tool sets. But people want local, private, cheap AI agents — and that means small models.

**OnsetLab solves this:** a framework that makes SLMs reliable at tool calling through architecture, not model size.

---

## What OnsetLab Actually Is

OnsetLab is a **Python SDK** (`pip install onsetlab`) that gives any model — especially small ones — structured, reliable tool calling via MCP (Model Context Protocol).

It is NOT:
- A fine-tuned model
- A hosted API
- A chatbot product

It IS:
- A framework that wraps any LLM/SLM with a battle-tested execution architecture
- A bridge between MCP servers and small language models
- A packaging system that exports agents to Docker/vLLM/standalone

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────┐
│  ROUTER  │  ← Classifies: does this need a tool or not?
└────┬────┘
     │
     ├── DIRECT ──────────────────────────────────────┐
     │   (no tools needed, answer from context)       │
     │                                                 │
     ├── REWOO ──┐                                    │
     │           │                                     │
     │     ┌─────▼──────┐                             │
     │     │   SKILLS    │ ← Injects tool-specific    │
     │     │  (detect)   │   prompt context            │
     │     └─────┬──────┘                             │
     │           │                                     │
     │     ┌─────▼──────┐                             │
     │     │  PLANNER    │ ← Generates execution      │
     │     │             │   plan upfront              │
     │     └─────┬──────┘                             │
     │           │                                     │
     │     ┌─────▼──────┐                             │
     │     │  EXECUTOR   │ ← Runs tools (parallel     │
     │     │             │   or sequential)            │
     │     └─────┬──────┘                             │
     │           │                                     │
     │           ├── Success ──┐                       │
     │           │             │                       │
     │           ├── Failure ──┼── REACT FALLBACK      │
     │           │             │   (iterative fix)     │
     │           │             │                       │
     │     ┌─────▼──────┐     │                       │
     │     │   SOLVER    │◄───┘                       │
     │     │             │ ← Synthesizes natural      │
     │     └─────┬──────┘   language answer           │
     │           │                                     │
     └── REACT ──┘                                    │
         (exploratory                                  │
          tasks)                                       │
                                                       │
     ◄─────────────────────────────────────────────────┘
     │
     ▼
  Response
```

---

## The Pipeline — Step by Step

Here is exactly what happens when a user sends a query, in order.

### Step 1: Router

**File:** `onsetlab/router.py`

**What it does:** Classifies the user's query into one of three strategies:

| Strategy | When | Example |
|----------|------|---------|
| **DIRECT** | No tools needed | "What tools do you have?", "Hello", "Thanks" |
| **REWOO** | Structured, predictable task | "Create a GitHub issue titled Bug Fix" |
| **REACT** | Exploratory, search-like task | "Search for recent issues in my repo" |

**How it decides:**
1. Checks for meta/help patterns → DIRECT
2. Matches query words against tool names/descriptions → scored relevance
3. Checks for exploratory words (search, find, explore) → REACT
4. Checks for sequential words (then, after, first...then) → REWOO
5. Default: REWOO (safest — plan first, react on failure)

**When MCP tools are connected:** If the Router initially says DIRECT, an additional **model-driven intent classification** call asks the LLM itself: "Does this need a tool call?" (ACTION vs DIRECT). This prevents the router from misclassifying tool-requiring queries as conversational.

---

### Step 2: Skills Detection

**File:** `onsetlab/skills/__init__.py`

**What it does:** Detects which API/service the query is about and injects a specialized prompt into the Planner.

**Why this matters:** A 3B model told "call the GitHub API" will hallucinate parameter names. A 3B model told "split owner/repo into SEPARATE params, issue_write needs method='create'" performs significantly better. Skills are essentially **few-shot prompts specialized per API**.

**Currently defined skills:**

| Skill | Triggers On | What It Teaches the Model |
|-------|-------------|--------------------------|
| GitHub | issue, repo, commit, PR... | Split owner/repo, method is REQUIRED for issue_read/write |
| Slack | slack, channel, message | Channel IDs look like "C01ABCDEF", use list_channels first |
| Notion | notion, page, database | IDs are UUIDs, use search first to find IDs |
| Search | tavily, search, research | Use keywords not full sentences for queries |
| Code | python, execute, run_code | Write complete runnable code, use print() for output |
| Built-in | calculator, datetime... | Exact parameter formats for each built-in tool |

**How detection works:**
1. Score each skill by how many of its keywords appear in the active tool names
2. Fallback: check the query text itself for skill-related keywords
3. Winner gets its `context` string injected into the Planner prompt

**The skill context string** goes into the Planner as additional rules. For example, the GitHub skill injects:
```
SKILL: GitHub API
Rules:
- Split owner/repo into SEPARATE params: owner="user", repo="name"
- issue_read: method is REQUIRED ("get", "get_comments", ...)
- list_issues: owner, repo REQUIRED. Use this to find latest/recent issues
- Do NOT invent tool names. Use EXACT names from the tool list.
Examples: list_issues(owner="user", repo="myrepo"), ...
```

---

### Step 3: Tool Filtering

**File:** `web/backend/app/services/agent_service.py` → `_filter_tools_for_query()`

**What it does:** When MCP servers expose 40+ tools, the model can't handle all of them. This step scores each tool's relevance to the query and keeps only the top N (default: 8).

**How scoring works:**
- Tool name word overlap with query → 5 points per match
- Substring match in tool name → 3 points
- Description word overlap → 1.5 points per match
- Synonym expansion (e.g., "PRs" → "pull requests") improves matching

**Critical design choice:** The **Planner** and **ReAct** only see filtered tools (small prompt), but the **Executor** keeps ALL tools (so any valid tool name works even if it wasn't in the prompt).

---

### Step 4: Planner (REWOO)

**File:** `onsetlab/rewoo/planner.py`

**What it does:** Generates an execution plan BEFORE running any tools. This is the core of the REWOO architecture — "Reasoning WithOut Observation."

**The prompt structure:**
```
Call ONE tool to complete the task.

Tools:
list_issues(owner="...", repo="...") - List issues in a repository
issue_write(method="create" or "update", ...) - Create or update an issue
...

Examples: list_issues(owner="user", repo="myrepo"), ...

Rules:
- Copy exact values from task
- Use format: tool(param="value")
SKILL: GitHub API    ← [Injected by Skills]
Rules:               ← [Skill-specific rules]
- Split owner/repo into SEPARATE params
...

CONTEXT (previous results):   ← [Conversation history + prior results]
...

Task: Create a GitHub issue titled "Bug fix" in user/myrepo
#E1 =
```

**The model completes:** `issue_write(method="create", owner="user", repo="myrepo", title="Bug fix")`

**Post-processing:**
1. Parse the response → extract tool name + parameters
2. Deduplicate steps (model sometimes repeats)
3. Validate steps:
   - Does the tool exist? (strict name matching, no fuzzy guessing)
   - Are required parameters present?
   - If validation fails → step gets an `error` field

**SLM-specific optimizations:**
- Only required parameters shown in tool descriptions (less confusion)
- Concrete example values instead of abstract types
- Enum values shown inline (e.g., `method="get" or "create"`)
- Response capped at 150 tokens with aggressive stop sequences
- Temperature 0.0 for deterministic output

---

### Step 5: Executor

**File:** `onsetlab/rewoo/executor.py`

**What it does:** Takes the plan from Step 4 and actually calls the tools.

**Key capabilities:**
- **Dependency resolution:** If Step 2 references `#E1` (result of Step 1), the executor substitutes the actual value before calling
- **Parallel execution:** Independent steps (no shared dependencies) run concurrently via `ThreadPoolExecutor`. Steps are grouped into "waves" using topological sort (Kahn's algorithm)
- **Parameter normalization:** Handles camelCase↔snake_case conversion
- **Positional argument mapping:** If model outputs `tool("val1", "val2")` instead of named params, maps them to parameter names in order

**Error handling:** If a step has a validation error from the Planner, it returns the error string instead of calling the tool. This error propagates to the fallback system.

---

### Step 6: Failure Recovery (3 layers)

If the Executor returns errors, the system tries to recover before giving up.

#### Layer 1: Parameter Rescue

**Where:** `agent_service.py` → `_rescue_missing_params()`

**What:** If the error is "missing required parameter X", try to extract the value from the original query using regex patterns.

**Example:** Query is "Create issue titled 'Bug fix' in user/myrepo" but model forgot `method`. The rescue system detects `create` intent and injects `method="create"`. Re-executes the same plan — no extra LLM call.

#### Layer 2: ReAct Fallback (1 iteration)

**File:** `onsetlab/rewoo/react_fallback.py`

**What:** If rescue fails, switch to ReAct — give the model the error context and let it try ONE more time with the Thought→Action→Observation loop.

**Key difference from standard ReAct:** Limited to 1-2 iterations (not 5+). REWOO already identified the right tool — ReAct just needs to fix the parameters. More iterations = more chances for the model to hallucinate or duplicate actions.

**Write Guard:** Write/mutate tools (create, update, delete) are wrapped in a `_WriteGuardTool` that only allows ONE successful execution. Prevents the model from creating 5 duplicate GitHub issues in a ReAct loop.

#### Layer 3: Solver Synthesis

Even if ReAct hits max iterations, if ANY step produced a successful result, the Solver is used to synthesize a human-readable answer from whatever data was collected.

---

### Step 7: Solver

**File:** `onsetlab/rewoo/solver.py`

**What it does:** Takes the raw tool results and the original question, and generates a natural language answer.

**The prompt:**
```
Question: What are the open issues in user/myrepo?

Data:
[{"title": "Bug fix", "state": "open", ...}, ...]

Write a focused answer using ONLY the data above. Stay on topic. Stop when done.

Answer:
```

**Optimizations:**
- Result data capped at 800 chars (small models get confused by large payloads)
- Error results are flagged so the Solver can explain what went wrong
- Response cleaned: removes common prefixes ("Based on the results,..."), parenthetical notes, incomplete sentences

---

## MCP Integration

**Files:** `onsetlab/mcp/`

OnsetLab connects to any MCP server and wraps its tools as `BaseTool` instances.

### How it works:

```python
from onsetlab import Agent, MCPServer

# Connect to GitHub MCP server
github = MCPServer.from_registry("github", env={"GITHUB_TOKEN": "..."})

# Create agent with MCP tools
agent = Agent(model="qwen2.5:3b", mcp_servers=[github])
result = agent.run("List issues in user/myrepo")
```

### Under the hood:
1. `MCPServer.connect()` spawns the MCP server process (via npx/docker/python)
2. `SyncMCPClient` discovers available tools via the MCP protocol
3. Each MCP tool is wrapped in `MCPToolWrapper` which:
   - Converts MCP input schema to OnsetLab's parameter format
   - Coerces types (string→int, string→array via JSON parse)
   - Auto-reconnects on connection loss
   - Returns errors as strings (never throws to the agent)

### Registry:
Pre-configured MCP servers in `onsetlab/mcp/registry/`:
- GitHub, Slack, Notion, Tavily, Google Calendar, Filesystem

---

## Built-in Tools

**File:** `onsetlab/tools/`

6 tools that work without any external server:

| Tool | What it does |
|------|-------------|
| Calculator | Math expressions (`expression="2+2"`) |
| DateTime | Current time, date arithmetic |
| UnitConverter | km↔miles, celsius↔fahrenheit, etc. |
| TextProcessor | Uppercase, reverse, word count |
| RandomGenerator | Random int, float, choice |
| CodeExecutor | Run Python/JS/Bash in a sandbox |

---

## Packaging & Deployment

**File:** `onsetlab/packaging/`

Export any agent for deployment:

| Format | What you get | Use case |
|--------|-------------|----------|
| `config` | YAML/JSON config file | Version control, sharing |
| `docker` | Dockerfile + docker-compose (Ollama) | Standard deployment |
| `docker` + `engine="vllm"` | Dockerfile + docker-compose (vLLM) | GPU-accelerated, 5-10x faster |
| `binary` | Standalone Python script | Quick distribution |

```python
# Export as Docker with Ollama
agent.export("docker", "./deploy/")

# Export as Docker with vLLM (GPU)
agent.export("docker", "./deploy/", engine="vllm")
```

---

## Web Playground

**Files:** `web/frontend/` (React + Vite) and `web/backend/` (FastAPI)

A browser-based demo UI that showcases the SDK. NOT the product — the product is the SDK.

**What the playground does:**
- Connect MCP servers with BYO token
- Chat with the agent, see the plan/results/strategy in real-time
- Export agent config/Docker/vLLM packages
- Shows skill badges, strategy labels, step-by-step execution

---

## User Journey (SDK)

### 1. Install
```bash
pip install onsetlab
```

### 2. Connect tools
```python
from onsetlab import Agent, MCPServer

github = MCPServer.from_registry("github", env={"GITHUB_TOKEN": "ghp_..."})
```

### 3. Create agent
```python
agent = Agent(
    model="qwen2.5:3b",        # Any Ollama model
    mcp_servers=[github],       # Any MCP servers
    react_fallback=True,        # Auto-recover from failures
    routing=True,               # Smart strategy selection
)
```

### 4. Use it
```python
result = agent.run("Create an issue titled 'Bug fix' in user/myrepo")
print(result.answer)
print(result.strategy_used)   # "rewoo"
print(result.slm_calls)       # 2 (plan + solve)
```

### 5. Deploy
```python
agent.export("docker", "./my_agent/")
# → Generates Dockerfile, docker-compose.yml, entrypoint.py, README
# → `docker compose up` to run
```

---

## What Makes This Different

| Feature | LangChain | CrewAI | OnsetLab |
|---------|-----------|--------|----------|
| Optimized for SLMs | No | No | **Yes** |
| MCP native | No (adapters) | No | **Yes** |
| Plan-first (REWOO) | No (ReAct only) | No | **Yes** |
| Auto-fallback | No | No | **Yes** (REWOO→ReAct) |
| Skill prompts | No | No | **Yes** |
| Tool filtering | No | No | **Yes** (top-N per query) |
| One-command packaging | No | No | **Yes** (Docker/vLLM) |
| Write guards | No | No | **Yes** |
| Parameter rescue | No | No | **Yes** |

---

## Current Limitations (Honest)

1. **SLMs still fail.** The architecture compensates but doesn't eliminate model limitations. Tool calling accuracy drops significantly beyond 5-8 tools.
2. **Skills are manually defined.** Each new API needs a hand-written skill definition.
3. **Router is rule-based.** The Router uses regex patterns + word matching, not semantic understanding.
4. **No streaming.** Results come back all at once, not token-by-token.
5. **Single-turn MCP.** Each query is independent — no multi-turn tool conversations.

---

## Where to Double Down

1. **Model selection guidance** — Not all SLMs are equal. Recommend models that are actually good at tool calling (Qwen 2.5 Instruct, Hermes 3, Functionary).
2. **Aggressive tool filtering** — Cap at 5 tools per query, not 8. Fewer choices = more reliable.
3. **SDK-first marketing** — The value is `pip install onsetlab`, not the playground.
4. **Documentation & examples** — Make it dead simple for a developer to go from zero to working agent in 5 minutes.
5. **Benchmarks** — Show concrete accuracy numbers: "Qwen 2.5 3B with OnsetLab achieves X% tool-calling accuracy vs Y% without."
