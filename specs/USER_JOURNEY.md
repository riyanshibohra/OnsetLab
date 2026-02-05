# OnsetLab Website - User Journeys

## Website Purpose

1. **Prove it works** — Live demo showing REWOO in action
2. **Show the difference** — Side-by-side ReAct vs REWOO comparison
3. **Get users started** — Quick copy-paste code
4. **Benchmark results** — Which SLM is best for what

---

## Site Structure

```
onsetlab.dev/
├── / (landing)           # Hero, value prop, demo link
├── /demo                  # Live interactive demo
├── /benchmark             # SLM leaderboard
├── /docs                  # Documentation
│   ├── /quickstart
│   ├── /tools
│   ├── /mcp
│   ├── /packaging
│   └── /api
└── /examples              # Code examples
```

---

## Page 1: Landing (`/`)

### Above the Fold

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                         OnsetLab                                │
│                                                                 │
│     Reliable SLM agents. Plan once, execute fast.               │
│                                                                 │
│         [Try Demo]              [Get Started]                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Problem / Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ReAct agents make 5-10 SLM calls per task.                     │
│  OnsetLab makes 2-3.                                            │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │ ReAct               │    │ OnsetLab (REWOO)    │             │
│  │                     │    │                     │             │
│  │ Think → Act →       │    │ Plan → Execute →    │             │
│  │ Observe → Think →   │    │ Verify → Answer     │             │
│  │ Act → Observe →     │    │                     │             │
│  │ Think → Answer      │    │ Done.               │             │
│  │                     │    │                     │             │
│  │ 6-10 SLM calls      │    │ 2-3 SLM calls       │             │
│  │ ~8 seconds          │    │ ~3 seconds          │             │
│  └─────────────────────┘    └─────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code Example

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, WebSearch

agent = Agent(
    model="phi3.5",
    tools=[Calculator(), WebSearch()],
    memory=True,
)

result = agent.run("What's 15% tip on $84.50?")
print(result)  # "The tip would be $12.68"
```

### Three Features

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   ⚡ REWOO       │  │   📊 Benchmark   │  │   📦 Package     │
│                  │  │                  │  │                  │
│  2-3 SLM calls   │  │  Which SLM is    │  │  Docker, .exe,   │
│  vs 5-10 for     │  │  best? We        │  │  share with      │
│  ReAct. Faster.  │  │  tested them.    │  │  anyone.         │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Install

```bash
pip install onsetlab
ollama pull phi3.5
```

---

## Page 2: Live Demo (`/demo`)

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  🧪 Try OnsetLab                                [5 queries left]│
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ What percentage of Japan's population lives in Tokyo?     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                        [Run ▶]  │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  📋 PLAN (1 SLM call)                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ #E1 = WebSearch("Tokyo population 2024")                │    │
│  │ #E2 = WebSearch("Japan population 2024")                │    │
│  │ #E3 = Calculator(#E1 / #E2 * 100)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ⚡ EXECUTE (parallel)                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ #E1 = 13,960,000 ✓                                      │    │
│  │ #E2 = 125,700,000 ✓                                     │    │
│  │ #E3 = 11.1 ✓                                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ✅ VERIFY (1 SLM call)                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ All values reasonable. Math verified. VALID.            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  💬 ANSWER                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Tokyo's population of ~14 million represents about      │    │
│  │ 11.1% of Japan's total population of 126 million.       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ⏱️ 3 SLM calls | 2.8 seconds | Tools: WebSearch, Calculator   │
│                                                                 │
│  [Get Started →]  [View Benchmark →]                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Demo Behavior

1. **Rate limited:** 5-10 queries per session
2. **Tools available:** Calculator, WebSearch, DateTime
3. **Backend:** Groq free tier (fast inference)
4. **Shows REWOO trace:** Plan → Execute → Verify → Answer

### Demo Queries (Suggested)

- "What's 15% tip on $84.50?"
- "What percentage of Japan's population lives in Tokyo?"
- "What day of the week was January 1, 2000?"
- "Compare the populations of NYC and LA"

---

## Page 3: Benchmark (`/benchmark`)

### SLM Leaderboard

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  📊 SLM Tool-Calling Benchmark                                  │
│                                                                 │
│  Which small model is best for agent tasks? We tested.          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Model           │ Accuracy │ Latency │ Best For         │    │
│  ├─────────────────┼──────────┼─────────┼──────────────────│    │
│  │ 🥇 Qwen2.5-3B   │ 91%      │ 1.5s    │ Math, Code       │    │
│  │ 🥈 Phi-3.5      │ 87%      │ 1.2s    │ General          │    │
│  │ 🥉 Llama-3.2-3B │ 82%      │ 1.1s    │ Speed            │    │
│  │    Mistral-7B   │ 89%      │ 2.1s    │ Complex tasks    │    │
│  │    Gemma-2-2B   │ 76%      │ 0.9s    │ Ultra-fast       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Methodology: 500 tool-calling tasks across Calculator,         │
│  WebSearch, and DateTime. Measured on M2 MacBook Air.           │
│                                                                 │
│  [Run Your Own Benchmark →]                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Page 4: Quickstart (`/docs/quickstart`)

### Step 1: Install

```bash
pip install onsetlab
ollama pull phi3.5
```

### Step 2: Basic Agent

```python
from onsetlab import Agent
from onsetlab.tools import Calculator

agent = Agent(
    model="phi3.5",
    tools=[Calculator()],
)

result = agent.run("What's 1234 * 5678?")
print(result)
```

### Step 3: Add Memory

```python
agent = Agent(
    model="phi3.5",
    tools=[Calculator(), WebSearch()],
    memory=True,  # Remembers conversation
)

# First message
agent.run("Search for Python release dates")

# Follow-up (remembers context)
agent.run("When was version 3.10 released?")
```

### Step 4: Connect MCP Servers

```python
from onsetlab import Agent, MCPServer

github = MCPServer(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}
)

agent = Agent(
    model="phi3.5",
    mcp_servers=[github],
    memory=True,
)

agent.run("List open issues in myorg/myrepo")
agent.run("Close the oldest one")  # Remembers the issues
```

### Step 5: Package and Share

```bash
# Package as Docker
onsetlab package ./my_agent --format=docker

# Package as executable
onsetlab package ./my_agent --format=exe --platform=macos

# Share with anyone - no Python needed!
```

---

## User Journeys

### Journey 1: "Curious Developer"

**Who:** Saw OnsetLab, wants to see if REWOO is actually faster

```
1. Lands on /
2. Sees "2-3 SLM calls vs 5-10" comparison
3. Clicks [Try Demo]
4. Runs: "What percentage of Japan lives in Tokyo?"
5. Watches Plan → Execute → Verify → Answer
6. Sees "3 SLM calls, 2.8 seconds"
7. Thinks: "That is faster than my LangChain agent"
8. Clicks [Get Started]
9. Runs locally, works
10. Checks /benchmark to pick best model
```

**Time:** 5 minutes

### Journey 2: "Builder with Sharing Need"

**Who:** Built a tool-calling agent, wants to share with non-technical colleague

```
1. Already has OnsetLab agent working
2. Goes to /docs/packaging
3. Runs: onsetlab package ./my_agent --format=exe
4. Gets my_agent.app
5. Sends to colleague
6. Colleague double-clicks, uses agent
7. No Python, no setup, just works
```

**Time:** 10 minutes

### Journey 3: "Performance Optimizer"

**Who:** Wants the fastest SLM for their use case

```
1. Goes to /benchmark
2. Sees Qwen2.5-3B is best for math
3. Sees Llama-3.2-3B is fastest overall
4. Runs own benchmark: onsetlab benchmark qwen2.5-3b phi3.5 --tools calculator
5. Picks winner for their specific task
```

**Time:** 15 minutes

---

## Technical Implementation

### Demo Backend

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│   FastAPI   │────▶│   Groq API  │
│             │◀────│   + REWOO   │◀────│   (fast)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ Rate Limiter│
                    └─────────────┘
```

### Rate Limiting

```python
# Per IP: 5 queries per hour
# Per session: 10 queries total
# Global cap with spending limit
```

### Cost Estimate

| Traffic | Queries/day | Cost/month |
|---------|-------------|------------|
| Low | 100 | $0 (Groq free) |
| Medium | 1,000 | ~$10 |
| High | 10,000 | ~$100 |

---

## Content Checklist

### Landing Page
- [ ] Hero with tagline
- [ ] ReAct vs REWOO comparison
- [ ] Code example
- [ ] Three feature cards
- [ ] Install commands

### Demo Page
- [ ] Input box
- [ ] Query counter
- [ ] REWOO trace display (Plan/Execute/Verify/Answer)
- [ ] Timing info
- [ ] Tool badges

### Benchmark Page
- [ ] SLM leaderboard table
- [ ] Methodology explanation
- [ ] Link to run own benchmark

### Docs
- [ ] Quickstart
- [ ] Tools reference
- [ ] MCP integration
- [ ] Memory guide
- [ ] Packaging guide
- [ ] API reference

---

## Design Notes

- **Show the difference:** Always compare to ReAct (our advantage)
- **Timing visible:** Show SLM calls and seconds prominently
- **Dark mode:** Default
- **Fast demo:** <3s response or feels broken
