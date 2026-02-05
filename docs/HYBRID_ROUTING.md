# OnsetLab Hybrid Routing

OnsetLab uses a **hybrid approach** that intelligently selects the best execution strategy for each task.

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                    TASK INPUT                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    ROUTER                               │
│   Analyzes task → Selects optimal strategy              │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       DIRECT          REWOO           REACT
    (no tools)      (plan-first)   (iterative)
          │               │               │
          │               ▼               │
          │         SUCCESS? ─────NO────►│
          │               │               │
          │              YES              │
          └───────────────┴───────────────┘
                          │
                          ▼
                       RESULT
```

## Strategies

### DIRECT
**When:** Meta questions, help requests, no tools needed
- "What tools do you have?"
- "Help me understand this"
- "What can you do?"

**How:** Answers directly without executing tools.

### REWOO (Plan-First)
**When:** Predictable, structured tasks
- "What is 15 + 27?"
- "Convert 100 km to miles"
- "Calculate X, then convert to Y"

**How:**
1. Generate complete execution plan upfront
2. Execute all steps
3. Synthesize final answer

**Pros:** Fast (fewer LLM calls), efficient for known patterns
**Cons:** Can fail on dynamic tasks that need intermediate results

### REACT (Iterative)
**When:** Exploratory, dynamic, or search tasks
- "Search for Python documentation"
- "Find information about X"
- Tasks where next step depends on previous result

**How:**
1. Think about what to do
2. Execute one action
3. Observe result
4. Repeat until done

**Pros:** Flexible, self-correcting
**Cons:** Slower (more LLM calls)

## Automatic Fallback

If REWOO fails (empty plan, validation errors, execution errors), the agent automatically falls back to ReAct:

```python
# REWOO attempt fails
plan = planner.plan(task)  # Returns empty or invalid

# Automatic fallback triggers
answer = react.run(task, error_context)
```

The fallback includes error context from the failed REWOO attempt, helping ReAct avoid the same mistakes.

## How Routing Works

The router uses **tool-semantic matching** (not fragile keywords):

1. **Meta Pattern Detection**
   - Checks for help/meta questions
   - Routes to DIRECT if matched

2. **Tool Relevance Scoring**
   - Matches task against tool names and descriptions
   - Higher scores for:
     - Tool name in task (+5)
     - Tool name parts in task (+3)
     - Description keywords match (+2 each)
     - Action verbs match tool purpose (+4)

3. **Exploratory Detection**
   - Patterns like "search", "find", "explore"
   - Routes to REACT if matched AND search tools available

4. **Default Behavior**
   - If tools match → REWOO
   - If no tools match → DIRECT

## Configuration

```python
from onsetlab import Agent

agent = Agent(
    model="phi3.5",
    tools=[...],
    routing=True,      # Enable hybrid routing (default)
    react_fallback=True,  # Enable fallback (default)
)
```

Disable routing to force REWOO-only:
```python
agent = Agent(..., routing=False)
```

## Observability

The `AgentResult` includes routing information:

```python
result = agent.run("What is 15 + 27?")

print(result.strategy_used)      # "rewoo", "react", "direct", or "rewoo->react"
print(result.routing_decision)   # Full routing decision with confidence
print(result.used_react_fallback)  # True if fallback was triggered
```

## When Each Strategy is Best

| Task Type | Best Strategy | Why |
|-----------|---------------|-----|
| Single tool, clear params | REWOO | Fast, one plan cycle |
| Multi-step, predictable | REWOO | Plan all at once |
| Search/explore | ReAct | Need intermediate results |
| Meta questions | Direct | No tools needed |
| Ambiguous | REWOO → ReAct | Try structured, fallback if needed |

## Benchmark Results

With hybrid routing and built-in tools:

| Model | Accuracy | Speed |
|-------|----------|-------|
| phi3.5 (3.8B) | 85% | ~1s |
| qwen2.5:3b (3B) | 70% | ~0.8s |

Routing improves efficiency by avoiding unnecessary ReAct iterations for simple tasks.
