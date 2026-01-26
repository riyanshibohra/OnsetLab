# OnsetLab Meta-Agent (Registry-Based v2.0)

The Meta-Agent is a LangGraph-powered backend service that loads tools from a curated registry and generates Colab notebooks for building AI agents.

## Overview

```
User Problem Statement
         ↓
┌────────────────────────────────────────────────────┐
│         META-AGENT (Registry-Based)               │
│                                                    │
│  1. Parse problem → identify required services     │
│  2. Load tools from registry (JSON files)         │
│  3. Filter tools → LLM selects 15-20 relevant      │
│  4. Human-in-the-Loop → User reviews/approves    │
│  5. Generate token setup guides                   │
│  6. Generate Colab notebook                       │
└────────────────────────────────────────────────────┘
         ↓
Colab Notebook + Token Guides
```

## Key Features

- ✅ **Registry-Based**: No web search, uses verified tool schemas
- ✅ **Human-in-the-Loop**: User reviews and approves selected tools
- ✅ **Simplified**: 6 nodes, 1 decision point, 3 LLM calls
- ✅ **Fast**: No discovery delays, direct registry loading
- ✅ **Reliable**: Pre-verified tools with correct schemas

## Installation

```bash
cd meta_agent
pip install -r requirements.txt
```

## Required API Keys

| Key | Purpose | Get it from |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | LLM calls (parsing, filtering) | https://console.anthropic.com/ |
| `GITHUB_TOKEN` (optional) | Upload notebook to Gist | https://github.com/settings/tokens |

**Note:** No Tavily API key needed anymore! We use a registry instead of web search.

## Usage

### Run as API Server

```bash
# Set environment variables
export ANTHROPIC_API_KEY=sk-ant-...

# Start the server
uvicorn meta_agent.api.server:app --reload --port 8000

# Or
python -m meta_agent.api.server
```

API Docs: http://localhost:8000/docs

### Use Programmatically

```python
from meta_agent.graph import run_meta_agent_sync

result = run_meta_agent_sync(
    problem_statement="I need an agent that manages my Google Calendar and sends Slack messages",
    anthropic_api_key="sk-ant-...",
)

# Result contains:
# - colab_notebook: The generated notebook JSON
# - final_tools: User-approved tools
# - mcp_servers: MCP server configs from registry
# - token_guides: Setup instructions
# - registry_services: Services loaded
```

### With Human-in-the-Loop

```python
from meta_agent.graph import run_with_hitl

result = run_with_hitl(
    problem_statement="Manage GitHub issues",
    anthropic_api_key="sk-ant-...",
)

# Will pause and show you selected tools:
# "📋 Selected 8 tools: ..."
# "Your feedback: " ← Type here
# Options:
#   - "looks good" → Continue
#   - "add search_repositories" → Add that tool
#   - "remove list_issues" → Remove that tool
```

## API Endpoints

### `POST /api/generate-agent`

Generate a Colab notebook for building an AI agent.

**Request:**
```json
{
    "problem_statement": "I need an agent that manages my calendar",
    "anthropic_api_key": "sk-ant-...",
    "github_token": "ghp-...",  // Optional
    "upload_to_gist": true       // Optional
}
```

**Response:**
```json
{
    "success": true,
    "colab_notebook": "...",
    "colab_notebook_url": "https://colab.research.google.com/gist/...",
    "final_tools": [...],
    "mcp_servers": [...],
    "token_guides": [...],
    "registry_services": ["github", "slack"],
    "errors": []
}
```

## Architecture

```
meta_agent/
├── __init__.py          # Main exports
├── state.py             # LangGraph state schemas
├── graph.py             # LangGraph workflow definition
├── nodes/               # Graph nodes
│   ├── parse_problem.py      # Extract services from problem
│   ├── load_registry.py      # Load tools from JSON files
│   ├── filter_tools.py       # LLM selects relevant tools
│   ├── process_feedback.py   # HITL: Process user feedback
│   ├── generate_guides.py    # Generate token setup guides
│   └── generate_notebook.py  # Generate Colab notebook
├── registry/            # Tool registry (JSON files)
│   ├── _builtin_memory.json
│   ├── github.json
│   ├── slack.json
│   ├── google_calendar.json
│   ├── tavily.json
│   ├── filesystem.json
│   └── notion.json
├── utils/               # Utilities
│   └── gist_upload.py
├── api/                 # FastAPI server
│   └── server.py
├── requirements.txt
└── README.md
```

## Flow Diagram

```
parse_problem
     │
     ▼
load_registry      (Load from meta_agent/registry/*.json)
     │
     ▼
filter_tools       (LLM selects 15-20 relevant tools)
     │
     ▼
process_feedback   ← HITL: User reviews tools
     │
     ├─── approved ────► generate_token_guides
     │
     ├─── add_tools ────► load_registry (loop)
     │
     └─── remove_tools ──► filter_tools (loop)
                              │
                              ▼
                    generate_token_guides
                              │
                              ▼
                    generate_notebook
                              │
                              ▼
                             END
```

## Registry

The registry contains pre-verified MCP server definitions:

- **memory** (built-in) - 4 tools
- **github** - 17 tools
- **slack** - 5 tools
- **google_calendar** - 6 tools
- **tavily** - 4 tools
- **filesystem** - 8 tools
- **notion** - 10 tools

**Total: 54 tools**

Each registry file (`meta_agent/registry/*.json`) contains:
- Package information (npm/docker/binary)
- Authentication details
- Tool schemas with descriptions
- Setup instructions

## Run Tests

```bash
# Set environment variables
export ANTHROPIC_API_KEY=sk-ant-...

# Test registry loading
python test_meta_agent_registry.py

# Test full flow (requires API key)
python -m meta_agent.test_meta_agent
```

## License

MIT
