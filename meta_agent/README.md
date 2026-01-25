# OnsetLab Meta-Agent

The Meta-Agent is a LangGraph-powered backend service that discovers MCP servers and generates Colab notebooks for building AI agents.

## Overview

```
User Problem Statement
         ↓
┌────────────────────────────────────────────────────┐
│              META-AGENT (LangGraph)                │
│                                                    │
│  1. Parse problem → identify required services     │
│  2. Search for MCP servers (Tavily)                │
│  3. Evaluate results → MCP or API fallback         │
│  4. Extract tool schemas from docs                 │
│  5. Generate token setup guides                    │
│  6. Generate Colab notebook                        │
└────────────────────────────────────────────────────┘
         ↓
Colab Notebook + Token Guides
```

## Installation

```bash
cd meta_agent
pip install -r requirements.txt
```

## Required API Keys

| Key | Purpose | Get it from |
|-----|---------|-------------|
| `OPENAI_API_KEY` | LLM calls (parsing, extraction) | https://platform.openai.com/api-keys |
| `TAVILY_API_KEY` | Web search for MCP servers | https://tavily.com/ |
| `GITHUB_TOKEN` (optional) | Upload notebook to Gist | https://github.com/settings/tokens |

## Usage

### Run as API Server

```bash
# Set environment variables
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...

# Start the server
uvicorn meta_agent.api.server:app --reload --port 8000

# Or
python -m meta_agent.api.server
```

API Docs: http://localhost:8000/docs

### Use Programmatically

```python
from meta_agent import run_meta_agent

result = await run_meta_agent(
    problem_statement="I need an agent that manages my Google Calendar",
    anthropic_api_key="sk-ant-...",
    tavily_api_key="tvly-...",
)

# Result contains:
# - colab_notebook: The generated notebook JSON
# - mcp_servers: Discovered MCP servers
# - api_servers: Services needing API implementation
# - token_guides: Setup instructions
# - tool_schemas: All discovered tools
```

### Run Tests

```bash
# Set environment variables
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...

# Run with default problem
python -m meta_agent.test_meta_agent

# Run with custom problem
python -m meta_agent.test_meta_agent "I need an agent that sends Slack messages"
```

## API Endpoints

### `POST /api/generate-agent`

Generate a Colab notebook for building an AI agent.

**Request:**
```json
{
    "problem_statement": "I need an agent that manages my calendar",
    "anthropic_api_key": "sk-ant-...",
    "tavily_api_key": "tvly-...",
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
    "mcp_servers": [...],
    "api_servers": [...],
    "token_guides": [...],
    "tool_count": 15,
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
│   ├── parse_problem.py
│   ├── search_mcp.py
│   ├── evaluate_mcp.py
│   ├── extract_schemas.py
│   ├── mark_as_api.py
│   ├── compile_results.py
│   ├── generate_guides.py
│   └── generate_notebook.py
├── tools/               # LangChain tools
│   ├── tavily_search.py
│   ├── github_tools.py
│   └── npm_tools.py
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
[has services?] ──No──> compile_results
     │ Yes                    │
     ▼                        │
search_mcp_servers            │
     │                        │
     ▼                        │
evaluate_mcp_results          │
     │                        │
 ┌───┴───┐                    │
 ▼       ▼                    │
good   no_mcp                 │
 │       │                    │
 ▼       ▼                    │
extract  mark_as_api          │
schemas      │                │
 │           │                │
 └─────┬─────┘                │
       │                      │
  [more services?] ──Yes──> loop
       │ No                   │
       └──────────────────────┘
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

## License

MIT
