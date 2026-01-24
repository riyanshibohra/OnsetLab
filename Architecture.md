# OnsetLab Agent Builder - Complete Architecture Specification

**Version**: 2.0
**Date**: January 11, 2026
**Author**: Riyanshi Bohra
**Status**: Design Phase

---

## Executive Summary

OnsetLab Agent Builder is a **meta-agent system** that transforms problem statements into production-ready AI agents. The architecture consists of three layers:

1. **Frontend Website**: User interface for problem statement input


2. **Meta-Agent Backend**: Intelligent system that researches MCP servers and generates Colab notebooks


3. **SDK Package**: Published library that executes the fine-tuning pipeline and generates agent runtime



**User Journey**: Problem statement → Meta-agent research → Generated Colab notebook → Execute pipeline → Download agent → Deploy locally

**Key Innovation**: A meta-agent that writes code for building agents, eliminating manual MCP research and pipeline configuration.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Frontend (onsetlab.app)                           │
│                                                              │
│  User Input:                                                │
│  • Problem Statement                                        │
│  • OpenAI/Claude API Key                                    │
│  • [Optional] Preferences                                   │
│                                                              │
│  Output: Generated Colab Notebook URL                       │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP POST
                 │
┌────────────────▼────────────────────────────────────────────┐
│ LAYER 2: Meta-Agent Backend                                │
│                                                              │
│  Tool 1: web_search                                         │
│  • Research MCP servers for problem domain                  │
│  • Find authentication requirements                         │
│  • Analyze tool capabilities                                │
│                                                              │
│  Tool 2: code_generator                                     │
│  • Generate complete Colab notebook                         │
│  • Include setup, configuration, execution                  │
│  • Add documentation and instructions                       │
│                                                              │
│  Output: Colab notebook (uploaded to Gist)                  │
└────────────────┬────────────────────────────────────────────┘
                 │ User opens notebook in Colab
                 │
┌────────────────▼────────────────────────────────────────────┐
│ LAYER 3: SDK Package (onsetlab-agent-builder)              │
│                                                              │
│  pip install onsetlab-agent-builder                         │
│                                                              │
│  Pipeline Execution:                                        │
│  1. MCP Tool Discovery                                      │
│  2. System Prompt Generation (via LLM)                      │
│  3. Synthetic Training Data Generation (via LLM)            │
│  4. Model Fine-Tuning (Unsloth/OnsetLab)                   │
│  5. Agent Runtime Packaging                                 │
│                                                              │
│  Output: Agent runtime files (downloadable)                 │
└────────────────┬────────────────────────────────────────────┘
                 │ User downloads and deploys
                 │
┌────────────────▼────────────────────────────────────────────┐
│ LAYER 4: Generated Agent (User's Machine)                  │
│                                                              │
│  Runtime Components:                                        │
│  • Fine-tuned SLM (GGUF format)                            │
│  • MCP Client (connects to servers)                         │
│  • Agent Runtime (inference + orchestration)                │
│  • System Prompt                                            │
│  • Configuration files                                      │
│                                                              │
│  Usage: python agent.py                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Frontend Website

### Purpose

User-facing interface for initiating agent generation. Collects problem statement and API credentials, triggers meta-agent backend.

### Technology Stack

- **Framework**: Next.js 14 (React + TypeScript)


- **Styling**: Tailwind CSS


- **Hosting**: Vercel


- **State Management**: React hooks (useState, useEffect)



### User Interface Design

**Main Page**:

```
┌─────────────────────────────────────────────────────────────┐
│                    OnsetLab Agent Builder                    │
│        From Problem Statement to Production Agent           │
│                      in 15 Minutes                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  What should your agent do?                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ I need an agent that can:                              │ │
│  │ 1. Check my calendar for upcoming events               │ │
│  │ 2. Send email reminders before meetings                │ │
│  │ 3. Summarize my schedule on demand                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  LLM API Key (for synthesis only)                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ sk-••••••••••••••••••••••••••••                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Provider: ○ OpenAI  ○ Claude                               │
│                                                              │
│  [Advanced Options ▼]                                       │
│                                                              │
│                   [ Generate Agent ]                         │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  ✓ Your agent will run locally (no API costs)               │
│  ✓ API key only used for training data generation           │
│  ✓ Fine-tuning runs on free Google Colab GPU                │
└─────────────────────────────────────────────────────────────┘
```

**Results Page**:

```
┌─────────────────────────────────────────────────────────────┐
│  🎉 Your Agent Notebook is Ready!                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Recommended MCP Servers:                                   │
│  ✓ @modelcontextprotocol/server-google-calendar            │
│  ✓ @modelcontextprotocol/server-gmail                      │
│                                                              │
│  Estimated Build Time: ~15 minutes                          │
│                                                              │
│  [ Open in Google Colab ]                                   │
│                                                              │
│  Next Steps:                                                │
│  1. Click the button above to open the notebook             │
│  2. Get your MCP server access tokens (see guides below)    │
│  3. Run all cells in the notebook                           │
│  4. Download your agent and deploy locally                  │
│                                                              │
│  Setup Guides:                                              │
│  → How to get Google Calendar MCP token                     │
│  → How to get Gmail MCP token                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Functional Description

The frontend application serves as the primary interaction point for users. It collects the user's problem statement and their LLM API key, along with optional advanced preferences. Upon submission, it validates the input and sends a request to the Meta-Agent Backend. Once the backend processes the request and generates a Colab notebook, the frontend displays a results page, providing the URL to the generated notebook, a list of recommended MCP servers, and estimated build time. It also guides the user through the next steps for executing the agent building pipeline.

### Deployment

The frontend is deployed as a serverless application on Vercel, leveraging its integration with Next.js for efficient hosting and scaling.

---

## Component 2: Meta-Agent Backend

### Purpose

Intelligent backend that researches MCP servers and generates customized Colab notebooks. Acts as the "brain" that understands user intent and produces executable code.

### Technology Stack

- **Framework**: FastAPI (Python 3.11+)


- **LLM Integration**: Anthropic Claude API / OpenAI API


- **Tools**: Web search (Exa/Tavily), Code generation


- **Hosting**: Modal (serverless) or Railway


- **Storage**: GitHub Gists (for notebook hosting)



### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Application                                         │
│                                                              │
│  POST /api/generate-agent                                   │
│    ↓                                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Meta-Agent Orchestrator                              │   │
│  │                                                        │   │
│  │  1. Parse problem statement                           │   │
│  │  2. Execute Tool 1: web_search                        │   │
│  │     → Research MCP servers                            │   │
│  │     → Find auth requirements                          │   │
│  │  3. Execute Tool 2: code_generator                    │   │
│  │     → Generate Colab notebook                         │   │
│  │  4. Upload notebook to GitHub Gist                    │   │
│  │  5. Return Colab URL                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Tools:                                                      │
│  • web_search: Query Exa/Tavily for MCP servers            │
│  • code_generator: Generate Python notebook code            │
└─────────────────────────────────────────────────────────────┘
```

### Functional Description

The Meta-Agent Backend is a serverless application responsible for the core intelligence of the OnsetLab Agent Builder. It receives a problem statement and LLM API key from the frontend. Its primary function is to orchestrate a multi-step process:

1. **MCP Server Research**: It utilizes a large language model (LLM) in conjunction with a web search tool (e.g., Exa) to identify relevant Model Context Protocol (MCP) servers that can address the user's problem statement. This involves understanding the required capabilities, finding official and community servers, and extracting details like package names, authentication types, and setup guide URLs.


2. **Colab Notebook Generation**: Based on the identified MCP servers and the problem statement, the meta-agent generates a complete Google Colab notebook. This notebook is pre-configured to use the OnsetLab Agent Builder SDK, including cells for installation, configuration (with placeholders for API keys and MCP tokens), execution of the agent building pipeline, testing, and downloading the final agent.


3. **Notebook Hosting**: The generated Colab notebook is uploaded to a GitHub Gist, and its public URL is returned to the frontend.



The system employs an agentic loop where the LLM can dynamically decide to perform multiple web searches to gather comprehensive information before generating the final notebook. This ensures robust and accurate server discovery and notebook customization.

### Deployment

The Meta-Agent Backend is designed for serverless deployment, primarily using Modal, which allows it to scale automatically based on demand and efficiently manage computational resources for LLM interactions and web searches.

---

## Component 3: SDK Package (onsetlab-agent-builder)

### Purpose

Published Python package that executes the complete agent-building pipeline. Users install this in their Colab notebooks.

### Package Structure

```
onsetlab-agent-builder/
├── onsetlab_agent_builder/
│   ├── __init__.py
│   ├── builder.py              # Main AgentBuilder class
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── client.py           # MCP protocol client
│   │   └── discovery.py        # Tool enumeration
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── prompt_generator.py # System prompt generation
│   │   └── data_generator.py   # Training data synthesis
│   ├── training/
│   │   ├── __init__.py
│   │   ├── unsloth_trainer.py  # Fine-tuning with Unsloth
│   │   └── onsetlab_wrapper.py # OnsetLab integration
│   └── runtime/
│       ├── __init__.py
│       ├── packager.py         # Agent file generation
│       └── templates/          # agent.py, mcp_client.py templates
├── examples/
│   ├── calendar_agent.py
│   └── github_agent.py
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

### Core API

The `onsetlab-agent-builder` SDK provides a high-level Python interface, primarily through the `AgentBuilder` class, to orchestrate the entire agent creation pipeline.

The `AgentBuilder` class is initialized with the user's problem statement, a list of target MCP servers, their corresponding access tokens, and an LLM API key. It encapsulates the following sequential steps:

1. **MCP Tool Discovery**: Connects to the specified MCP servers using provided tokens to enumerate and retrieve the schemas of all available tools. This step identifies the specific functionalities the agent can leverage.


2. **System Prompt Generation**: Utilizes an LLM to generate a comprehensive system prompt. This prompt instructs the fine-tuned model on its role, how to use the discovered tools, and how to interact with users, ensuring effective and safe agent behavior.


3. **Synthetic Training Data Generation**: Leverages an LLM to create a dataset of synthetic conversational examples. These examples demonstrate how the agent should respond to user queries, including when and how to call the discovered MCP tools. The data is generated based on the problem statement, tool schemas, and the system prompt.


4. **Model Fine-Tuning**: Takes a pre-selected base language model (e.g., Phi-3.5-mini) and fine-tunes it using the generated synthetic training data. This process typically uses efficient fine-tuning techniques like LoRA (Low-Rank Adaptation) via libraries like Unsloth, making it feasible on readily available GPU resources (e.g., Google Colab T4 GPUs). The output is a specialized, fine-tuned model optimized for the agent's specific tasks.


5. **Agent Runtime Packaging**: Assembles all necessary components into a standalone agent runtime directory. This includes the fine-tuned model, the generated system prompt, MCP client code, configuration files, and a `README.md` with usage instructions.



Upon successful completion of these steps, the `build()` method returns an `Agent` instance, which can then be used for local testing or saved for deployment. The `Agent` class provides methods for interacting with the generated agent, such as `test()` for querying and `save()` for exporting the agent's files to a specified directory.

### Publishing to PyPI

The SDK is published as a Python package to PyPI, making it easily installable via `pip` in any Python environment, including Google Colab.

---

## Component 4: Generated Agent Runtime

### Purpose

The final deliverable: a standalone agent that runs locally without API dependencies.

### Directory Structure

```
my_agent/
├── model/
│   ├── model.gguf              # Quantized fine-tuned model (2-4GB)
│   ├── config.json             # Model configuration
│   └── tokenizer.json          # Tokenizer
│
├── mcp_client.py               # MCP protocol client implementation
├── agent.py                    # Main agent runtime
├── system_prompt.txt           # Generated system prompt
├── config.yaml                 # Agent configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── README.md                   # Setup & usage guide
│
└── examples/
    ├── basic_usage.py
    ├── cli_interface.py
    └── api_server.py
```

### Agent Runtime Description

The generated agent is a self-contained application designed for local execution. It consists of several key components:

- **Fine-tuned Model**: A quantized Small Language Model (SLM) in GGUF format, specifically trained to understand user intent and generate tool calls or direct responses. This model is loaded locally using inference engines like Ollama.


- **MCP Client**: A Python module (`mcp_client.py`) that implements the Model Context Protocol. It handles communication with various MCP servers, enabling the agent to execute external tools (e.g., Google Calendar API calls) by abstracting the underlying server interactions and authentication.


- **Agent Orchestration Logic**: The main `agent.py` script contains the core logic for the agent. It manages conversation history, constructs prompts for the SLM, parses tool calls from the SLM's output, executes these tools via the MCP client, and integrates tool results back into the conversation to generate a final, coherent response to the user.


- **System Prompt**: A `system_prompt.txt` file containing the detailed instructions and guidelines for the fine-tuned model, ensuring it adheres to the desired behavior and safety protocols.


- **Configuration**: A `config.yaml` file stores agent-specific settings, including the model name, MCP server tokens (often loaded from environment variables for security), and inference parameters.


- **Documentation and Examples**: A `README.md` provides comprehensive instructions for setting up, configuring, and using the agent. It covers dependency installation, model loading, token configuration, and various usage modes (CLI, Python API, API server). Example scripts demonstrate different ways to interact with the agent.



The agent operates by taking a user query, augmenting it with conversation history and the system prompt, and feeding it to the local SLM. If the SLM determines a tool call is necessary, it outputs a structured tool call. The agent then executes this tool call through the MCP client, receives the result, and uses the SLM again to synthesize a natural language response incorporating the tool's output. This design ensures that the agent can perform complex tasks by intelligently combining language understanding with external capabilities.

---

## Data Flow Diagram

```

User Problem Statement
↓
┌───────────────────────────────────────────────────────────┐
│ Frontend: Collect input + API key │
└───────────────┬───────────────────────────────────────────┘
↓ HTTP POST
┌───────────────────────────────────────────────────────────┐
│ Meta-Agent Backend │
│ │
│ ┌──────────────────────────────────────────────────┐ │
│ │ Tool 1: web_search │ │
│ │ • Query: "MCP servers for calendar management" │ │
│ │ • Results: [@modelcontextprotocol/server-gcal] │ │
│ └──────────────────────────────────────────────────┘ │
│ ↓ │
│ ┌──────────────────────────────────────────────────┐ │
│ │ Tool 2: code_generator │ │
│ │ • Input: Problem + MCP servers │ │
│ │ • Output: Complete Colab notebook │ │
│ └──────────────────────────────────────────────────┘ │
│ ↓ │
│ Upload to GitHub Gist → Return Colab URL │
└───────────────┬───────────────────────────────────────────┘
↓
┌───────────────────────────────────────────────────────────┐
│ User Opens Colab Notebook │
│ │
│ Cell 1: Install onsetlab-agent-builder │
│ Cell 2: Configure (API keys, MCP tokens) │
│ Cell 3: Execute builder.build() │
└───────────────┬───────────────────────────────────────────┘
↓
┌───────────────────────────────────────────────────────────┐
│ SDK Pipeline Execution (onsetlab-agent-builder) │
│ │
│ Step 1: MCP Discovery │
│ • Connect to google-calendar MCP server │
│ • Extract tools: [get_events, create_event, ...] │
│ │
│ Step 2: System Prompt Generation (via GPT-4) │
│ • Input: Problem + tool schemas │
│ • Output: 2000-token system prompt │
│ │
│ Step 3: Training Data Generation (via GPT-4) │
│ • Generate 500 examples │
│ • Validate against schemas │
│ • Output: training_data.jsonl │
│ │
│ Step 4: Fine-Tuning (Unsloth) │
│ • Base: phi-3.5-mini (3.8B) │
│ • Method: LoRA │
│ • Time: ~15 minutes on T4 │
│ • Output: model.gguf (quantized) │
│ │
│ Step 5: Agent Packaging │
│ • Generate agent.py from template │
│ • Generate mcp_client.py │
│ • Copy system_prompt.txt │
│ • Create config.yaml, README.md │
│ • Output: my_agent/ directory │
└───────────────┬───────────────────────────────────────────┘
↓
┌───────────────────────────────────────────────────────────┐
│ Download Agent (ZIP file) │
└───────────────┬───────────────────────────────────────────┘
↓
┌───────────────────────────────────────────────────────────┐
│ User's Local Machine │
│ │
│ $ unzip my_agent.zip │
│ $ cd my_agent │
│ $ pip install -r requirements.txt │
│ $ ollama create calendar-agent -f Modelfile │
│ $ python agent.py │
│ │
│ > Agent: How can I help you? │
│ > User: What's on my calendar tomorrow? │
│ > Agent: [Calls get_events tool via MCP] │
│ > You have 2 meetings: ... │
└───────────────────────────────────────────────────────────┘

```

---

## Technology Stack Summary

| Layer | Technology | Purpose |
| --- | --- | --- |
| **Frontend** | Next.js + Vercel | User interface |
| **Meta-Agent** | FastAPI + Modal | Research & code generation |
| **LLM** | Claude 3.5 Sonnet / GPT-4 | Tool research, code gen |
| **Web Search** | Exa API | MCP server discovery |
| **SDK** | Python package (PyPI) | Pipeline orchestration |
| **Fine-Tuning** | Unsloth + Colab T4 | Model training |
| **Base Models** | Phi-3.5-mini, Llama-3.2-3B | Foundation models |
| **MCP** | @modelcontextprotocol/sdk | Tool protocol |
| **Inference** | Ollama / llama.cpp | Local deployment |
| **Storage** | GitHub Gists | Notebook hosting |

---

## Future Enhancements

### V2 Features

- **Visual agent builder**: No-code interface


- **Agent monitoring dashboard**: Usage analytics, error tracking


- **Multi-agent orchestration**: Agents calling other agents


- **Continuous learning**: Improve from user feedback



### V3 Features

- **Meta-learning agents**: Handle new tools zero-shot


- **Agent marketplace**: Buy/sell pre-trained agents


- **Federated training**: Learn from all users' agents


- **Enterprise features**: Team collaboration, access control



---

## Conclusion

This architecture delivers a complete meta-agent system that automates the entire agent-building process. Key advantages:

1. **Separation of Concerns**: Frontend → Meta-Agent → SDK → Generated Agent


2. **User Flexibility**: Use SDK directly or via frontend


3. **Cost Efficiency**: LLM only for synthesis, not runtime


4. **Scalability**: Each component can scale independently


5. **Extensibility**: Easy to add new MCP servers, base models, deployment options



The meta-agent approach (agent that writes agent-building code) eliminates manual research and configuration, making AI agent development accessible to anyone with a problem statement.