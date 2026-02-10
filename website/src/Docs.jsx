import { useState, useEffect } from 'react'
import { ArrowLeft } from 'lucide-react'

const SECTIONS = [
  { id: 'quick-start', label: 'Quick Start' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'built-in-tools', label: 'Built-in Tools' },
  { id: 'mcp-servers', label: 'MCP Servers' },
  { id: 'configuration', label: 'Configuration' },
  { id: 'cli', label: 'CLI' },
  { id: 'export-deploy', label: 'Export & Deploy' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'tested-models', label: 'Tested Models' },
]

function SideNav({ active }) {
  return (
    <nav className="docs-sidenav">
      <a href="/" className="docs-back">
        <ArrowLeft size={14} />
        Back to home
      </a>
      <div className="docs-sidenav-title">Documentation</div>
      {SECTIONS.map(s => (
        <a
          key={s.id}
          href={`#${s.id}`}
          className={`docs-sidenav-link ${active === s.id ? 'active' : ''}`}
        >
          {s.label}
        </a>
      ))}
    </nav>
  )
}

function Code({ children, lang = '' }) {
  return (
    <div className="docs-code">
      {lang && <span className="docs-code-lang">{lang}</span>}
      <pre>{children}</pre>
    </div>
  )
}

function InlineCode({ children }) {
  return <code className="docs-inline-code">{children}</code>
}

export default function Docs() {
  const [active, setActive] = useState('quick-start')

  useEffect(() => {
    document.title = 'OnsetLab | Docs'
  }, [])

  useEffect(() => {
    const handleScroll = () => {
      const offset = 120
      for (let i = SECTIONS.length - 1; i >= 0; i--) {
        const el = document.getElementById(SECTIONS[i].id)
        if (el && el.getBoundingClientRect().top < offset) {
          setActive(SECTIONS[i].id)
          break
        }
      }
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div className="docs-layout">
      <SideNav active={active} />

      <main className="docs-content">
        {/* Quick Start */}
        <section id="quick-start">
          <h1>Quick Start</h1>
          <p>Get a tool-calling agent running locally in under a minute.</p>

          <h3>1. Install OnsetLab</h3>
          <Code lang="bash">pip install onsetlab</Code>

          <h3>2. Install and pull a model with Ollama</h3>
          <p>
            OnsetLab uses <a href="https://ollama.com" target="_blank" rel="noopener noreferrer">Ollama</a> to
            run models locally. Install it, then pull a model:
          </p>
          <Code lang="bash">ollama pull phi3.5</Code>

          <h3>3. Create and run an agent</h3>
          <Code lang="python">{`from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime

agent = Agent("phi3.5", tools=[Calculator(), DateTime()])

result = agent.run("What's 15% tip on $84.50?")
print(result.answer)`}</Code>
          <p>
            The agent routes the query, builds an execution plan, calls the calculator, and returns the answer.
            No prompt engineering required.
          </p>
        </section>

        {/* Architecture */}
        <section id="architecture">
          <h1>Architecture</h1>
          <p>
            OnsetLab uses a hybrid <strong>REWOO + ReAct</strong> architecture. The framework handles planning,
            execution, and error recovery. The model only does what it's good at: one step at a time.
          </p>

          <div className="arch-diagram">
            <img
              src={`https://mermaid.ink/svg/${btoa(`flowchart TD
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
    style A fill:#4a6670,stroke:#4a6670,color:#fff`)}`}
              alt="OnsetLab Architecture: Query → Router → Planner → Executor → Solver → Answer, with Direct Answer and ReAct Fallback branches"
              style={{ width: '100%', maxWidth: '550px', display: 'block', margin: '0 auto' }}
            />
          </div>

          <h3>Router</h3>
          <p>
            The model itself decides the strategy. No regex, no keyword matching. The SLM reads the query and
            available tools, then classifies: <InlineCode>REWOO</InlineCode> (needs tools)
            or <InlineCode>DIRECT</InlineCode> (answer from knowledge). Trivial greetings are caught before
            the model is even called.
          </p>

          <h3>Planner</h3>
          <p>
            Generates a structured <InlineCode>THINK → PLAN</InlineCode> output. Each plan step specifies a tool,
            parameters, and dependencies on previous steps. Tool rules are auto-generated from JSON schemas, so the
            model sees exactly what each tool can do.
          </p>

          <h3>Executor</h3>
          <p>
            Resolves dependencies between steps and runs tool calls in order. If step 2 depends on step 1's output,
            the result is substituted automatically.
          </p>

          <h3>ReAct Fallback</h3>
          <p>
            If REWOO planning fails (bad format, wrong tool, missing params), the agent switches to
            iterative <InlineCode>Thought → Action → Observation</InlineCode> loops.
            Catches edge cases that structured planning misses.
          </p>
        </section>

        {/* Built-in Tools */}
        <section id="built-in-tools">
          <h1>Built-in Tools</h1>
          <p>OnsetLab ships with five tools that cover common use cases out of the box.</p>
          <p className="docs-note" style={{ fontSize: '0.9em', marginTop: '0.5em', opacity: 0.8 }}>
            More tools will be added over time.
          </p>

          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Tool</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><InlineCode>Calculator</InlineCode></td>
                  <td>Math expressions, percentages, sqrt/sin/log</td>
                </tr>
                <tr>
                  <td><InlineCode>DateTime</InlineCode></td>
                  <td>Current time, timezones, date math, day of week</td>
                </tr>
                <tr>
                  <td><InlineCode>UnitConverter</InlineCode></td>
                  <td>Length, weight, temperature, volume, speed, data</td>
                </tr>
                <tr>
                  <td><InlineCode>TextProcessor</InlineCode></td>
                  <td>Word count, find/replace, case transforms, pattern extraction</td>
                </tr>
                <tr>
                  <td><InlineCode>RandomGenerator</InlineCode></td>
                  <td>Random numbers, UUIDs, passwords, dice rolls, coin flips</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>Usage</h3>
          <Code lang="python">{`from onsetlab.tools import Calculator, DateTime, UnitConverter

agent = Agent("phi3.5", tools=[Calculator(), DateTime(), UnitConverter()])`}</Code>
          <p>
            Each tool auto-generates its own JSON schema, which the planner uses to understand available
            functions and parameter types.
          </p>
        </section>

        {/* MCP Servers */}
        <section id="mcp-servers">
          <h1>MCP Servers</h1>
          <p>
            Connect any <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer">MCP-compatible</a> server
            to give your agent access to external tools like GitHub, Slack, Notion, and more.
          </p>

          <h3>From the built-in registry</h3>
          <Code lang="python">{`from onsetlab import Agent, MCPServer

server = MCPServer.from_registry("filesystem", extra_args=["/path/to/dir"])

agent = Agent("phi3.5")
agent.add_mcp_server(server)

result = agent.run("List all Python files in the directory")
print(result.answer)

agent.disconnect_mcp_servers()`}</Code>

          <h3>Any MCP server from npm</h3>
          <Code lang="python">{`# Web search (requires API key)
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

agent.add_mcp_server(server)`}</Code>

          <h3>Built-in registry</h3>
          <p>
            The following servers are available out of the box
            via <InlineCode>MCPServer.from_registry()</InlineCode>:
          </p>
          <div className="docs-chips">
            {['filesystem', 'github', 'slack', 'notion', 'google_calendar', 'tavily'].map(s => (
              <span key={s} className="docs-chip">{s}</span>
            ))}
          </div>
          <p>
            Some servers require environment variables for authentication (e.g. <InlineCode>GITHUB_PERSONAL_ACCESS_TOKEN</InlineCode>,{' '}
            <InlineCode>SLACK_MCP_XOXP_TOKEN</InlineCode>). Check the registry JSON files for required config.
          </p>
        </section>

        {/* Configuration */}
        <section id="configuration">
          <h1>Configuration</h1>
          <p>The <InlineCode>Agent</InlineCode> constructor accepts the following options:</p>

          <Code lang="python">{`agent = Agent(
    model="phi3.5",            # any Ollama model
    tools=[...],               # built-in tools
    memory=True,               # conversation memory
    verify=True,               # pre-execution plan verification
    routing=True,              # model-driven strategy selection
    react_fallback=True,       # automatic fallback on planning failure
    debug=False,               # verbose logging
)`}</Code>

          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Default</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><InlineCode>model</InlineCode></td>
                  <td>required</td>
                  <td>Ollama model name (e.g. "phi3.5", "qwen2.5:7b")</td>
                </tr>
                <tr>
                  <td><InlineCode>tools</InlineCode></td>
                  <td><InlineCode>[]</InlineCode></td>
                  <td>List of tool instances to make available</td>
                </tr>
                <tr>
                  <td><InlineCode>memory</InlineCode></td>
                  <td><InlineCode>True</InlineCode></td>
                  <td>Enable conversation memory across runs</td>
                </tr>
                <tr>
                  <td><InlineCode>verify</InlineCode></td>
                  <td><InlineCode>True</InlineCode></td>
                  <td>Validate the execution plan before running tools</td>
                </tr>
                <tr>
                  <td><InlineCode>routing</InlineCode></td>
                  <td><InlineCode>True</InlineCode></td>
                  <td>Let the model decide between REWOO and DIRECT strategies</td>
                </tr>
                <tr>
                  <td><InlineCode>react_fallback</InlineCode></td>
                  <td><InlineCode>True</InlineCode></td>
                  <td>Fall back to ReAct if REWOO planning fails</td>
                </tr>
                <tr>
                  <td><InlineCode>debug</InlineCode></td>
                  <td><InlineCode>False</InlineCode></td>
                  <td>Print detailed logs of planning, tool calls, and fallback</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* CLI */}
        <section id="cli">
          <h1>CLI</h1>
          <p>
            OnsetLab includes a command-line interface for interactive chat, benchmarking, and exporting agents.
          </p>

          <h3>Interactive mode</h3>
          <p>Start a chat session with your agent directly in the terminal.</p>
          <Code lang="bash">{`python -m onsetlab
python -m onsetlab --model qwen3-a3b
python -m onsetlab --debug`}</Code>

          <p>Once running, the following commands are available:</p>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Command</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr><td><InlineCode>/tools</InlineCode></td><td>List available tools</td></tr>
                <tr><td><InlineCode>/memory</InlineCode></td><td>Show conversation history</td></tr>
                <tr><td><InlineCode>/clear</InlineCode></td><td>Clear conversation memory</td></tr>
                <tr><td><InlineCode>/debug</InlineCode></td><td>Toggle debug mode (shows planning, tool calls)</td></tr>
                <tr><td><InlineCode>/quit</InlineCode></td><td>Exit the session</td></tr>
              </tbody>
            </table>
          </div>

          <h3>CLI options</h3>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Flag</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr><td><InlineCode>--model</InlineCode></td><td>Ollama model name (default: phi3.5)</td></tr>
                <tr><td><InlineCode>--no-memory</InlineCode></td><td>Disable conversation memory</td></tr>
                <tr><td><InlineCode>--no-verify</InlineCode></td><td>Skip plan verification step</td></tr>
                <tr><td><InlineCode>--debug</InlineCode></td><td>Enable verbose logging</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Export & Deploy */}
        <section id="export-deploy">
          <h1>Export & Deploy</h1>
          <p>
            OnsetLab agents can be exported in multiple formats for deployment. Use the playground's export
            buttons or the CLI.
          </p>

          <h3>YAML config</h3>
          <p>
            Export your agent's configuration as a portable YAML file. Useful for version control
            or loading the same setup across machines.
          </p>
          <Code lang="bash">{`python -m onsetlab export --format config -o agent_config.yaml`}</Code>
          <p>This generates a file like:</p>
          <Code lang="yaml">{`version: '1.0'
onsetlab:
  model: qwen3-a3b
  settings:
    memory: true
    verify: true
    routing: true
    react_fallback: true
tools:
  - name: Calculator
    class: Calculator
  - name: DateTime
    class: DateTime`}</Code>

          <h3>Docker</h3>
          <p>
            Generates a complete Docker package with Dockerfile, docker-compose.yml,
            entrypoint script, and a requirements file. Ollama runs as a sidecar container.
          </p>
          <Code lang="bash">{`python -m onsetlab export --format docker -o ./my-agent

# Then deploy:
cd my-agent
docker-compose up --build`}</Code>
          <p>
            The agent exposes a <InlineCode>/chat</InlineCode> endpoint on port 8000:
          </p>
          <Code lang="bash">{`curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is 15% tip on $84.50?"}'`}</Code>

          <h3>vLLM (GPU)</h3>
          <p>
            Same as Docker but uses vLLM for GPU-accelerated inference instead of Ollama.
            Requires an NVIDIA GPU with CUDA 12.1+ and the NVIDIA Container Toolkit.
          </p>
          <Code lang="bash">{`# From the playground: click Export > vLLM
# The ZIP contains everything needed

cd onsetlab_vllm_docker
docker-compose up --build`}</Code>
          <p>
            vLLM automatically maps Ollama model names to HuggingFace model IDs
            (e.g. <InlineCode>qwen3-a3b</InlineCode> becomes <InlineCode>Qwen/Qwen3-A3B</InlineCode>).
          </p>

          <h3>Standalone script</h3>
          <p>
            Exports a single Python file that runs your agent. No Docker, no compose.
            Just Python + Ollama.
          </p>
          <Code lang="bash">{`python -m onsetlab export --format binary -o agent.py

# Run interactively:
python agent.py

# Or single query:
python agent.py "What is 2 to the power of 10?"`}</Code>

          <h3>Loading from config</h3>
          <p>
            You can also export from an existing YAML config instead of specifying options manually:
          </p>
          <Code lang="bash">{`python -m onsetlab export --format docker -o ./my-agent --config agent_config.yaml`}</Code>
        </section>

        {/* Benchmark */}
        <section id="benchmark">
          <h1>Benchmark</h1>
          <p>
            OnsetLab includes a built-in benchmark for quickly validating whether a model handles tool calling
            correctly. It tests tool selection (does the model pick the right tool?) and parameter extraction
            (does it fill in the right values?).
          </p>
          <p className="docs-note" style={{ fontSize: '0.9em', marginTop: '0.5em', opacity: 0.8 }}>
            This is a quick validation suite, not a comprehensive evaluation. It covers basic tool-calling
            scenarios across the built-in tools to help you verify a model works before deploying.
          </p>

          <h3>Run on a single model</h3>
          <Code lang="bash">{`python -m onsetlab benchmark
python -m onsetlab benchmark --model qwen3-a3b
python -m onsetlab benchmark --model qwen3-a3b --verbose`}</Code>

          <h3>Compare multiple models</h3>
          <Code lang="bash">{`python -m onsetlab benchmark --compare qwen3-a3b,qwen2.5:7b,phi3.5`}</Code>
          <p>
            This runs the full test suite on each model and prints a comparison table with accuracy
            and average latency.
          </p>

          <h3>Filter by category</h3>
          <Code lang="bash">{`python -m onsetlab benchmark --categories tool_selection
python -m onsetlab benchmark --categories param_extraction`}</Code>

          <h3>Programmatic usage</h3>
          <Code lang="python">{`from onsetlab import Benchmark

# Single model
result = Benchmark.run(model="qwen3-a3b", verbose=True)
result.print_summary()

print(f"Accuracy: {result.accuracy:.1%}")
print(f"Avg latency: {result.avg_time_ms:.0f}ms")

# Compare models
results = Benchmark.compare(
    models=["qwen3-a3b", "qwen2.5:7b", "phi3.5"],
    verbose=True
)
Benchmark.print_comparison(results)`}</Code>

          <h3>What it tests</h3>
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Tests</th>
                  <th>What it checks</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Tool selection</td>
                  <td>10</td>
                  <td>Does the model pick the correct tool for the query?</td>
                </tr>
                <tr>
                  <td>Param extraction</td>
                  <td>10</td>
                  <td>Does it extract the right parameter values?</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            Covers Calculator, DateTime, UnitConverter, and TextProcessor across both categories.
            The scorer handles common variations (e.g. "F" vs "fahrenheit", "**" vs "^").
          </p>
        </section>

        {/* Tested Models */}
        <section id="tested-models">
          <h1>Tested Models</h1>
          <p>
            The following models have been tested with OnsetLab's tool-calling architecture.
            Any Ollama-compatible model should work, but results vary by model.
          </p>

          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Size</th>
                  <th>RAM</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><InlineCode>phi3.5</InlineCode></td>
                  <td>3.8B</td>
                  <td>4GB+</td>
                  <td>Default. Good balance of speed and quality</td>
                </tr>
                <tr>
                  <td><InlineCode>qwen2.5:3b</InlineCode></td>
                  <td>3B</td>
                  <td>4GB+</td>
                  <td>Fast, good for simple tasks</td>
                </tr>
                <tr>
                  <td><InlineCode>qwen2.5:7b</InlineCode></td>
                  <td>7B</td>
                  <td>8GB+</td>
                  <td>Strong tool calling</td>
                </tr>
                <tr>
                  <td><InlineCode>qwen3-a3b</InlineCode></td>
                  <td>MoE (3B active / 30B total)</td>
                  <td>16GB+</td>
                  <td>Best tool calling accuracy</td>
                </tr>
                <tr>
                  <td><InlineCode>llama3.2:3b</InlineCode></td>
                  <td>3B</td>
                  <td>4GB+</td>
                  <td>General purpose</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="docs-note">
            The default is <InlineCode>phi3.5</InlineCode>, which runs on most hardware. For stronger tool calling,
            use <InlineCode>qwen2.5:7b</InlineCode> (8GB+ RAM) or <InlineCode>qwen3-a3b</InlineCode> (16GB+ RAM).
            Verify any model with <InlineCode>python -m onsetlab benchmark --model your-model</InlineCode>.
          </p>
        </section>

        <footer className="docs-footer">
          <p>
            onsetlab · Apache 2.0 ·{' '}
            <a href="https://github.com/riyanshibohra/OnsetLab" target="_blank" rel="noopener noreferrer">GitHub</a> ·{' '}
            <a href="https://pypi.org/project/onsetlab/" target="_blank" rel="noopener noreferrer">PyPI</a>
          </p>
        </footer>
      </main>
    </div>
  )
}
