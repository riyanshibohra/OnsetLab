import { useState, useEffect } from 'react'
import { ArrowLeft } from 'lucide-react'

const SECTIONS = [
  { id: 'quick-start', label: 'Quick Start' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'built-in-tools', label: 'Built-in Tools' },
  { id: 'mcp-servers', label: 'MCP Servers' },
  { id: 'configuration', label: 'Configuration' },
  { id: 'export-deploy', label: 'Export & Deploy' },
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
          <Code lang="bash">ollama pull qwen2.5:7b</Code>

          <h3>3. Create and run an agent</h3>
          <Code lang="python">{`from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime

agent = Agent("qwen2.5:7b", tools=[Calculator(), DateTime()])

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

          <div className="docs-flow">
            <div className="docs-flow-row">
              <div className="docs-flow-node docs-flow-start">Query</div>
              <span className="docs-flow-arrow">&#x2192;</span>
              <div className="docs-flow-node">Router</div>
            </div>
            <div className="docs-flow-branch">
              <div className="docs-flow-path">
                <span className="docs-flow-path-label">tools needed</span>
                <span className="docs-flow-arrow">&#x2193;</span>
                <div className="docs-flow-node">Planner</div>
                <span className="docs-flow-arrow">&#x2193;</span>
                <div className="docs-flow-node">Executor</div>
                <span className="docs-flow-arrow">&#x2193;</span>
                <div className="docs-flow-node">Solver</div>
              </div>
              <div className="docs-flow-path">
                <span className="docs-flow-path-label">no tools</span>
                <span className="docs-flow-arrow">&#x2193;</span>
                <div className="docs-flow-node">Direct Answer</div>
              </div>
              <div className="docs-flow-path">
                <span className="docs-flow-path-label">plan fails</span>
                <span className="docs-flow-arrow">&#x2193;</span>
                <div className="docs-flow-node docs-flow-fallback">ReAct Fallback</div>
              </div>
            </div>
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

agent = Agent("qwen2.5:7b", tools=[Calculator(), DateTime(), UnitConverter()])`}</Code>
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

agent = Agent("qwen2.5:7b")
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
    model="qwen2.5:7b",       # any Ollama model
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
                  <td>Ollama model name (e.g. "qwen2.5:7b")</td>
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

        {/* Export & Deploy */}
        <section id="export-deploy">
          <h1>Export & Deploy</h1>
          <p>
            OnsetLab agents can be exported in multiple formats for deployment. Use the playground's export
            buttons or the SDK directly.
          </p>

          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Format</th>
                  <th>What it does</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><InlineCode>YAML</InlineCode></td>
                  <td>Exports agent config (model, tools, MCP servers) as a portable YAML file</td>
                </tr>
                <tr>
                  <td><InlineCode>Docker</InlineCode></td>
                  <td>Generates a Dockerfile + docker-compose.yml with Ollama bundled</td>
                </tr>
                <tr>
                  <td><InlineCode>vLLM</InlineCode></td>
                  <td>Docker setup using vLLM for GPU-accelerated inference (5-10x faster)</td>
                </tr>
                <tr>
                  <td><InlineCode>Script</InlineCode></td>
                  <td>Standalone Python script ready to run on any machine with Ollama</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Tested Models */}
        <section id="tested-models">
          <h1>Tested Models</h1>
          <p>
            The following models have been tested and confirmed to work well with OnsetLab's tool-calling
            architecture. Any Ollama-compatible model should work.
          </p>

          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Size</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><InlineCode>qwen2.5:7b</InlineCode></td>
                  <td>7B</td>
                  <td>Best results for tool calling</td>
                </tr>
                <tr>
                  <td><InlineCode>qwen2.5:3b</InlineCode></td>
                  <td>3B</td>
                  <td>Fast, good for simple tasks</td>
                </tr>
                <tr>
                  <td><InlineCode>phi3.5</InlineCode></td>
                  <td>3.8B</td>
                  <td>Solid balance of speed and quality</td>
                </tr>
                <tr>
                  <td><InlineCode>llama3.2:3b</InlineCode></td>
                  <td>3B</td>
                  <td>General purpose</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="docs-note">
            For best results, we recommend <InlineCode>qwen2.5:7b</InlineCode>. Smaller models (3B) work
            well for simple tool calls but may struggle with complex multi-step plans.
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
