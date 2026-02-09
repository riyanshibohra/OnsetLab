import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { GitBranch, Plug, RefreshCcw, Cpu, ArrowRight, Copy, Check, BookOpen, Layers, Zap, CheckCircle, Star, Package, Play } from 'lucide-react'

const FadeIn = ({ children, delay = 0, className = '' }) => (
  <motion.div
    className={className}
    initial={{ opacity: 0, y: 16 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5, delay }}
  >
    {children}
  </motion.div>
)

const ASCII_ART = `
 ██████╗ ███╗   ██╗███████╗███████╗████████╗██╗      █████╗ ██████╗ 
██╔═══██╗████╗  ██║██╔════╝██╔════╝╚══██╔══╝██║     ██╔══██╗██╔══██╗
██║   ██║██╔██╗ ██║███████╗█████╗     ██║   ██║     ███████║██████╔╝
██║   ██║██║╚██╗██║╚════██║██╔══╝     ██║   ██║     ██╔══██║██╔══██╗
╚██████╔╝██║ ╚████║███████║███████╗   ██║   ███████╗██║  ██║██████╔╝
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═════╝ 
`

const CODE_LINES = [
  { tokens: [{ text: 'from ', cls: '' }, { text: 'onsetlab', cls: 't-str' }, { text: ' import ', cls: '' }, { text: 'Agent', cls: 't-kw' }] },
  { tokens: [] },
  { tokens: [{ text: 'agent ', cls: '' }, { text: '= ', cls: '' }, { text: 'Agent', cls: 't-kw' }, { text: '(', cls: '' }] },
  { tokens: [{ text: '    model', cls: '' }, { text: '=', cls: '' }, { text: '"qwen3-1.7b"', cls: 't-str' }, { text: ',', cls: '' }] },
  { tokens: [{ text: '    tools', cls: '' }, { text: '=', cls: '' }, { text: '[', cls: '' }, { text: '"Calculator"', cls: 't-str' }, { text: ', ', cls: '' }, { text: '"DateTime"', cls: 't-str' }, { text: ']', cls: '' }] },
  { tokens: [{ text: ')', cls: '' }] },
  { tokens: [] },
  { tokens: [{ text: 'agent.', cls: '' }, { text: 'connect_mcp', cls: 't-kw' }, { text: '(', cls: '' }, { text: '"github"', cls: 't-str' }, { text: ', token', cls: '' }, { text: '=', cls: '' }, { text: '"..."', cls: 't-str' }, { text: ')', cls: '' }] },
  { tokens: [{ text: 'result ', cls: '' }, { text: '= ', cls: '' }, { text: 'agent.', cls: '' }, { text: 'run', cls: 't-kw' }, { text: '(', cls: '' }, { text: '"Summarize my PRs"', cls: 't-str' }, { text: ')', cls: '' }] },
]

function CodeSnippet() {
  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-block-dot" />
        <span className="code-block-dot" />
        <span className="code-block-dot" />
        <span className="code-block-lang">python</span>
      </div>
      <pre className="code-block-body">
        {CODE_LINES.map((line, i) => (
          <div key={i} className="code-line">
            <span className="code-lineno">{i + 1}</span>
            {line.tokens.length === 0 ? '\n' : line.tokens.map((t, j) => (
              <span key={j} className={t.cls}>{t.text}</span>
            ))}
          </div>
        ))}
      </pre>
    </div>
  )
}

function PipInstall() {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText('pip install onsetlab')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="pip-install">
      <code><span style={{ opacity: 0.35 }}>$</span> pip install onsetlab</code>
      <button onClick={handleCopy} className="pip-copy-btn" title="Copy to clipboard">
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  )
}

const FLOW_NODES = [
  { 
    id: 'planner', 
    icon: Layers,
    label: 'Plans execution', 
    desc: 'Get Tokyo weather, then calculate hours until 6:30pm JST',
    accent: '#7aa2f7',
    type: 'core' 
  },
  { 
    id: 'tool1', 
    icon: Zap,
    label: 'Weather.get("Tokyo")', 
    accent: 'var(--accent)',
    type: 'tool',
    mono: true 
  },
  { 
    id: 'tool2', 
    icon: Zap,
    label: 'DateTime.hours_until("18:30 JST")', 
    accent: 'var(--accent)',
    type: 'tool',
    mono: true 
  },
  { 
    id: 'fallback', 
    icon: RefreshCcw,
    label: 'Auto-corrects on error', 
    desc: 'Wrong timezone? Bad format? Retries with fixes',
    accent: '#e0af68',
    type: 'fallback' 
  },
  { 
    id: 'answer', 
    icon: CheckCircle,
    label: 'Clear skies, 72°F. Flight in 4 hours.',
    accent: '#9ece6a',
    type: 'end' 
  },
]

function ArchitectureFlow() {
  return (
    <div className="arch-flow">
      <div className="arch-flow-question">
        "What's the weather in Tokyo and how long until my flight at 6:30pm?"
      </div>
      {FLOW_NODES.map((node, i) => (
        <motion.div
          key={node.id}
          initial={{ opacity: 0, x: 16 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.35, delay: 0.12 + i * 0.08 }}
        >
          <div className="arch-connector"><div className="arch-line" /></div>
          <div className={`arch-node ${node.type === 'fallback' ? 'arch-node-fallback' : ''}`} style={{ borderLeftColor: node.accent }}>
            <div className="arch-node-header">
              <node.icon size={14} style={{ color: node.accent }} strokeWidth={2.5} />
              <span className="arch-node-label" style={{ color: node.accent }}>
                {node.type === 'tool' ? 'Tool Call' : node.type === 'core' ? 'REWOO Planner' : node.type === 'fallback' ? 'ReAct Fallback' : 'Answer'}
              </span>
            </div>
            <p className={node.mono ? 'mono' : ''}>
              {node.label}
            </p>
            {node.desc && <p className="arch-node-desc">{node.desc}</p>}
          </div>
        </motion.div>
      ))}
    </div>
  )
}

function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-5" style={{ background: 'rgba(223, 227, 232, 0.9)', backdropFilter: 'blur(20px)' }}>
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        <a href="/" className="font-medium" style={{ color: 'var(--text)' }}>
          onsetlab
        </a>
        <div className="flex items-center gap-6 text-sm">
          <a href="/docs" style={{ color: 'var(--text-secondary)' }} className="hover:opacity-70 transition">Docs</a>
          <a href="/playground" className="btn btn-primary py-2 px-4 text-sm">Playground</a>
          <a href="https://github.com/riyanshibohra/OnsetLab" className="btn btn-secondary py-2 px-4 text-sm" style={{ gap: '5px' }} target="_blank" rel="noopener noreferrer">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}

function Hero() {
  return (
    <section className="hero-section relative overflow-hidden px-6">
      <div className="hero-dot-grid" />

      <div className="max-w-5xl mx-auto w-full relative z-10 pt-32 pb-16">
        {/* Centered identity block */}
        <div className="text-center mb-16">
          <FadeIn>
            <pre
              className="leading-tight mb-6 inline-block"
              style={{ color: 'var(--accent)', fontSize: '8px', opacity: 0.9 }}
              aria-hidden="true"
            >{ASCII_ART}</pre>
          </FadeIn>

          <FadeIn delay={0.1}>
            <h1 className="text-2xl md:text-3xl lg:text-4xl font-normal leading-snug mb-4">
              Tool-calling AI agents
              <br />
              <span style={{ color: 'var(--text-secondary)' }}>that run locally.</span>
            </h1>
          </FadeIn>

          <FadeIn delay={0.15}>
            <p className="text-sm max-w-sm mx-auto" style={{ color: 'var(--text-secondary)', lineHeight: 1.8 }}>
              Build once, run anywhere.
              <br />
              Your models, your tools, your machine.
            </p>
          </FadeIn>
        </div>

        {/* Split: code left, architecture right */}
        <div className="hero-split">
          <div className="hero-split-left">
            <FadeIn delay={0.2}>
              <CodeSnippet />
            </FadeIn>

            <FadeIn delay={0.25}>
              <PipInstall />
            </FadeIn>

            <FadeIn delay={0.3}>
              <div style={{ display: 'flex', gap: '10px', width: '100%' }}>
                <a href="/docs" className="btn btn-secondary" style={{ gap: '6px', flex: 1, justifyContent: 'center' }}>
                  <BookOpen size={15} />
                  Read the Docs
                </a>
                <a href="https://github.com/riyanshibohra/OnsetLab" className="btn btn-primary" style={{ gap: '6px', flex: 1, justifyContent: 'center' }} target="_blank" rel="noopener noreferrer">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                  Star on GitHub
                </a>
              </div>
            </FadeIn>

            <FadeIn delay={0.35}>
              <div style={{ display: 'flex', gap: '10px', width: '100%' }}>
                <a href="https://pypi.org/project/onsetlab/" className="btn btn-secondary" style={{ gap: '6px', flex: 1, justifyContent: 'center' }} target="_blank" rel="noopener noreferrer">
                  <Package size={15} />
                  PyPI
                </a>
                <a href="/playground" className="btn btn-secondary" style={{ gap: '6px', flex: 1, justifyContent: 'center' }}>
                  <Play size={15} />
                  Try Playground
                </a>
              </div>
            </FadeIn>

            <FadeIn delay={0.4}>
              <p className="text-xs mt-3" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
                Qwen, Mistral, Hermes, Gemma + any Ollama model
              </p>
            </FadeIn>
          </div>

          <FadeIn delay={0.2} className="hero-split-right">
            <ArchitectureFlow />
          </FadeIn>
        </div>
      </div>
    </section>
  )
}

function Features() {
  const features = [
    {
      Icon: GitBranch,
      title: 'REWOO + ReAct',
      desc: 'Plans all steps upfront, executes in sequence, and falls back to step-by-step reasoning when a tool call fails.',
      accent: '#7aa2f7',
    },
    {
      Icon: Plug,
      title: 'MCP Servers',
      desc: 'GitHub, Slack, Notion, filesystem, web search - connect any MCP server with a single line of config.',
      accent: '#9ece6a',
    },
    {
      Icon: RefreshCcw,
      title: 'Self-Correcting',
      desc: 'Wrong parameters? The agent retries with corrected inputs. Bad plan? ReAct fallback kicks in automatically.',
      accent: '#e0af68',
    },
    {
      Icon: Cpu,
      title: 'Any Ollama Model',
      desc: 'Qwen, Mistral, Hermes, Gemma - use whatever fits your hardware. No API keys, no cloud dependency.',
      accent: 'var(--accent)',
    },
  ]

  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <FadeIn>
          <h2 className="text-2xl font-medium text-center mb-12" style={{ color: 'var(--text)' }}>
            What makes it different
          </h2>
        </FadeIn>
        <div className="grid md:grid-cols-2 gap-5">
          {features.map((f, i) => (
            <FadeIn key={i} delay={i * 0.08}>
              <div className="feature-card-v2">
                <div className="feature-icon" style={{ background: f.accent }}>
                  <f.Icon size={16} strokeWidth={2} color="white" />
                </div>
                <div>
                  <h3 className="text-sm font-medium mb-1.5">{f.title}</h3>
                  <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{f.desc}</p>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  )
}

function Demo() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-4xl mx-auto">
        <FadeIn>
          <h2 className="text-2xl font-medium text-center mb-4" style={{ color: 'var(--text)' }}>
            See it in action
          </h2>
          <p className="text-sm text-center mb-12" style={{ color: 'var(--text-secondary)', opacity: 0.7 }}>
            Watch how OnsetLab handles real queries from start to finish
          </p>
        </FadeIn>

        <FadeIn delay={0.1}>
          <div className="video-container">
            <iframe
              src="https://www.youtube.com/embed/dQw4w9WgXcQ"
              title="OnsetLab Demo"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </FadeIn>
      </div>
    </section>
  )
}

function GetStarted() {
  return (
    <section id="start" className="py-24 px-6">
      <div className="max-w-5xl mx-auto text-center">
        <FadeIn>
          <h2 className="text-2xl md:text-3xl font-normal mb-4">
            From install to deploy
          </h2>
          <p className="text-sm mb-12" style={{ color: 'var(--text-secondary)' }}>
            Four simple steps to ship your agent.
          </p>
        </FadeIn>

        <FadeIn delay={0.1}>
          <div className="journey-track">
            <div className="journey-step">
              <span className="step-number">1</span>
              <h3 className="text-sm font-medium mb-1">Install</h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>pip install onsetlab</p>
            </div>

            <ArrowRight size={18} className="journey-arrow" style={{ color: 'var(--border)' }} />

            <div className="journey-step">
              <span className="step-number">2</span>
              <h3 className="text-sm font-medium mb-1">Configure</h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Pick a model, add tools &amp; MCP servers</p>
            </div>

            <ArrowRight size={18} className="journey-arrow" style={{ color: 'var(--border)' }} />

            <div className="journey-step">
              <span className="step-number">3</span>
              <h3 className="text-sm font-medium mb-1">Run</h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>agent.run("your task")</p>
            </div>

            <ArrowRight size={18} className="journey-arrow" style={{ color: 'var(--border)' }} />

            <div className="journey-step">
              <span className="step-number">4</span>
              <h3 className="text-sm font-medium mb-1">Deploy</h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Export as Docker, YAML, or script</p>
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={0.25}>
          <div className="mt-16">
            <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
              Ready to build?
            </p>
            <a href="/docs" className="btn btn-primary" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
              Get Started
              <ArrowRight size={16} />
            </a>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="py-12 px-6 border-t" style={{ borderColor: 'var(--border)' }}>
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          onsetlab · Apache 2.0
        </div>
        <div className="flex items-center gap-6 text-sm" style={{ color: 'var(--text-secondary)' }}>
          <a href="/playground" className="hover:opacity-70 transition">Playground</a>
          <a href="https://github.com/riyanshibohra/OnsetLab" className="hover:opacity-70 transition" target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href="https://pypi.org/project/onsetlab/" className="hover:opacity-70 transition" target="_blank" rel="noopener noreferrer">PyPI</a>
        </div>
      </div>
    </footer>
  )
}

function App() {
  useEffect(() => {
    document.title = 'OnsetLab | Tool-calling AI agents that run locally'
  }, [])

  return (
    <div>
      <Nav />
      <Hero />
      <Features />
      <div style={{ height: '1px', background: 'var(--border)', maxWidth: '72rem', margin: '0 auto' }} />
      <Demo />
      <div style={{ height: '1px', background: 'var(--border)', maxWidth: '72rem', margin: '0 auto' }} />
      <GetStarted />
      <Footer />
    </div>
  )
}

export default App
