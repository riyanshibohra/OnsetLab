import { motion } from 'framer-motion'
import { GitBranch, Plug, RefreshCcw, Cpu, ArrowRight } from 'lucide-react'

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

const PIPELINE_STEPS = [
  {
    label: 'Plan',
    accent: '#7aa2f7',
    content: 'Sum all invoice totals and hours, compute average hourly rate, then find days remaining in the quarter.',
  },
  {
    label: 'Tool Call',
    accent: 'var(--accent)',
    content: 'Calculator.divide(5235, 77)',
    mono: true,
  },
  {
    label: 'Tool Call',
    accent: 'var(--accent)',
    content: 'DateTime.days_until("2024-03-31")',
    mono: true,
  },
  {
    label: 'Result',
    accent: '#9ece6a',
    content: 'Your average rate is $67.99/hr across 77 hours. 74 days left in Q1.',
  },
]

function PipelineDemo() {
  return (
    <div className="pipeline-demo">
      <div className="pipeline-header">
        <span className="pipeline-prompt">"I have 3 invoices: $2,400 for 40hrs, $1,875 for 25hrs, $960 for 12hrs. What's my average hourly rate and how many days until end of quarter?"</span>
      </div>
      {PIPELINE_STEPS.map((step, i) => (
        <motion.div
          key={`${step.label}-${i}`}
          className="pipeline-step"
          style={{ borderLeftColor: step.accent }}
          initial={{ opacity: 0, x: 12 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4, delay: 0.2 + i * 0.15 }}
        >
          <span className="pipeline-label" style={{ color: step.accent }}>{step.label}</span>
          <p className={step.mono ? 'mono' : ''} style={{ color: step.mono ? 'var(--text)' : 'var(--text-secondary)' }}>
            {step.content}
          </p>
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
          <a href="#features" style={{ color: 'var(--text-secondary)' }} className="hover:opacity-70 transition">Features</a>
          <a href="#start" style={{ color: 'var(--text-secondary)' }} className="hover:opacity-70 transition">Get Started</a>
          <a href="/playground" className="btn btn-primary py-2 px-4 text-sm">Playground</a>
          <a href="https://github.com/riyanshibohra/OnsetLab" className="btn btn-secondary py-2 px-4 text-sm" target="_blank" rel="noopener noreferrer">GitHub</a>
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
              style={{ color: 'var(--accent)', fontSize: '8px', opacity: 0.35 }}
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

        {/* Split: CTA left, pipeline right */}
        <div className="hero-split">
          <div className="hero-split-left">
            <FadeIn delay={0.2}>
              <div className="hero-actions-col">
                <a href="/playground" className="btn btn-primary">Try in Browser</a>
                <a href="https://github.com/riyanshibohra/OnsetLab" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">GitHub</a>
                <div className="code-inline">
                  <span style={{ opacity: 0.35 }}>$</span> pip install onsetlab
                </div>
              </div>
            </FadeIn>

            <FadeIn delay={0.3}>
              <p className="text-xs mt-8" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
                Works with Qwen, Mistral, Hermes, Gemma
                <br />
                and any Ollama-compatible model.
              </p>
            </FadeIn>
          </div>

          <FadeIn delay={0.2} className="hero-split-right">
            <PipelineDemo />
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
      desc: 'GitHub, Slack, Notion, filesystem, web search — connect any MCP server with a single line of config.',
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
      desc: 'Qwen, Mistral, Hermes, Gemma — use whatever fits your hardware. No API keys, no cloud dependency.',
      accent: 'var(--accent)',
    },
  ]

  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
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

function GetStarted() {
  return (
    <section id="start" className="py-24 px-6">
      <div className="max-w-5xl mx-auto text-center">
        <FadeIn>
          <h2 className="text-2xl md:text-3xl font-normal mb-4">
            From install to deploy
          </h2>
          <p className="text-sm mb-12" style={{ color: 'var(--text-secondary)' }}>
            Build your agent, connect your tools, ship it anywhere.
          </p>
        </FadeIn>

        <FadeIn delay={0.1}>
          <div className="journey-track">
            <div className="journey-step">
              <span className="step-number">1</span>
              <h3 className="text-sm font-medium mb-1">Install</h3>
              <code className="text-xs" style={{ color: 'var(--accent)' }}>pip install onsetlab</code>
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
              <code className="text-xs" style={{ color: 'var(--accent)' }}>agent.run("your task")</code>
            </div>

            <ArrowRight size={18} className="journey-arrow" style={{ color: 'var(--border)' }} />

            <div className="journey-step">
              <span className="step-number">4</span>
              <h3 className="text-sm font-medium mb-1">Deploy</h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Export as Docker, YAML, or script</p>
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={0.2}>
          <div className="flex gap-3 justify-center mt-14">
            <a href="/playground" className="btn btn-primary">Try the Playground</a>
            <a href="https://github.com/riyanshibohra/OnsetLab#readme" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">Documentation</a>
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
  return (
    <div>
      <Nav />
      <Hero />
      <Features />
      <div style={{ height: '1px', background: 'var(--border)', maxWidth: '72rem', margin: '0 auto' }} />
      <GetStarted />
      <Footer />
    </div>
  )
}

export default App
