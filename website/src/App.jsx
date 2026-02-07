import { motion } from 'framer-motion'

const FadeIn = ({ children, delay = 0 }) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5, delay }}
  >
    {children}
  </motion.div>
)

// Navigation
function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-5" style={{ background: 'rgba(228, 224, 217, 0.9)', backdropFilter: 'blur(20px)' }}>
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <a href="#" className="font-medium">
          onsetlab
        </a>
        
        <div className="flex items-center gap-8 text-sm">
          <a href="#features" style={{ color: 'var(--text-secondary)' }} className="hover:opacity-70 transition">Features</a>
          <a href="#start" style={{ color: 'var(--text-secondary)' }} className="hover:opacity-70 transition">Get Started</a>
          <a href="/playground" className="btn btn-primary py-2 px-4 text-sm">
            Try it
          </a>
          <a 
            href="https://github.com/riyanshibohra/OnsetLab" 
            className="btn btn-secondary py-2 px-4 text-sm"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}

// Hero
function Hero() {
  return (
    <section className="min-h-screen flex flex-col justify-center px-6 pt-20">
      <div className="max-w-4xl mx-auto w-full">
        <FadeIn>
          <pre 
            className="text-[7px] sm:text-[9px] md:text-[11px] leading-tight mb-12 overflow-x-auto"
            style={{ color: 'var(--accent)' }}
          >
{`
 ██████╗ ███╗   ██╗███████╗███████╗████████╗██╗      █████╗ ██████╗ 
██╔═══██╗████╗  ██║██╔════╝██╔════╝╚══██╔══╝██║     ██╔══██╗██╔══██╗
██║   ██║██╔██╗ ██║███████╗█████╗     ██║   ██║     ███████║██████╔╝
██║   ██║██║╚██╗██║╚════██║██╔══╝     ██║   ██║     ██╔══██║██╔══██╗
╚██████╔╝██║ ╚████║███████║███████╗   ██║   ███████╗██║  ██║██████╔╝
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═════╝ 
`}
          </pre>
        </FadeIn>
        
        <FadeIn delay={0.1}>
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-normal leading-snug mb-6">
            Build AI agents that run
            <br />
            <span style={{ color: 'var(--text-secondary)' }}>entirely on your machine.</span>
          </h1>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <p className="text-lg mb-10 max-w-lg" style={{ color: 'var(--text-secondary)' }}>
            No cloud, no API keys, no data leaving your computer. 
            Local LLMs with 50+ tools, ready in under a minute.
          </p>
        </FadeIn>
        
        <FadeIn delay={0.3}>
          <div className="flex flex-wrap gap-3">
            <a href="/playground" className="btn btn-primary">
              Try in Browser
            </a>
            <a href="#start" className="btn btn-secondary">
              Install SDK
            </a>
            <a href="https://github.com/riyanshibohra/OnsetLab" className="btn btn-secondary">
              GitHub
            </a>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}

// Features
function Features() {
  const features = [
    {
      title: 'Private by Default',
      desc: 'Everything runs locally. Your data, your models, your hardware. Nothing ever phones home.',
    },
    {
      title: '50+ MCP Tools',
      desc: 'GitHub, Slack, Notion, filesystem, databases — all work out of the box. Add your own easily.',
    },
    {
      title: 'Self-Correcting',
      desc: 'REWOO planning with ReAct fallback. When something fails, it automatically tries another approach.',
    },
    {
      title: 'Any Local Model',
      desc: 'Works with Ollama. Phi-3, Llama, Mistral, Qwen — use whatever fits your hardware.',
    },
  ]

  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-4xl mx-auto">
        <FadeIn>
          <p className="text-xs font-medium tracking-widest mb-12" style={{ color: 'var(--text-secondary)' }}>
            WHY ONSETLAB
          </p>
        </FadeIn>
        
        <div className="grid md:grid-cols-2 gap-6">
          {features.map((f, i) => (
            <FadeIn key={i} delay={i * 0.1}>
              <div className="card">
                <h3 className="text-lg font-medium mb-2">{f.title}</h3>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{f.desc}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  )
}

// Code Example
function CodeExample() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-4xl mx-auto">
        <FadeIn>
          <p className="text-xs font-medium tracking-widest mb-8" style={{ color: 'var(--text-secondary)' }}>
            SIMPLE API
          </p>
        </FadeIn>
        
        <FadeIn delay={0.1}>
          <div className="code-block">
            <pre className="text-sm leading-loose">
              <code>
                <span style={{ color: 'var(--text-secondary)' }}>from</span> onsetlab <span style={{ color: 'var(--text-secondary)' }}>import</span> Agent, MCPServer{'\n'}
{'\n'}
<span style={{ color: 'var(--text-secondary)' }}># Create agent with local model</span>{'\n'}
agent = Agent(<span style={{ color: 'var(--accent)' }}>"phi3.5"</span>){'\n'}
{'\n'}
<span style={{ color: 'var(--text-secondary)' }}># Add tools (optional)</span>{'\n'}
agent.add_mcp_server(MCPServer.from_registry(<span style={{ color: 'var(--accent)' }}>"github"</span>)){'\n'}
{'\n'}
<span style={{ color: 'var(--text-secondary)' }}># Run tasks in natural language</span>{'\n'}
result = agent.run(<span style={{ color: 'var(--accent)' }}>"Create an issue for the login bug"</span>){'\n'}
print(result.answer)
              </code>
            </pre>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}

// Get Started
function GetStarted() {
  return (
    <section id="start" className="py-24 px-6">
      <div className="max-w-4xl mx-auto">
        <FadeIn>
          <p className="text-xs font-medium tracking-widest mb-4" style={{ color: 'var(--text-secondary)' }}>
            GET STARTED
          </p>
        </FadeIn>
        
        <FadeIn delay={0.1}>
          <h2 className="text-2xl md:text-3xl font-normal mb-8">
            One command to install.
          </h2>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <div className="code-block inline-block mb-10">
            <code className="text-base">
              <span style={{ color: 'var(--text-secondary)' }}>$</span> pip install onsetlab
            </code>
          </div>
        </FadeIn>
        
        <FadeIn delay={0.3}>
          <p className="mb-8 max-w-md" style={{ color: 'var(--text-secondary)' }}>
            Requires Python 3.9+ and Ollama. No accounts, no API tokens, 
            no cloud services. Just install and start building.
          </p>
        </FadeIn>
        
        <FadeIn delay={0.4}>
          <div className="flex flex-wrap gap-3">
            <a 
              href="https://github.com/riyanshibohra/OnsetLab#readme" 
              className="btn btn-primary"
            >
              Read Documentation
            </a>
            <a 
              href="https://github.com/riyanshibohra/OnsetLab" 
              className="btn btn-secondary"
            >
              Star on GitHub
            </a>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}

// Footer
function Footer() {
  return (
    <footer className="py-16 px-6 border-t" style={{ borderColor: 'var(--border)' }}>
      <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <div className="font-medium mb-1">onsetlab</div>
          <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Open source · Apache 2.0 License
          </div>
        </div>
        
        <div className="flex items-center gap-6 text-sm" style={{ color: 'var(--text-secondary)' }}>
          <a href="/playground" className="hover:opacity-70 transition">
            Playground
          </a>
          <a href="https://github.com/riyanshibohra/OnsetLab" className="hover:opacity-70 transition">
            GitHub
          </a>
          <a href="https://github.com/riyanshibohra/OnsetLab#readme" className="hover:opacity-70 transition">
            Docs
          </a>
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
      <div className="divider max-w-4xl mx-auto" />
      <Features />
      <CodeExample />
      <div className="divider max-w-4xl mx-auto" />
      <GetStarted />
      <Footer />
    </div>
  )
}

export default App
