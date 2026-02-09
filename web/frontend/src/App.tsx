import { useState, useRef, useEffect } from 'react';
import { api, ApiError } from './api/client';
import type { Message, ToolInfo, MCPServerInfo } from './types';

// Default data (fallback if API fails)
const DEFAULT_TOOLS: ToolInfo[] = [
  { name: 'Calculator', description: 'Math expressions', enabled_by_default: true, category: 'builtin' },
  { name: 'DateTime', description: 'Date/time queries', enabled_by_default: true, category: 'builtin' },
  { name: 'UnitConverter', description: 'Unit conversion', enabled_by_default: true, category: 'builtin' },
  { name: 'TextProcessor', description: 'Text operations', enabled_by_default: true, category: 'builtin' },
  { name: 'RandomGenerator', description: 'Random values', enabled_by_default: true, category: 'builtin' },
];


const DEFAULT_MCP_SERVERS: MCPServerInfo[] = [
  { name: 'GitHub', description: 'Issues, PRs, repos', registry_key: 'github', requires_token: true, token_label: 'Personal Access Token', setup_url: 'https://github.com/settings/tokens', token_hint: 'ghp_... with repo scope' },
  { name: 'Slack', description: 'Messages, channels', registry_key: 'slack', requires_token: true, token_label: 'User OAuth Token', setup_url: 'https://api.slack.com/apps', token_hint: 'xoxp-... from Slack app' },
  { name: 'Notion', description: 'Pages, databases', registry_key: 'notion', requires_token: true, token_label: 'Integration Secret', setup_url: 'https://www.notion.so/profile/integrations', token_hint: 'secret_... from integration' },
  { name: 'Tavily', description: 'Web search', registry_key: 'tavily', requires_token: true, token_label: 'API Key', setup_url: 'https://app.tavily.com/', token_hint: 'tvly-... from dashboard' },
];

// MCP connection state per server
interface MCPConnectionState {
  connected: boolean;
  connecting: boolean;
  tools: string[];
  error?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestsRemaining, setRequestsRemaining] = useState(5);
  const [isRateLimited, setIsRateLimited] = useState(false);
  
  const [tools] = useState<ToolInfo[]>(DEFAULT_TOOLS);
  const [mcpServers] = useState<MCPServerInfo[]>(DEFAULT_MCP_SERVERS);
  const [selectedTools, setSelectedTools] = useState<string[]>(
    DEFAULT_TOOLS.filter(t => t.enabled_by_default).map(t => t.name)
  );
  const selectedModel = 'qwen3-a3b';

  // Sidebar collapsible sections
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    tools: true,
    mcp: true,
    export: true,
  });

  const toggleSidebar = (key: string) => {
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Pipeline trace expansion state (inline in messages)
  const [expandedTraces, setExpandedTraces] = useState<Set<string>>(new Set());
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  // Pipeline panel state (right side drawer, latest query)
  const [pipelineTrace, setPipelineTrace] = useState<any>(null);
  const [pipelinePlan, setPipelinePlan] = useState<any[]>([]);
  const [pipelineResults, setPipelineResults] = useState<Record<string, string>>({});
  const [pipelineStrategy, setPipelineStrategy] = useState('');
  const [pipelineAnswer, setPipelineAnswer] = useState('');
  const [pipelineOpen, setPipelineOpen] = useState(false);

  // MCP state
  const [mcpConnections, setMcpConnections] = useState<Record<string, MCPConnectionState>>({});
  const [tokenModal, setTokenModal] = useState<MCPServerInfo | null>(null);
  const [tokenInput, setTokenInput] = useState('');
  const [tokenError, setTokenError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 50);
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading || isRateLimited) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    // Reset pipeline panel for new query
    setPipelineTrace(null);
    setPipelinePlan([]);
    setPipelineResults({});
    setPipelineStrategy('');
    setPipelineAnswer('');

    try {
      const response = await api.chat(input.trim(), selectedTools, selectedModel);

      // Update pipeline drawer
      setPipelineTrace(response.trace || null);
      setPipelinePlan(response.plan);
      setPipelineResults(response.results || {});
      setPipelineStrategy(response.strategy);
      setPipelineAnswer(response.answer);
      setPipelineOpen(true);
      
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.answer,
        plan: response.plan,
        strategy: response.strategy,
        trace: response.trace,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      setRequestsRemaining(response.requests_remaining);
      
      if (response.requests_remaining <= 0) {
        setIsRateLimited(true);
      }
    } catch (err) {
      if (err instanceof ApiError && err.isRateLimited) {
        setIsRateLimited(true);
        setRequestsRemaining(0);
      } else {
        setError(err instanceof Error ? err.message : 'Connection failed');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = async (format: 'config' | 'docker' | 'docker-vllm' | 'binary') => {
    try {
      const blob = await api.exportAgent(format, selectedTools, selectedModel);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = format === 'docker-vllm' ? 'onsetlab_vllm_docker.zip' :
                   format === 'docker' ? 'onsetlab_docker.zip' : 
                   format === 'config' ? 'agent.yaml' : 'agent.py';
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      console.error('Export failed');
    }
  };

  const toggleTool = (name: string) => {
    setSelectedTools(prev => 
      prev.includes(name) ? prev.filter(t => t !== name) : [...prev, name]
    );
  };

  // Pipeline trace handlers
  const toggleTrace = (msgId: string) => {
    setExpandedTraces(prev => {
      const next = new Set(prev);
      if (next.has(msgId)) next.delete(msgId);
      else next.add(msgId);
      return next;
    });
  };

  const toggleSection = (key: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // MCP handlers
  const openTokenModal = (server: MCPServerInfo) => {
    setTokenModal(server);
    setTokenInput('');
    setTokenError(null);
  };

  const handleMCPConnect = async () => {
    if (!tokenModal || !tokenInput.trim()) return;

    const key = tokenModal.registry_key;

    setMcpConnections(prev => ({
      ...prev,
      [key]: { connected: false, connecting: true, tools: [] },
    }));
    setTokenError(null);

    try {
      const result = await api.connectMCP(key, tokenInput.trim());
      setMcpConnections(prev => ({
        ...prev,
        [key]: { connected: true, connecting: false, tools: result.tools },
      }));
      setTokenModal(null);
      setTokenInput('');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Connection failed';
      setMcpConnections(prev => ({
        ...prev,
        [key]: { connected: false, connecting: false, tools: [], error: msg },
      }));
      setTokenError(msg);
    }
  };

  const handleMCPDisconnect = async (registryKey: string) => {
    try {
      await api.disconnectMCP(registryKey);
    } catch {
      // Ignore disconnect errors
    }
    setMcpConnections(prev => {
      const next = { ...prev };
      delete next[registryKey];
      return next;
    });
  };

  const connectedMCPCount = Object.values(mcpConnections).filter(c => c.connected).length;

  // Render inline markdown (bold, italic, code)
  const renderMarkdown = (text: string) => {
    const parts: React.ReactNode[] = [];
    // Split by **bold**, *italic*, `code`
    const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
    let lastIndex = 0;
    let match;
    let key = 0;

    while ((match = regex.exec(text)) !== null) {
      // Text before match
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      if (match[2]) {
        // **bold**
        parts.push(<strong key={key++}>{match[2]}</strong>);
      } else if (match[3]) {
        // *italic*
        parts.push(<em key={key++}>{match[3]}</em>);
      } else if (match[4]) {
        // `code`
        parts.push(
          <code key={key++} className="mono text-xs px-1 py-0.5 rounded" style={{ background: 'rgba(0,0,0,0.06)' }}>
            {match[4]}
          </code>
        );
      }
      lastIndex = match.index + match[0].length;
    }
    // Remaining text
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    return parts.length > 0 ? parts : text;
  };

  const SUGGESTION_CARDS = [
    { icon: '🧾', label: 'Tip calculator', query: "I'm splitting a $284.50 dinner bill between 4 people with 20% tip. What's each person's share?" },
    { icon: '📅', label: 'Date math', query: "How many days are between March 15 and December 25 this year?" },
    { icon: '⚖️', label: 'Unit conversion', query: "A recipe needs 2.5 cups of flour. I only have a 1/3 cup measure. How many scoops?" },
    { icon: '🔑', label: 'Password gen', query: "Generate a secure 20-character password with uppercase, lowercase, numbers and symbols" },
    { icon: '🕐', label: 'Time zones', query: "What time is it right now, and how many hours until midnight?" },
  ];

  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav 
        className="fixed top-0 left-0 right-0 z-50 px-6 py-4"
        style={{ background: 'rgba(223, 227, 232, 0.9)', backdropFilter: 'blur(20px)' }}
      >
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <span className="font-medium" style={{ color: 'var(--text)' }}>onsetlab</span>
            <span className="tag">playground</span>
          </a>
          
          <div className="flex items-center gap-6 text-sm">
            <span style={{ color: 'var(--text-secondary)' }}>
              {requestsRemaining} requests left
            </span>
            <a 
              href="https://github.com/riyanshibohra/OnsetLab" 
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary btn-sm"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="px-6" style={{ position: 'fixed', top: '56px', bottom: 0, left: 0, right: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div className="max-w-5xl mx-auto w-full flex-1 flex flex-col" style={{ minHeight: 0, paddingTop: '12px', paddingBottom: '12px' }}>
          {/* Layout */}
          <div className="flex gap-6 flex-1" style={{ minHeight: 0 }}>
            {/* Sidebar */}
            <aside className="w-64 shrink-0 flex flex-col" style={{ minHeight: 0 }}>
              <div className="card flex-1" style={{ overflowY: 'auto', padding: '16px 18px' }}>
                {/* Tools */}
                <div className="mb-3">
                  <button onClick={() => toggleSidebar('tools')} className="section-label flex items-center justify-between w-full" style={{ cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}>
                    <span>Built-in Tools</span>
                    <span style={{ fontSize: '10px' }}>{openSections.tools ? '▾' : '▸'}</span>
                  </button>
                  {openSections.tools && (
                    <>
                      <div className="space-y-0.5">
                        {tools.map(tool => (
                          <label
                            key={tool.name}
                            className="flex items-center gap-2 py-1.5 px-2 rounded cursor-pointer transition-colors"
                            style={{ 
                              background: selectedTools.includes(tool.name) ? 'rgba(74, 102, 112, 0.08)' : 'transparent'
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={selectedTools.includes(tool.name)}
                              onChange={() => toggleTool(tool.name)}
                            />
                            <span className="text-sm" style={{ color: 'var(--text)' }}>
                              {tool.name}
                            </span>
                          </label>
                        ))}
                      </div>
                      <p className="text-[10px] mt-2 px-2" style={{ color: 'var(--text-secondary)', opacity: 0.7 }}>
                        More tools coming soon
                      </p>
                    </>
                  )}
                </div>

                {/* MCP Servers */}
                <div className="mb-3">
                  <button onClick={() => toggleSidebar('mcp')} className="section-label flex items-center justify-between w-full" style={{ cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}>
                    <span className="flex items-center gap-2">
                      MCP Servers
                      {connectedMCPCount > 0 && (
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'rgba(74, 124, 89, 0.15)', color: 'var(--success)' }}>
                          {connectedMCPCount} live
                        </span>
                      )}
                    </span>
                    <span style={{ fontSize: '10px' }}>{openSections.mcp ? '▾' : '▸'}</span>
                  </button>
                  {openSections.mcp && <div className="space-y-0.5">
                    {mcpServers.map(server => {
                      const conn = mcpConnections[server.registry_key];
                      const isConnected = conn?.connected;
                      const isConnecting = conn?.connecting;

                      return (
                        <div
                          key={server.registry_key}
                          className="flex items-center justify-between py-1.5 px-2 rounded"
                          style={{ background: isConnected ? 'rgba(74, 124, 89, 0.08)' : 'transparent' }}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <div
                              className="w-1.5 h-1.5 rounded-full shrink-0"
                              style={{ background: isConnected ? 'var(--success)' : 'var(--border)' }}
                            />
                            <div className="min-w-0">
                              <div className="text-sm truncate" style={{ color: 'var(--text)' }}>
                                {server.name}
                              </div>
                              {isConnected && conn.tools.length > 0 && (
                                <div className="text-[10px] font-mono truncate" style={{ color: 'var(--text-secondary)' }}>
                                  {conn.tools.length} tools
                                </div>
                              )}
                            </div>
                          </div>
                          {isConnected ? (
                            <button
                              onClick={() => handleMCPDisconnect(server.registry_key)}
                              className="text-[10px] px-2 py-0.5 rounded shrink-0"
                              style={{ color: 'var(--error)', border: '1px solid var(--error)', background: 'transparent' }}
                            >
                              ×
                            </button>
                          ) : (
                            <button
                              onClick={() => openTokenModal(server)}
                              disabled={isConnecting}
                              className="text-[10px] px-2 py-0.5 rounded shrink-0 sidebar-btn"
                              style={{ color: 'var(--accent)', border: '1px solid var(--accent)', background: 'transparent', opacity: isConnecting ? 0.5 : 1, cursor: 'pointer' }}
                            >
                              {isConnecting ? '...' : 'Connect'}
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>}
                </div>

                <div style={{ height: '1px', background: 'var(--border)', margin: '12px 0' }} />

                {/* Model */}
                <div className="mb-3">
                  <div className="section-label" style={{ padding: 0 }}>Model</div>
                  <div className="flex items-center gap-2 px-2 py-1.5 rounded text-sm" style={{ background: 'rgba(74, 102, 112, 0.08)' }}>
                    <span style={{ color: 'var(--text)' }} className="font-medium">Qwen3 A3B</span>
                    <span className="opacity-50 text-xs">3B</span>
                    <span className="ml-auto text-[9px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'rgba(74, 124, 89, 0.2)', color: 'var(--success)' }}>active</span>
                  </div>
                  <div className="text-[10px] mt-1 px-2 opacity-50">MoE, best for tool calling. Free via OpenRouter</div>
                </div>

                {/* Export */}
                <div>
                  <button onClick={() => toggleSidebar('export')} className="section-label flex items-center justify-between w-full" style={{ cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}>
                    <span>Export</span>
                    <span style={{ fontSize: '10px' }}>{openSections.export ? '▾' : '▸'}</span>
                  </button>
                  {openSections.export && <><div className="flex flex-wrap gap-1">
                    <button
                      onClick={() => handleExport('config')}
                      className="text-xs py-1.5 px-3 rounded sidebar-btn"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)', cursor: 'pointer' }}
                    >
                      YAML
                    </button>
                    <button
                      onClick={() => handleExport('docker')}
                      className="text-xs py-1.5 px-3 rounded sidebar-btn"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)', cursor: 'pointer' }}
                    >
                      Docker
                    </button>
                    <button
                      onClick={() => handleExport('docker-vllm')}
                      className="text-xs py-1.5 px-3 rounded sidebar-btn"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)', cursor: 'pointer' }}
                      title="GPU-accelerated (5-10x faster)"
                    >
                      vLLM
                    </button>
                    <button
                      onClick={() => handleExport('binary')}
                      className="text-xs py-1.5 px-3 rounded sidebar-btn"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)', cursor: 'pointer' }}
                    >
                      Script
                    </button>
                  </div>
                  <div className="text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                    vLLM = GPU, 5-10x faster
                  </div></>}
                </div>
              </div>
            </aside>

            {/* Chat Area */}
            <div className="flex-1 flex flex-col" style={{ minHeight: 0 }}>
              <div className="card flex-1 flex flex-col overflow-hidden" style={{ padding: 0 }}>
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6">
                  {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center py-12">
                      <pre
                        className="leading-tight mb-4"
                        style={{ color: 'var(--accent)', fontSize: '6px', opacity: 0.9 }}
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
                      <p className="text-base font-medium mb-1" style={{ color: 'var(--text)' }}>
                        What do you want to solve?
                      </p>
                      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        Pick a suggestion below or type your own question.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {messages.map(msg => (
                        <div
                          key={msg.id}
                          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div className={`message ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                            {/* Strategy badge + Pipeline toggle */}
                            {msg.role === 'assistant' && msg.strategy && (
                              <div className="flex items-center gap-2 mb-2 flex-wrap">
                                <span className="tag tag-accent">{msg.strategy?.toUpperCase()}</span>
                                {msg.plan && msg.plan.length > 0 && (
                                  <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                                    {msg.plan.length} step{msg.plan.length > 1 ? 's' : ''}
                                  </span>
                                )}
                                {msg.trace && (
                                  <button
                                    onClick={() => toggleTrace(msg.id)}
                                    className="text-[10px] font-mono px-1.5 py-0.5 rounded cursor-pointer transition-colors"
                                    style={{
                                      background: expandedTraces.has(msg.id) ? 'rgba(74, 124, 89, 0.18)' : 'rgba(255,255,255,0.04)',
                                      color: expandedTraces.has(msg.id) ? 'var(--success)' : 'var(--text-secondary)',
                                      border: '1px solid var(--border)',
                                    }}
                                  >
                                    {expandedTraces.has(msg.id) ? '▾' : '▸'} Pipeline
                                  </button>
                                )}
                                {msg.trace?.fallback_used && (
                                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                                    style={{ background: 'rgba(200, 150, 50, 0.12)', color: '#c89632' }}>
                                    fallback
                                  </span>
                                )}
                              </div>
                            )}

                            {/* Pipeline Trace Panel */}
                            {msg.trace && expandedTraces.has(msg.id) && (
                              <div
                                className="mb-3 rounded-lg text-xs font-mono overflow-hidden"
                                style={{ background: 'rgba(0,0,0,0.15)', border: '1px solid var(--border)' }}
                              >
                                {/* Router */}
                                <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                                  <span style={{ color: 'var(--text-secondary)' }}>Router: </span>
                                  <span style={{ color: msg.trace.router_decision === 'REWOO' ? 'var(--accent)' : 'var(--success)' }}>
                                    {msg.trace.router_decision}
                                  </span>
                                  <span style={{ color: 'var(--text-secondary)' }}> - {msg.trace.router_reason}</span>
                                </div>

                                {/* Tools */}
                                {msg.trace.tools_filtered > 0 && (
                                  <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                                    <span style={{ color: 'var(--text-secondary)' }}>Tools: </span>
                                    <span style={{ color: 'var(--accent)' }}>
                                      {msg.trace.tools_filtered} of {msg.trace.tools_total} selected
                                    </span>
                                    <span style={{ color: 'var(--text-secondary)' }}>
                                      {' - '}{msg.trace.tools_selected.slice(0, 5).join(', ')}
                                      {msg.trace.tools_selected.length > 5 ? ` +${msg.trace.tools_selected.length - 5}` : ''}
                                    </span>
                                  </div>
                                )}

                                {/* Reasoning (THINK) */}
                                {msg.trace.planner_think && (
                                  <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                                    <span style={{ color: 'var(--text-secondary)' }}>Reasoning: </span>
                                    <span style={{ color: 'var(--text-primary)' }}>{msg.trace.planner_think}</span>
                                  </div>
                                )}

                                {/* Tool Rules (collapsible) */}
                                {msg.trace.tool_rules && (
                                  <div style={{ borderBottom: '1px solid var(--border)' }}>
                                    <button
                                      onClick={() => toggleSection(`rules-${msg.id}`)}
                                      className="w-full px-3 py-2 text-left cursor-pointer"
                                      style={{ color: 'var(--text-secondary)' }}
                                    >
                                      {expandedSections.has(`rules-${msg.id}`) ? '▾' : '▸'} Tool Rules ({msg.trace.tool_rules.split('\n').length} rules)
                                    </button>
                                    {expandedSections.has(`rules-${msg.id}`) && (
                                      <pre className="px-3 pb-2 whitespace-pre-wrap break-all" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                                        {msg.trace.tool_rules}
                                      </pre>
                                    )}
                                  </div>
                                )}

                                {/* Execution Plan */}
                                {msg.plan && msg.plan.length > 0 && (
                                  <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                                    <span style={{ color: 'var(--text-secondary)' }}>Execution: </span>
                                    {msg.plan.map(step => (
                                      <span key={step.id}>
                                        <span style={{ color: 'var(--text-secondary)' }}>{step.id} </span>
                                        <span style={{ color: 'var(--accent)' }}>{step.tool}</span>
                                        <span style={{ color: step.status === 'error' ? 'var(--error)' : 'var(--success)' }}>
                                          {' '}→ {step.status === 'error' ? 'Error' : 'OK'}
                                          {step.result ? ` (${step.result.length} chars)` : ''}
                                        </span>
                                        {'  '}
                                      </span>
                                    ))}
                                  </div>
                                )}

                                {/* Fallback */}
                                {msg.trace.fallback_used && (
                                  <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                                    <span style={{ color: '#c89632' }}>Fallback: </span>
                                    <span style={{ color: 'var(--text-secondary)' }}>{msg.trace.fallback_reason}</span>
                                  </div>
                                )}

                                {/* Solver */}
                                {msg.strategy && msg.strategy !== 'direct' && (
                                  <div className="px-3 py-2">
                                    <span style={{ color: 'var(--text-secondary)' }}>Solver: </span>
                                    <span style={{ color: 'var(--success)' }}>Synthesized from {msg.plan?.length || 0} result{(msg.plan?.length || 0) !== 1 ? 's' : ''}</span>
                                  </div>
                                )}
                              </div>
                            )}

                            <p className="text-sm leading-relaxed whitespace-pre-line">
                              {renderMarkdown(msg.content)}
                            </p>
                            
                          </div>
                        </div>
                      ))}
                      
                      {/* Loading */}
                      {isLoading && (
                        <div className="flex justify-start">
                          <div 
                            className="message message-assistant flex items-center gap-2"
                            style={{ padding: '16px 20px' }}
                          >
                            <div className="loading-dot" style={{ animationDelay: '0ms' }} />
                            <div className="loading-dot" style={{ animationDelay: '150ms' }} />
                            <div className="loading-dot" style={{ animationDelay: '300ms' }} />
                          </div>
                        </div>
                      )}
                      
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </div>

                {/* Error */}
                {error && (
                  <div 
                    className="mx-6 mb-4 p-4 rounded-lg text-sm"
                    style={{ 
                      background: 'rgba(139, 74, 74, 0.1)',
                      border: '1px solid var(--error)',
                      color: 'var(--error)'
                    }}
                  >
                    {error}
                  </div>
                )}

                {/* Suggestion pills */}
                <div style={{ padding: '8px 24px 0', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {SUGGESTION_CARDS.map((card) => (
                    <button
                      key={card.label}
                      className="suggestion-pill"
                      onClick={() => { setInput(card.query); }}
                    >
                      <span style={{ fontSize: '12px' }}>{card.icon}</span> {card.label}
                    </button>
                  ))}
                </div>

                {/* Input */}
                <div style={{ padding: '12px 24px 16px', borderTop: 'none' }}>
                  <form
                    onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                    className="flex gap-3"
                  >
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder={isRateLimited ? "Limit reached - download the SDK" : "Type your question..."}
                      disabled={isRateLimited || isLoading}
                      style={{ flex: 1 }}
                    />
                    <button
                      type="submit"
                      disabled={isRateLimited || isLoading || !input.trim()}
                      className="btn btn-primary"
                    >
                      Send
                    </button>
                  </form>
                </div>
              </div>
            </div>

          </div>
        </div>
      </main>

      {/* Pipeline Toggle Button */}
      {pipelineTrace && !pipelineOpen && (
        <button className="pipeline-toggle-btn" onClick={() => setPipelineOpen(true)}>
          ⚡ Pipeline
        </button>
      )}

      {/* Pipeline Drawer */}
      {pipelineOpen && (
        <>
          <div className="pipeline-drawer-backdrop" onClick={() => setPipelineOpen(false)} />
          <div className="pipeline-drawer">
            <div className="pipeline-drawer-header">
              <span className="pipeline-drawer-title">Pipeline</span>
              <button className="pipeline-drawer-close" onClick={() => setPipelineOpen(false)}>×</button>
            </div>
            <div className="pipeline-body">
              <div className="pipeline-timeline">
                {/* Router */}
                <div className="pipeline-step" data-type="router" style={{ animationDelay: '0ms' }}>
                  <div className={`pipeline-dot ${pipelineTrace ? 'done' : 'active'}`} />
                  <div className="pipeline-step-header">
                    <span className="pipeline-step-label">Router</span>
                    <span className={`pipeline-step-status ${pipelineTrace ? 'done' : 'active'}`}>
                      {pipelineTrace ? '✓' : '● routing'}
                    </span>
                  </div>
                  {pipelineTrace && (
                    <div className="pipeline-detail">
                      <div className="pipeline-detail-label">Decision</div>
                      {pipelineTrace.router_decision}{pipelineTrace.router_reason ? ` - ${pipelineTrace.router_reason}` : ''}
                      {pipelineTrace.tools_selected && pipelineTrace.tools_selected.length > 0 && (
                        <>
                          {'\n'}
                          <div className="pipeline-detail-label">Tools</div>
                          {pipelineTrace.tools_selected.join(', ')}
                        </>
                      )}
                    </div>
                  )}
                </div>

                {/* Planner */}
                {pipelineStrategy !== 'direct' && (
                  <div className="pipeline-step" data-type="planner" style={{ animationDelay: '80ms' }}>
                    <div className={`pipeline-dot ${pipelinePlan.length > 0 ? 'done' : 'waiting'}`} />
                    <div className="pipeline-step-header">
                      <span className="pipeline-step-label">Planner</span>
                      <span className={`pipeline-step-status ${pipelinePlan.length > 0 ? 'done' : ''}`}>
                        {pipelinePlan.length > 0 ? `✓ ${pipelinePlan.length} steps` : ''}
                      </span>
                    </div>
                    {(pipelineTrace?.planner_think || pipelinePlan.length > 0) && (
                      <div className="pipeline-detail">
                        {pipelineTrace?.planner_think && (
                          <>
                            <div className="pipeline-detail-label">Reasoning</div>
                            {pipelineTrace.planner_think}
                          </>
                        )}
                        {pipelinePlan.length > 0 && (
                          <>
                            {pipelineTrace?.planner_think && '\n'}
                            <div className="pipeline-detail-label">Plan</div>
                            {pipelinePlan.map((s: any) => `${s.id} = ${s.tool}(${Object.entries(s.params || {}).map(([k,v]) => `${k}="${v}"`).join(', ')})`).join('\n')}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Executor */}
                {pipelineStrategy !== 'direct' && (
                  <div className="pipeline-step" data-type="executor" style={{ animationDelay: '160ms' }}>
                    <div className={`pipeline-dot ${Object.keys(pipelineResults).length > 0 ? 'done' : 'waiting'}`} />
                    <div className="pipeline-step-header">
                      <span className="pipeline-step-label">Executor</span>
                      <span className={`pipeline-step-status ${Object.keys(pipelineResults).length > 0 ? 'done' : ''}`}>
                        {Object.keys(pipelineResults).length > 0 ? `✓ ${Object.keys(pipelineResults).length} calls` : ''}
                      </span>
                    </div>
                    {Object.keys(pipelineResults).length > 0 && (
                      <div className="pipeline-detail">
                        {pipelinePlan.map((step: any) => (
                          <div key={step.id} className="pipeline-tool-call">
                            <div className="pipeline-tool-name">{step.id}: {step.tool}</div>
                            {pipelineResults[step.id] && (
                              <div className="pipeline-tool-result">
                                → {pipelineResults[step.id].length > 150 ? pipelineResults[step.id].slice(0, 150) + '…' : pipelineResults[step.id]}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* ReAct Fallback */}
                {pipelineTrace?.fallback_used && (
                  <div className="pipeline-step" data-type="react" style={{ animationDelay: '240ms' }}>
                    <div className={`pipeline-dot ${pipelineAnswer ? 'done' : 'active'}`} />
                    <div className="pipeline-step-header">
                      <span className="pipeline-step-label">ReAct Fallback</span>
                      <span className="pipeline-step-status" style={{ color: '#E06B5E' }}>fallback</span>
                    </div>
                    <div className="pipeline-detail">
                      {pipelineTrace.fallback_reason || 'REWOO plan failed, switching to iterative reasoning'}
                    </div>
                  </div>
                )}

                {/* Solver */}
                <div className="pipeline-step" data-type="solver" style={{ animationDelay: pipelineStrategy === 'direct' ? '80ms' : '240ms' }}>
                  <div className={`pipeline-dot ${pipelineAnswer ? 'done' : 'waiting'}`} />
                  <div className="pipeline-step-header">
                    <span className="pipeline-step-label">{pipelineStrategy === 'direct' ? 'Answer' : 'Solver'}</span>
                    <span className={`pipeline-step-status ${pipelineAnswer ? 'done' : ''}`}>
                      {pipelineAnswer ? '✓' : ''}
                    </span>
                  </div>
                  {pipelineAnswer && (
                    <div className="pipeline-detail">
                      {pipelineAnswer.length > 300 ? pipelineAnswer.slice(0, 300) + '…' : pipelineAnswer}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Token Modal */}
      {tokenModal && (
        <div className="modal-overlay" onClick={() => setTokenModal(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '420px' }}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium" style={{ color: 'var(--text)' }}>
                Connect {tokenModal.name}
              </h2>
              <button
                onClick={() => setTokenModal(null)}
                className="text-lg"
                style={{ color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}
              >
                ×
              </button>
            </div>

            <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
              {tokenModal.description}. Paste your {tokenModal.token_label || 'token'} below.
              {' '}Token is stored in memory only, never saved.
            </p>

            <div className="mb-3">
              <input
                type="password"
                value={tokenInput}
                onChange={e => setTokenInput(e.target.value)}
                placeholder={tokenModal.token_hint || 'Paste token...'}
                onKeyDown={e => { if (e.key === 'Enter') handleMCPConnect(); }}
                autoFocus
                style={{ width: '100%', padding: '12px 16px' }}
              />
            </div>

            {tokenError && (
              <div
                className="text-xs mb-3 p-2 rounded"
                style={{ background: 'rgba(139, 74, 74, 0.1)', color: 'var(--error)' }}
              >
                {tokenError}
              </div>
            )}

            {tokenModal.setup_url && (
              <p className="text-xs mb-4" style={{ color: 'var(--text-secondary)' }}>
                Get one at{' '}
                <a
                  href={tokenModal.setup_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: 'var(--accent)', textDecoration: 'underline' }}
                >
                  {tokenModal.setup_url.replace('https://', '')}
                </a>
              </p>
            )}

            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setTokenModal(null)}
                className="btn btn-secondary btn-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleMCPConnect}
                disabled={!tokenInput.trim() || mcpConnections[tokenModal.registry_key]?.connecting}
                className="btn btn-primary btn-sm"
              >
                {mcpConnections[tokenModal.registry_key]?.connecting ? 'Connecting...' : 'Connect'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rate Limit Modal */}
      {isRateLimited && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2 className="text-xl font-medium mb-2" style={{ color: 'var(--text)' }}>
              You've tried 5 requests!
            </h2>
            <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
              Run unlimited locally with the OnsetLab SDK.
            </p>
            
            <div className="code-block mb-6">
              <code style={{ color: 'var(--accent)' }}>$ pip install onsetlab</code>
            </div>
            
            <div className="flex gap-2 justify-center mb-6">
              <button onClick={() => handleExport('config')} className="btn btn-secondary btn-sm">
                YAML
              </button>
              <button onClick={() => handleExport('docker')} className="btn btn-secondary btn-sm">
                Docker
              </button>
              <button onClick={() => handleExport('docker-vllm')} className="btn btn-secondary btn-sm">
                vLLM
              </button>
              <button onClick={() => handleExport('binary')} className="btn btn-secondary btn-sm">
                Script
              </button>
            </div>
            
            <a
              href="https://github.com/riyanshibohra/OnsetLab"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              Star on GitHub
            </a>
            
            <p className="text-xs mt-4" style={{ color: 'var(--text-secondary)', opacity: 0.7 }}>
              If you found this useful, a star really helps!
            </p>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer 
        className="px-6 py-8"
        style={{ borderTop: '1px solid var(--border)' }}
      >
        <div className="max-w-5xl mx-auto flex items-center justify-between text-sm">
          <span style={{ color: 'var(--text-secondary)' }}>
            onsetlab · Open source
          </span>
          <div className="flex items-center gap-6" style={{ color: 'var(--text-secondary)' }}>
            <a href="https://github.com/riyanshibohra/OnsetLab" className="hover:opacity-70 transition" style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              GitHub
            </a>
            <a href="https://github.com/riyanshibohra/OnsetLab#readme" className="hover:opacity-70 transition">
              Docs
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
