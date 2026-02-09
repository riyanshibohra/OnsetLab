import { useState, useRef, useEffect } from 'react';
import { api, ApiError } from './api/client';
import type { Message, ToolInfo, ModelInfo, MCPServerInfo } from './types';

// Default data (fallback if API fails)
const DEFAULT_TOOLS: ToolInfo[] = [
  { name: 'Calculator', description: 'Math expressions', enabled_by_default: true, category: 'builtin' },
  { name: 'DateTime', description: 'Date/time queries', enabled_by_default: true, category: 'builtin' },
  { name: 'UnitConverter', description: 'Unit conversion', enabled_by_default: true, category: 'builtin' },
  { name: 'TextProcessor', description: 'Text operations', enabled_by_default: true, category: 'builtin' },
  { name: 'RandomGenerator', description: 'Random values', enabled_by_default: true, category: 'builtin' },
];

const DEFAULT_MODELS: ModelInfo[] = [
  { id: 'qwen3-a3b', display_name: 'Qwen3 A3B', description: 'Best for tool calling — MoE', params: '3B active', badge: 'recommended' },
  { id: 'qwen2.5:7b', display_name: 'Qwen 2.5 7B', description: 'Strong all-rounder', params: '7B', badge: 'best value' },
  { id: 'hermes3:8b', display_name: 'Hermes 3 8B', description: 'Excellent function calling', params: '8B', badge: 'function calling' },
  { id: 'mistral:7b', display_name: 'Mistral 7B', description: 'Fast, reliable', params: '7B', badge: 'fastest' },
  { id: 'gemma3:4b', display_name: 'Gemma 3 4B', description: 'Smallest, simple tasks', params: '4B' },
  { id: 'qwen2.5-coder:7b', display_name: 'Qwen 2.5 Coder 7B', description: 'Optimized for code', params: '7B', badge: 'code' },
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
  const [models] = useState<ModelInfo[]>(DEFAULT_MODELS);
  const [mcpServers] = useState<MCPServerInfo[]>(DEFAULT_MCP_SERVERS);
  const [selectedTools, setSelectedTools] = useState<string[]>(
    DEFAULT_TOOLS.filter(t => t.enabled_by_default).map(t => t.name)
  );
  const [selectedModel, setSelectedModel] = useState('qwen3-a3b');

  // Pipeline trace expansion state
  const [expandedTraces, setExpandedTraces] = useState<Set<string>>(new Set());
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  // MCP state
  const [mcpConnections, setMcpConnections] = useState<Record<string, MCPConnectionState>>({});
  const [tokenModal, setTokenModal] = useState<MCPServerInfo | null>(null);
  const [tokenInput, setTokenInput] = useState('');
  const [tokenError, setTokenError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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

    try {
      const response = await api.chat(input.trim(), selectedTools, selectedModel);
      
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
    { label: 'Tip Calculator', query: "I'm splitting a $284.50 dinner bill between 4 people with 20% tip. What's each person's share?" },
    { label: 'Time Zones', query: "What time is it right now, and how many hours until midnight?" },
    { label: 'Unit Math', query: "A recipe needs 2.5 cups of flour. I only have a 1/3 cup measure. How many scoops?" },
    { label: 'Date Calc', query: "How many days are between March 15 and December 25 this year?" },
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
            >
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-20 pb-12 px-6">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <div className="py-8 mb-6">
            <h1 className="text-2xl font-medium mb-2" style={{ color: 'var(--text)' }}>
              Test the Agent
            </h1>
            <p style={{ color: 'var(--text-secondary)' }}>
              Try tool-calling with local SLMs. {requestsRemaining} free requests, then download the SDK.
            </p>
          </div>

          {/* Layout */}
          <div className="flex gap-6">
            {/* Sidebar */}
            <aside className="w-64 shrink-0">
              <div className="card sticky top-24" style={{ maxHeight: 'calc(100vh - 120px)', overflowY: 'auto', padding: '20px' }}>
                {/* Tools */}
                <div className="mb-4">
                  <div className="section-label">Built-in Tools</div>
                  <div className="space-y-1">
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
                </div>

                {/* MCP Servers */}
                <div className="mb-4">
                  <div className="section-label flex items-center justify-between">
                    <span>MCP Servers</span>
                    {connectedMCPCount > 0 && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'rgba(74, 124, 89, 0.15)', color: 'var(--success)' }}>
                        {connectedMCPCount} live
                      </span>
                    )}
                  </div>
                  <div className="space-y-1.5">
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
                  </div>
                </div>

                <div style={{ height: '1px', background: 'var(--border)', margin: '12px 0' }} />

                {/* Model */}
                <div className="mb-4">
                  <div className="section-label">Model</div>
                  <div className="space-y-1">
                    {models.map(m => (
                      <button
                        key={m.id}
                        onClick={() => setSelectedModel(m.id)}
                        className="w-full text-left px-3 py-2 rounded text-xs transition-colors cursor-pointer"
                        style={{
                          background: selectedModel === m.id ? 'rgba(74, 102, 112, 0.12)' : 'transparent',
                          border: selectedModel === m.id ? '1px solid var(--accent)' : '1px solid transparent',
                          color: selectedModel === m.id ? 'var(--text)' : 'var(--text-secondary)',
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-medium" style={{ color: selectedModel === m.id ? 'var(--text-primary)' : 'var(--text-primary)' }}>
                            {m.display_name}
                          </span>
                          <span className="opacity-50">{m.params}</span>
                          {m.badge && (
                            <span
                              className="ml-auto text-[9px] font-mono px-1.5 py-0.5 rounded"
                              style={{
                                background: m.badge === 'recommended'
                                  ? 'rgba(74, 124, 89, 0.2)'
                                  : 'rgba(255,255,255,0.06)',
                                color: m.badge === 'recommended'
                                  ? 'var(--success)'
                                  : 'var(--text-secondary)',
                              }}
                            >
                              {m.badge}
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] mt-0.5 opacity-60">{m.description}</div>
                      </button>
                    ))}
                  </div>
                  <div className="text-[11px] mt-1.5" style={{ color: 'var(--text-secondary)' }}>
                    All FREE via OpenRouter
                  </div>
                </div>

                {/* Export */}
                <div>
                  <div className="section-label">Export</div>
                  <div className="flex flex-wrap gap-1">
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
                  </div>
                </div>
              </div>
            </aside>

            {/* Chat Area */}
            <div className="flex-1 flex flex-col min-h-[560px]">
              <div className="card flex-1 flex flex-col overflow-hidden" style={{ padding: 0 }}>
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6">
                  {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center py-12">
                      <pre
                        className="leading-tight mb-4"
                        style={{ color: 'var(--accent)', fontSize: '6px', opacity: 0.35 }}
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
                                  <span style={{ color: 'var(--text-secondary)' }}> — {msg.trace.router_reason}</span>
                                </div>

                                {/* Tools */}
                                {msg.trace.tools_filtered > 0 && (
                                  <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                                    <span style={{ color: 'var(--text-secondary)' }}>Tools: </span>
                                    <span style={{ color: 'var(--accent)' }}>
                                      {msg.trace.tools_filtered} of {msg.trace.tools_total} selected
                                    </span>
                                    <span style={{ color: 'var(--text-secondary)' }}>
                                      {' — '}{msg.trace.tools_selected.slice(0, 5).join(', ')}
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
                            
                            {/* Execution Plan (compact, always visible) */}
                            {msg.plan && msg.plan.length > 0 && !expandedTraces.has(msg.id) && (
                              <div 
                                className="mt-3 pt-3"
                                style={{ borderTop: '1px solid var(--border)' }}
                              >
                                <div className="space-y-1.5">
                                  {msg.plan.map(step => (
                                    <div 
                                      key={step.id} 
                                      className="font-mono text-xs flex items-start gap-2"
                                    >
                                      <span style={{ color: 'var(--text-secondary)' }}>{step.id}</span>
                                      <span style={{ color: 'var(--accent)' }}>{step.tool}</span>
                                      <span style={{ color: 'var(--text-secondary)' }}>→</span>
                                      <span className="break-all" style={{ color: step.status === 'error' ? 'var(--error)' : 'var(--success)' }}>
                                        {step.result && step.result.length > 120 ? step.result.slice(0, 120) + '…' : step.result}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
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
                      {card.label}
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
                      placeholder={isRateLimited ? "Limit reached — download the SDK" : "Type your question..."}
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
              {' '}Token is stored in memory only — never saved.
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
            >
              View on GitHub
            </a>
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
            <a href="https://github.com/riyanshibohra/OnsetLab" className="hover:opacity-70 transition">
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
