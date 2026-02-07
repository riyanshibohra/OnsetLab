// API Types

export interface PlanStep {
  id: string;
  tool: string;
  params: Record<string, unknown>;
  result?: string;
  status: 'pending' | 'running' | 'done' | 'error';
}

export interface PipelineTrace {
  router_decision: string;
  router_reason: string;
  tools_total: number;
  tools_filtered: number;
  tools_selected: string[];
  tool_rules: string;
  planner_think: string;
  planner_prompt: string;
  fallback_used: boolean;
  fallback_reason: string;
}

export interface ChatResponse {
  answer: string;
  plan: PlanStep[];
  results: Record<string, string>;
  strategy: string;
  slm_calls: number;
  requests_remaining: number;
  trace?: PipelineTrace;
}

export interface SessionInfo {
  session_id: string;
  requests_used: number;
  requests_limit: number;
  requests_remaining: number;
  created_at: string;
  model: string;
  tools: string[];
}

export interface ToolInfo {
  name: string;
  description: string;
  enabled_by_default: boolean;
  category?: string;
}

export interface ModelInfo {
  id: string;
  display_name: string;
  description: string;
  params?: string;
  badge?: string;
}

export interface MCPServerInfo {
  name: string;
  description: string;
  registry_key: string;
  requires_token: boolean;
  token_label?: string;
  setup_url?: string;
  token_hint?: string;
}

export interface MCPConnectResponse {
  connected: boolean;
  server: string;
  tools: string[];
}

export interface MCPStatusServer {
  connected: boolean;
  tools: string[];
}

export interface MCPStatusResponse {
  servers: Record<string, MCPStatusServer>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  plan?: PlanStep[];
  strategy?: string;
  trace?: PipelineTrace;
  timestamp: Date;
}
