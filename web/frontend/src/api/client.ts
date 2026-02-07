// API Client

import type { ChatResponse, SessionInfo, ToolInfo, ModelInfo, MCPServerInfo, MCPConnectResponse, MCPStatusResponse } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(response.status, error.detail || 'Request failed');
    }

    return response.json();
  }

  async getSession(): Promise<SessionInfo> {
    return this.request<SessionInfo>('/api/session');
  }

  async getTools(): Promise<ToolInfo[]> {
    return this.request<ToolInfo[]>('/api/tools');
  }

  async getModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>('/api/models');
  }

  async getMCPServers(): Promise<MCPServerInfo[]> {
    return this.request<MCPServerInfo[]>('/api/mcp-servers');
  }

  async connectMCP(server: string, token: string): Promise<MCPConnectResponse> {
    return this.request<MCPConnectResponse>('/api/mcp/connect', {
      method: 'POST',
      body: JSON.stringify({ server, token }),
    });
  }

  async disconnectMCP(server: string): Promise<{ disconnected: boolean; server: string }> {
    return this.request('/api/mcp/disconnect', {
      method: 'POST',
      body: JSON.stringify({ server }),
    });
  }

  async getMCPStatus(): Promise<MCPStatusResponse> {
    return this.request<MCPStatusResponse>('/api/mcp/status');
  }

  async chat(
    message: string,
    tools: string[],
    model: string
  ): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message, tools, model }),
    });
  }

  async exportAgent(
    format: 'config' | 'docker' | 'docker-vllm' | 'binary',
    tools: string[],
    model: string
  ): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/export`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format, tools, model }),
    });

    if (!response.ok) {
      throw new ApiError(response.status, 'Export failed');
    }

    return response.blob();
  }
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : 'API Error');
    this.status = status;
    this.detail = detail;
  }

  get isRateLimited(): boolean {
    return this.status === 429;
  }
}

export const api = new ApiClient(API_URL);
