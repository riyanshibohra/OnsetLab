// Session hook

import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { SessionInfo, ToolInfo, ModelInfo } from '../types';

interface UseSessionReturn {
  session: SessionInfo | null;
  tools: ToolInfo[];
  models: ModelInfo[];
  isLoading: boolean;
  selectedTools: string[];
  selectedModel: string;
  setSelectedTools: (tools: string[]) => void;
  setSelectedModel: (model: string) => void;
  toggleTool: (toolName: string) => void;
}

export function useSession(): UseSessionReturn {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('llama3.1:8b');

  useEffect(() => {
    async function init() {
      try {
        const [sessionData, toolsData, modelsData] = await Promise.all([
          api.getSession(),
          api.getTools(),
          api.getModels(),
        ]);

        setSession(sessionData);
        setTools(toolsData);
        setModels(modelsData);

        // Set default selected tools
        const defaultTools = toolsData
          .filter((t) => t.enabled_by_default)
          .map((t) => t.name);
        setSelectedTools(defaultTools);

        // Set model from session
        setSelectedModel(sessionData.model);
      } catch (err) {
        console.error('Failed to initialize session:', err);
      } finally {
        setIsLoading(false);
      }
    }

    init();
  }, []);

  const toggleTool = (toolName: string) => {
    setSelectedTools((prev) =>
      prev.includes(toolName)
        ? prev.filter((t) => t !== toolName)
        : [...prev, toolName]
    );
  };

  return {
    session,
    tools,
    models,
    isLoading,
    selectedTools,
    selectedModel,
    setSelectedTools,
    setSelectedModel,
    toggleTool,
  };
}
