// Sidebar component

import type { ToolInfo, ModelInfo } from '../types';

interface Props {
  tools: ToolInfo[];
  models: ModelInfo[];
  selectedTools: string[];
  selectedModel: string;
  onToggleTool: (name: string) => void;
  onSelectModel: (id: string) => void;
  onExport: (format: 'config' | 'docker' | 'binary') => void;
}

export function Sidebar({
  tools,
  models,
  selectedTools,
  selectedModel,
  onToggleTool,
  onSelectModel,
  onExport,
}: Props) {
  return (
    <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-900">OnsetLab</h1>
        <p className="text-sm text-gray-500">Playground</p>
      </div>

      {/* Tools section */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
          Tools
        </h2>
        <div className="space-y-2">
          {tools.map((tool) => (
            <label
              key={tool.name}
              className="flex items-center gap-3 cursor-pointer group"
            >
              <input
                type="checkbox"
                checked={selectedTools.includes(tool.name)}
                onChange={() => onToggleTool(tool.name)}
                className="w-4 h-4 rounded border-gray-300 text-primary-500 focus:ring-primary-500"
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-700 group-hover:text-gray-900">
                  {tool.name}
                </div>
                <div className="text-xs text-gray-400 truncate">
                  {tool.description}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Model section */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
          Model
        </h2>
        <select
          value={selectedModel}
          onChange={(e) => onSelectModel(e.target.value)}
          className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.display_name}
            </option>
          ))}
        </select>
      </div>

      {/* Export section */}
      <div className="p-4 flex-1">
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
          Export
        </h2>
        <div className="space-y-2">
          <button
            onClick={() => onExport('config')}
            className="w-full px-3 py-2 text-sm text-left text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
          >
            📄 Config (YAML)
          </button>
          <button
            onClick={() => onExport('docker')}
            className="w-full px-3 py-2 text-sm text-left text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
          >
            🐳 Docker Package
          </button>
          <button
            onClick={() => onExport('binary')}
            className="w-full px-3 py-2 text-sm text-left text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
          >
            🐍 Python Script
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <a
          href="https://github.com/riyanshibohra/onsetlab"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path
              fillRule="evenodd"
              d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              clipRule="evenodd"
            />
          </svg>
          View on GitHub
        </a>
        <p className="text-xs text-gray-400 mt-2">
          pip install onsetlab
        </p>
      </div>
    </div>
  );
}
