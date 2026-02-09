import { type ToolInfo } from '../types';

interface HeaderProps {
  tools: ToolInfo[];
  enabledTools: string[];
  onToggleTool: (name: string) => void;
  model: string;
  requestsRemaining: number;
  requestsLimit: number;
}

export function Header({
  tools,
  enabledTools,
  onToggleTool,
  model,
  requestsRemaining,
  requestsLimit,
}: HeaderProps) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-logo">
          OnsetLab<span>Playground</span>
        </div>
        <div className="header-divider" />
        <div className="tools-bar">
          {tools.map((tool) => (
            <button
              key={tool.name}
              className={`tool-pill ${enabledTools.includes(tool.name) ? 'active' : ''}`}
              onClick={() => onToggleTool(tool.name)}
              title={tool.description}
            >
              {enabledTools.includes(tool.name) ? '✓ ' : ''}
              {tool.name}
            </button>
          ))}
        </div>
      </div>
      <div className="header-right">
        <div className="model-badge">
          {model}
        </div>
        <div className="query-counter">
          <span className="count">{requestsRemaining}/{requestsLimit}</span>
          queries left
        </div>
      </div>
    </header>
  );
}
