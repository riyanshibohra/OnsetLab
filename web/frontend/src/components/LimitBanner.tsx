// Rate limit banner

interface Props {
  onExport: (format: 'config' | 'docker' | 'binary') => void;
}

export function LimitBanner({ onExport }: Props) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 max-w-md mx-4 shadow-2xl">
        <div className="text-center">
          <div className="text-5xl mb-4">🎉</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            You've used all 5 free requests!
          </h2>
          <p className="text-gray-600 mb-6">
            Run unlimited locally with the OnsetLab SDK
          </p>

          {/* Install command */}
          <div className="bg-gray-900 rounded-lg p-4 mb-6">
            <code className="text-green-400 text-sm">
              pip install onsetlab
            </code>
          </div>

          {/* Export buttons */}
          <div className="space-y-2 mb-6">
            <p className="text-sm text-gray-500 mb-2">
              Download your agent configuration:
            </p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => onExport('config')}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
              >
                📄 Config
              </button>
              <button
                onClick={() => onExport('docker')}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
              >
                🐳 Docker
              </button>
              <button
                onClick={() => onExport('binary')}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
              >
                🐍 Script
              </button>
            </div>
          </div>

          {/* GitHub link */}
          <a
            href="https://github.com/riyanshibohra/onsetlab"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
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
        </div>
      </div>
    </div>
  );
}
