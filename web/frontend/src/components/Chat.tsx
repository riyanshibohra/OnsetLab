// Chat component

import { useState, useRef, useEffect } from 'react';
import { Message } from './Message';
import type { Message as MessageType } from '../types';

interface Props {
  messages: MessageType[];
  isLoading: boolean;
  error: string | null;
  requestsRemaining: number;
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function Chat({
  messages,
  isLoading,
  error,
  requestsRemaining,
  onSend,
  disabled,
}: Props) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && !isLoading) {
      onSend(input.trim());
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400">
            <div className="text-center">
              <div className="text-4xl mb-4">🤖</div>
              <p className="text-lg font-medium">OnsetLab Playground</p>
              <p className="text-sm mt-2">
                Try asking: "What is 15% tip on $84.50?"
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <Message key={msg.id} message={msg} />
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin h-4 w-4 border-2 border-primary-500 border-t-transparent rounded-full" />
                    <span className="text-gray-500">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error display */}
      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-200 text-red-600 text-sm">
          {error}
        </div>
      )}

      {/* Input area */}
      <form
        onSubmit={handleSubmit}
        className="p-4 border-t border-gray-200 bg-white"
      >
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              disabled
                ? "You've reached the limit. Download the SDK!"
                : 'Ask me anything...'
            }
            disabled={disabled || isLoading}
            className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-400"
          />
          <button
            type="submit"
            disabled={disabled || isLoading || !input.trim()}
            className="px-6 py-3 bg-primary-500 text-white rounded-xl font-medium hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
        <div className="mt-2 text-xs text-gray-400 text-center">
          {requestsRemaining} request{requestsRemaining !== 1 ? 's' : ''} remaining
        </div>
      </form>
    </div>
  );
}
