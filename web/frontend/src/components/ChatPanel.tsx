import { useRef, useEffect, useState } from 'react';
import type { Message } from '../types';

const EXAMPLES = [
  "What's 15% tip on $84.50?",
  "What day of the week was Jan 1, 2000?",
  "Convert 72°F to Celsius",
  "Generate a secure 16-char password",
];

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  onSend: (message: string) => void;
  disabled: boolean;
}

export function ChatPanel({ messages, isLoading, onSend, disabled }: ChatPanelProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading || disabled) return;
    onSend(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-panel">
      {isEmpty && !isLoading ? (
        <div className="empty-state">
          <div>
            <div className="empty-state-title">Try a query</div>
            <div className="empty-state-subtitle">
              See the agent plan, execute, and answer in real time
            </div>
          </div>
          <div className="example-cards">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                className="example-card"
                onClick={() => {
                  onSend(example);
                  inputRef.current?.focus();
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-bubble">
                {formatMessage(msg.content)}
              </div>
              <div className="message-meta">
                {msg.role === 'assistant' && msg.strategy && (
                  <span>{msg.strategy}</span>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="thinking-indicator">
              <div className="thinking-dots">
                <span />
                <span />
                <span />
              </div>
              Thinking...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <input
            ref={inputRef}
            className="chat-input"
            type="text"
            placeholder={disabled ? 'Query limit reached' : 'Ask something...'}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || disabled}
          />
          <button
            className="send-button"
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading || disabled}
            aria-label="Send message"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function formatMessage(content: string): React.ReactNode {
  // Simple markdown: **bold**, `code`, newlines
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*|`([^`]+)`|\n)/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = regex.exec(content)) !== null) {
    // Text before match
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }

    if (match[2]) {
      // Bold
      parts.push(<strong key={key++}>{match[2]}</strong>);
    } else if (match[3]) {
      // Code
      parts.push(<code key={key++}>{match[3]}</code>);
    } else if (match[0] === '\n') {
      parts.push(<br key={key++} />);
    }

    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }

  return parts.length > 0 ? parts : content;
}
