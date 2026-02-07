// Chat hook

import { useState, useCallback } from 'react';
import { api, ApiError } from '../api/client';
import type { Message, PlanStep } from '../types';

interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  requestsRemaining: number;
  isRateLimited: boolean;
  sendMessage: (content: string, tools: string[], model: string) => Promise<void>;
  clearMessages: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestsRemaining, setRequestsRemaining] = useState(5);
  const [isRateLimited, setIsRateLimited] = useState(false);

  const sendMessage = useCallback(
    async (content: string, tools: string[], model: string) => {
      if (isRateLimited) return;
      
      setError(null);
      setIsLoading(true);

      // Add user message
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      try {
        const response = await api.chat(content, tools, model);

        // Add assistant message
        const assistantMessage: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer,
          plan: response.plan,
          strategy: response.strategy,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setRequestsRemaining(response.requests_remaining);
        
        if (response.requests_remaining <= 0) {
          setIsRateLimited(true);
        }
      } catch (err) {
        if (err instanceof ApiError && err.isRateLimited) {
          setIsRateLimited(true);
          setRequestsRemaining(0);
        } else {
          setError(err instanceof Error ? err.message : 'Something went wrong');
        }
      } finally {
        setIsLoading(false);
      }
    },
    [isRateLimited]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    requestsRemaining,
    isRateLimited,
    sendMessage,
    clearMessages,
  };
}
