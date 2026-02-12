import React, { useRef, useEffect } from 'react';
import { MessageSquare, Trash2 } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { AgentStatusIndicator } from './AgentStatusIndicator';
import { useChatStore } from '../store/chatStore';

export const ChatPanel: React.FC = () => {
  const { messages, isLoading, error, clearMessages } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <MessageSquare className="text-primary-600" size={20} />
          <h2 className="font-semibold text-gray-800">Chat</h2>
        </div>

        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="p-2 text-gray-500 hover:text-red-500 hover:bg-red-50
                       rounded-lg transition-colors"
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 p-8">
            <MessageSquare size={48} className="mb-4 text-gray-300" />
            <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
            <p className="text-sm text-center max-w-md">
              Ask questions about Singapore government data, employment statistics,
              income trends, and more.
            </p>
            <div className="mt-6 space-y-2 text-sm">
              <p className="text-gray-600 font-medium">Try asking:</p>
              <ul className="space-y-1 text-gray-500">
                <li>"What is the employment rate in Singapore?"</li>
                <li>"Show me income trends over the years"</li>
                <li>"Compare labour force by education level"</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}

            {isLoading && (
              <div className="p-4">
                <AgentStatusIndicator />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-200 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Input */}
      <ChatInput />
    </div>
  );
};
