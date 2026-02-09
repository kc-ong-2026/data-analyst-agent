import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot, BarChart2 } from 'lucide-react';
import type { Message } from '../types';
import { useChatStore } from '../store/chatStore';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const { setVisualization } = useChatStore();
  const isUser = message.role === 'user';

  const handleShowVisualization = () => {
    if (message.visualization) {
      setVisualization(message.visualization);
    }
  };

  return (
    <div
      className={`flex gap-4 p-4 ${
        isUser ? 'bg-white' : 'bg-gray-50'
      }`}
    >
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-primary-600 text-white' : 'bg-gray-700 text-white'
        }`}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm text-gray-600 mb-1">
          {isUser ? 'You' : 'Assistant'}
        </div>

        <div className="markdown-content text-gray-800">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {message.visualization && (
          <button
            onClick={handleShowVisualization}
            className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 text-sm
                       bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200
                       transition-colors"
          >
            <BarChart2 size={16} />
            View Visualization
          </button>
        )}
      </div>
    </div>
  );
};
