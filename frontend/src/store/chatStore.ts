import { create } from 'zustand';
import type { Message, VisualizationData } from '../types';
import { chatApi } from '../api/client';

interface ChatState {
  messages: Message[];
  conversationId: string | null;
  isLoading: boolean;
  error: string | null;
  currentVisualization: VisualizationData | null;

  // Actions
  sendMessage: (content: string) => Promise<void>;
  clearMessages: () => void;
  setVisualization: (viz: VisualizationData | null) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  conversationId: null,
  isLoading: false,
  error: null,
  currentVisualization: null,

  sendMessage: async (content: string) => {
    const { conversationId } = get();

    // Add user message
    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date(),
    };

    set((state) => ({
      messages: [...state.messages, userMessage],
      isLoading: true,
      error: null,
    }));

    try {
      const response = await chatApi.sendMessage({
        message: content,
        conversation_id: conversationId || undefined,
        include_visualization: true,
      });

      const assistantMessage: Message = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
        visualization: response.visualization,
      };

      set((state) => ({
        messages: [...state.messages, assistantMessage],
        conversationId: response.conversation_id,
        isLoading: false,
        currentVisualization: response.visualization || state.currentVisualization,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      set({
        isLoading: false,
        error: errorMessage,
      });
    }
  },

  clearMessages: () => {
    const { conversationId } = get();
    if (conversationId) {
      chatApi.clearHistory(conversationId).catch(console.error);
    }
    set({
      messages: [],
      conversationId: null,
      currentVisualization: null,
      error: null,
    });
  },

  setVisualization: (viz) => {
    set({ currentVisualization: viz });
  },
}));
