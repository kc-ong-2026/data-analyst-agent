import { create } from 'zustand';
import type { Message, VisualizationData, AgentStageInfo } from '../types';
import { chatApi } from '../api/client';

const STAGE_LABELS: Record<string, string> = {
  verification: 'Verifying query',
  coordinator: 'Planning analysis',
  extraction: 'Extracting data',
  analytics: 'Analyzing results',
};

function getHumanReadableStage(agent: string, status: string): string | null {
  if (status !== 'running') return null;
  return STAGE_LABELS[agent] || agent;
}

interface ChatState {
  messages: Message[];
  conversationId: string | null;
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  currentVisualization: VisualizationData | null;
  currentAgent: string | null;
  streamingMessage: string;
  useStreaming: boolean;
  agentStages: AgentStageInfo[];
  currentStage: string | null;

  // Actions
  sendMessage: (content: string) => Promise<void>;
  clearMessages: () => void;
  setVisualization: (viz: VisualizationData | null) => void;
  toggleStreaming: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  conversationId: null,
  isLoading: false,
  isStreaming: false,
  error: null,
  currentVisualization: null,
  currentAgent: null,
  streamingMessage: '',
  useStreaming: true, // Enable streaming by default
  agentStages: [],
  currentStage: null,

  sendMessage: async (content: string) => {
    const { conversationId, useStreaming } = get();

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
      isStreaming: useStreaming,
      error: null,
      streamingMessage: '',
      currentAgent: null,
      agentStages: [],
      currentStage: null,
    }));

    try {
      if (useStreaming) {
        // Use SSE streaming for real-time response
        const assistantMessageId = `msg-${Date.now()}-assistant`;

        await chatApi.streamMessage(
          {
            message: content,
            conversation_id: conversationId || undefined,
            include_visualization: true,
          },
          {
            onStart: (newConversationId) => {
              console.log('[Stream] Started:', newConversationId);
              set({ conversationId: newConversationId });
            },

            onAgent: (agent, status, message) => {
              console.log(`[Stream] Agent: ${agent} - ${status}`, message);

              set((state) => {
                const newStages = [...state.agentStages];
                const existingIndex = newStages.findIndex(s => s.agent === agent);

                const stageInfo: AgentStageInfo = {
                  agent,
                  status: status as 'running' | 'complete',
                  message,
                  startTime: status === 'running' ? Date.now() : undefined,
                  endTime: status === 'complete' ? Date.now() : undefined,
                };

                if (existingIndex >= 0) {
                  // Update existing stage
                  newStages[existingIndex] = {
                    ...newStages[existingIndex],
                    ...stageInfo,
                    startTime: newStages[existingIndex].startTime || stageInfo.startTime,
                  };
                } else {
                  // Add new stage
                  newStages.push(stageInfo);
                }

                return {
                  agentStages: newStages,
                  currentStage: getHumanReadableStage(agent, status),
                  currentAgent: status === 'running' ? agent : null,
                };
              });
            },

            onToken: (token) => {
              // Progressive message building
              set((state) => {
                const newContent = state.streamingMessage + token;

                // Update or create assistant message
                const existingMsgIndex = state.messages.findIndex(m => m.id === assistantMessageId);
                const newMessages = [...state.messages];

                if (existingMsgIndex >= 0) {
                  // Update existing message
                  newMessages[existingMsgIndex] = {
                    ...newMessages[existingMsgIndex],
                    content: newContent,
                  };
                } else {
                  // Create new message
                  const newMsg: Message = {
                    id: assistantMessageId,
                    role: 'assistant',
                    content: newContent,
                    timestamp: new Date(),
                  };
                  newMessages.push(newMsg);
                }

                return {
                  messages: newMessages,
                  streamingMessage: newContent,
                };
              });
            },

            onVisualization: (viz) => {
              console.log('[Stream] Visualization received');
              set((state) => {
                // Add visualization to the assistant message
                const msgIndex = state.messages.findIndex(m => m.id === assistantMessageId);
                if (msgIndex >= 0) {
                  const newMessages = [...state.messages];
                  newMessages[msgIndex] = {
                    ...newMessages[msgIndex],
                    visualization: viz,
                  };
                  return {
                    messages: newMessages,
                    currentVisualization: viz,
                  };
                }
                return { currentVisualization: viz };
              });
            },

            onComplete: () => {
              console.log('[Stream] Complete');
              set({
                isLoading: false,
                isStreaming: false,
                streamingMessage: '',
                currentAgent: null,
                currentStage: null,
              });
            },

            onError: (error) => {
              console.error('[Stream] Error:', error);
              set({
                isLoading: false,
                isStreaming: false,
                error,
                streamingMessage: '',
                currentAgent: null,
                currentStage: null,
              });
            },
          }
        );
      } else {
        // Fallback to traditional request/response
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
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      set({
        isLoading: false,
        isStreaming: false,
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
      streamingMessage: '',
      currentAgent: null,
      agentStages: [],
      currentStage: null,
    });
  },

  setVisualization: (viz) => {
    set({ currentVisualization: viz });
  },

  toggleStreaming: () => {
    set((state) => ({ useStreaming: !state.useStreaming }));
  },
}));
