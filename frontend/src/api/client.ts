import axios from 'axios';
import type { ChatRequest, ChatResponse, VisualizationData } from '../types';
import { getApiUrl } from '../config/env';

const API_BASE_URL = getApiUrl();

/**
 * SSE Event types from backend
 */
export interface SSEEvent {
  type: 'start' | 'agent' | 'token' | 'data' | 'visualization' | 'done' | 'error';
  conversation_id?: string;
  agent?: string;
  status?: string;
  message?: string;
  content?: string;
  visualization?: VisualizationData;
  error?: string;
  data_type?: string;
  summary?: string;
}

/**
 * Callbacks for SSE streaming
 */
export interface StreamCallbacks {
  onStart?: (conversationId: string) => void;
  onAgent?: (agent: string, status: string, message?: string) => void;
  onToken?: (token: string, agent?: string) => void;
  onData?: (dataType: string, summary: string) => void;
  onVisualization?: (viz: VisualizationData) => void;
  onComplete?: (fullMessage: string) => void;
  onError?: (error: string) => void;
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatApi = {
  /**
   * Stream chat response using Server-Sent Events (SSE)
   * Provides real-time token streaming for immediate user feedback
   */
  streamMessage: async (
    request: ChatRequest,
    callbacks: StreamCallbacks
  ): Promise<void> => {
    const url = `${API_BASE_URL}/chat/stream`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('ReadableStream not supported');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let messageBuffer = '';

      // Read stream
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (lines ending with \n\n)
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) {
            continue;
          }

          // Parse SSE data
          const data = line.slice(6); // Remove 'data: ' prefix
          try {
            const event: SSEEvent = JSON.parse(data);

            switch (event.type) {
              case 'start':
                if (event.conversation_id && callbacks.onStart) {
                  callbacks.onStart(event.conversation_id);
                }
                break;

              case 'agent':
                if (event.agent && event.status && callbacks.onAgent) {
                  callbacks.onAgent(event.agent, event.status, event.message);
                }
                break;

              case 'token':
                if (event.content && callbacks.onToken) {
                  messageBuffer += event.content;
                  callbacks.onToken(event.content, event.agent);
                }
                break;

              case 'data':
                if (event.data_type && event.summary && callbacks.onData) {
                  callbacks.onData(event.data_type, event.summary);
                }
                break;

              case 'visualization':
                if (event.visualization && callbacks.onVisualization) {
                  callbacks.onVisualization(event.visualization);
                }
                break;

              case 'done':
                if (callbacks.onComplete) {
                  callbacks.onComplete(messageBuffer);
                }
                break;

              case 'error':
                if (event.error && callbacks.onError) {
                  callbacks.onError(event.error);
                }
                break;
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, data);
          }
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Stream connection failed';
      if (callbacks.onError) {
        callbacks.onError(errorMessage);
      }
      throw error;
    }
  },

  /**
   * Send message using traditional request/response (fallback)
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>('/chat/', request);
    return response.data;
  },

  getHistory: async (conversationId: string) => {
    const response = await apiClient.get(`/chat/history/${conversationId}`);
    return response.data;
  },

  clearHistory: async (conversationId: string) => {
    const response = await apiClient.delete(`/chat/history/${conversationId}`);
    return response.data;
  },

  listConversations: async () => {
    const response = await apiClient.get('/chat/conversations');
    return response.data;
  },
};

export const configApi = {
  healthCheck: async () => {
    const response = await apiClient.get('/config/health');
    return response.data;
  },
};

export const dataApi = {
  listDatasets: async () => {
    const response = await apiClient.get('/data/datasets');
    return response.data;
  },

  getDatasetInfo: async (datasetPath: string) => {
    const response = await apiClient.get(`/data/datasets/${datasetPath}/info`);
    return response.data;
  },

  queryDataset: async (datasetPath: string, params?: {
    query?: string;
    columns?: string;
    limit?: number;
  }) => {
    const response = await apiClient.get(`/data/datasets/${datasetPath}/query`, { params });
    return response.data;
  },
};

export default apiClient;
