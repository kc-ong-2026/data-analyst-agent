import axios from 'axios';
import type { ChatRequest, ChatResponse, ConfigResponse, ProvidersResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatApi = {
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
  getConfig: async (): Promise<ConfigResponse> => {
    const response = await apiClient.get<ConfigResponse>('/config/');
    return response.data;
  },

  getProviders: async (): Promise<ProvidersResponse> => {
    const response = await apiClient.get<ProvidersResponse>('/config/providers');
    return response.data;
  },

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
