// Mock implementation for API client
export const chatApi = {
  streamMessage: jest.fn(),
  sendMessage: jest.fn(),
  getHistory: jest.fn(),
  clearHistory: jest.fn().mockResolvedValue({ success: true }),
  listConversations: jest.fn(),
};

export const configApi = {
  healthCheck: jest.fn(),
};

export const dataApi = {
  listDatasets: jest.fn(),
  getDatasetInfo: jest.fn(),
  queryDataset: jest.fn(),
};

const apiClient = {
  post: jest.fn(),
  get: jest.fn(),
  delete: jest.fn(),
};

export default apiClient;
