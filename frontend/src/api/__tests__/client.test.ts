import type { ChatRequest, VisualizationData } from '../../types';

// Mock env config to avoid import.meta issues
jest.mock('../../config/env');

// Create mock axios instance
const mockPost = jest.fn();
const mockGet = jest.fn();
const mockDelete = jest.fn();

// Mock axios
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    post: mockPost,
    get: mockGet,
    delete: mockDelete,
  })),
}));

// Import after mocking
import { chatApi, configApi, dataApi } from '../client';

// Export mocks for use in tests
export { mockPost, mockGet, mockDelete };

describe('chatApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
    mockPost.mockClear();
    mockGet.mockClear();
    mockDelete.mockClear();
  });

  describe('streamMessage', () => {
    it('should handle SSE streaming correctly', async () => {
      const mockEvents = [
        'data: {"type":"start","conversation_id":"conv-123"}\n\n',
        'data: {"type":"agent","agent":"coordinator","status":"running","message":"Processing"}\n\n',
        'data: {"type":"token","content":"Hello"}\n\n',
        'data: {"type":"token","content":" world"}\n\n',
        'data: {"type":"done"}\n\n',
      ];

      const mockReadableStream = new ReadableStream({
        start(controller) {
          mockEvents.forEach(event => {
            controller.enqueue(new TextEncoder().encode(event));
          });
          controller.close();
        },
      });

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        body: mockReadableStream,
      });

      const callbacks = {
        onStart: jest.fn(),
        onAgent: jest.fn(),
        onToken: jest.fn(),
        onComplete: jest.fn(),
        onError: jest.fn(),
      };

      const request: ChatRequest = {
        message: 'Test message',
        include_visualization: true,
      };

      await chatApi.streamMessage(request, callbacks);

      expect(callbacks.onStart).toHaveBeenCalledWith('conv-123');
      expect(callbacks.onAgent).toHaveBeenCalledWith('coordinator', 'running', 'Processing');
      expect(callbacks.onToken).toHaveBeenCalledWith('Hello', undefined);
      expect(callbacks.onToken).toHaveBeenCalledWith(' world', undefined);
      expect(callbacks.onComplete).toHaveBeenCalledWith('Hello world');
      expect(callbacks.onError).not.toHaveBeenCalled();
    });

    it('should handle visualization events', async () => {
      const mockVisualization: VisualizationData = {
        chart_type: 'bar',
        title: 'Test Chart',
        data: [{ x: '2023', y: 100 }],
      };

      const mockEvents = [
        'data: {"type":"start","conversation_id":"conv-456"}\n\n',
        `data: {"type":"visualization","visualization":${JSON.stringify(mockVisualization)}}\n\n`,
        'data: {"type":"done"}\n\n',
      ];

      const mockReadableStream = new ReadableStream({
        start(controller) {
          mockEvents.forEach(event => {
            controller.enqueue(new TextEncoder().encode(event));
          });
          controller.close();
        },
      });

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        body: mockReadableStream,
      });

      const callbacks = {
        onVisualization: jest.fn(),
        onComplete: jest.fn(),
      };

      const request: ChatRequest = {
        message: 'Show chart',
        include_visualization: true,
      };

      await chatApi.streamMessage(request, callbacks);

      expect(callbacks.onVisualization).toHaveBeenCalledWith(mockVisualization);
    });

    it('should handle errors during streaming', async () => {
      const mockEvents = [
        'data: {"type":"start","conversation_id":"conv-789"}\n\n',
        'data: {"type":"error","error":"Something went wrong"}\n\n',
      ];

      const mockReadableStream = new ReadableStream({
        start(controller) {
          mockEvents.forEach(event => {
            controller.enqueue(new TextEncoder().encode(event));
          });
          controller.close();
        },
      });

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        body: mockReadableStream,
      });

      const callbacks = {
        onStart: jest.fn(),
        onError: jest.fn(),
      };

      const request: ChatRequest = {
        message: 'Test error',
      };

      await chatApi.streamMessage(request, callbacks);

      expect(callbacks.onStart).toHaveBeenCalledWith('conv-789');
      expect(callbacks.onError).toHaveBeenCalledWith('Something went wrong');
    });

    it('should handle HTTP errors', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 500,
      });

      const callbacks = {
        onError: jest.fn(),
      };

      const request: ChatRequest = {
        message: 'Test',
      };

      await expect(chatApi.streamMessage(request, callbacks)).rejects.toThrow('HTTP error! status: 500');
    });

    it('should handle network errors', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(new Error('Network failure'));

      const callbacks = {
        onError: jest.fn(),
      };

      const request: ChatRequest = {
        message: 'Test',
      };

      await expect(chatApi.streamMessage(request, callbacks)).rejects.toThrow('Network failure');
      expect(callbacks.onError).toHaveBeenCalledWith('Network failure');
    });

    it('should handle malformed JSON events gracefully', async () => {
      const mockEvents = [
        'data: {"type":"start","conversation_id":"conv-malformed"}\n\n',
        'data: {invalid json}\n\n', // Malformed JSON
        'data: {"type":"token","content":"Still working"}\n\n',
        'data: {"type":"done"}\n\n',
      ];

      const mockReadableStream = new ReadableStream({
        start(controller) {
          mockEvents.forEach(event => {
            controller.enqueue(new TextEncoder().encode(event));
          });
          controller.close();
        },
      });

      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        body: mockReadableStream,
      });

      const callbacks = {
        onStart: jest.fn(),
        onToken: jest.fn(),
        onComplete: jest.fn(),
      };

      const request: ChatRequest = {
        message: 'Test malformed',
      };

      // Should not throw, just log error and continue
      await chatApi.streamMessage(request, callbacks);

      expect(callbacks.onStart).toHaveBeenCalledWith('conv-malformed');
      expect(callbacks.onToken).toHaveBeenCalledWith('Still working', undefined);
      expect(callbacks.onComplete).toHaveBeenCalled();
    });
  });

  describe('sendMessage', () => {
    it('should send message via POST request', async () => {
      mockPost.mockResolvedValue({
        data: {
          message: 'Response message',
          conversation_id: 'conv-post',
        },
      });

      const request: ChatRequest = {
        message: 'Hello',
        conversation_id: 'conv-123',
        include_visualization: true,
      };

      const response = await chatApi.sendMessage(request);

      expect(mockPost).toHaveBeenCalledWith('/chat/', request);
      expect(response.message).toBe('Response message');
      expect(response.conversation_id).toBe('conv-post');
    });
  });

  describe('getHistory', () => {
    it('should fetch conversation history', async () => {
      mockGet.mockResolvedValue({
        data: {
          messages: [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there!' },
          ],
        },
      });

      const result = await chatApi.getHistory('conv-123');

      expect(mockGet).toHaveBeenCalledWith('/chat/history/conv-123');
      expect(result.messages).toHaveLength(2);
    });
  });

  describe('clearHistory', () => {
    it('should clear conversation history', async () => {
      mockDelete.mockResolvedValue({
        data: { success: true },
      });

      const result = await chatApi.clearHistory('conv-123');

      expect(mockDelete).toHaveBeenCalledWith('/chat/history/conv-123');
      expect(result.success).toBe(true);
    });
  });

  describe('listConversations', () => {
    it('should list all conversations', async () => {
      mockGet.mockResolvedValue({
        data: {
          conversations: [
            { id: 'conv-1', title: 'Chat 1' },
            { id: 'conv-2', title: 'Chat 2' },
          ],
        },
      });

      const result = await chatApi.listConversations();

      expect(mockGet).toHaveBeenCalledWith('/chat/conversations');
      expect(result.conversations).toHaveLength(2);
    });
  });
});

describe('configApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGet.mockClear();
  });

  it('should check health status', async () => {
    mockGet.mockResolvedValue({
      data: { status: 'healthy' },
    });

    const result = await configApi.healthCheck();

    expect(mockGet).toHaveBeenCalledWith('/config/health');
    expect(result.status).toBe('healthy');
  });
});

describe('dataApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGet.mockClear();
  });

  describe('listDatasets', () => {
    it('should list all datasets', async () => {
      mockGet.mockResolvedValue({
        data: {
          datasets: ['dataset1', 'dataset2'],
        },
      });

      const result = await dataApi.listDatasets();

      expect(mockGet).toHaveBeenCalledWith('/data/datasets');
      expect(result.datasets).toHaveLength(2);
    });
  });

  describe('getDatasetInfo', () => {
    it('should get dataset information', async () => {
      mockGet.mockResolvedValue({
        data: {
          name: 'dataset1',
          rows: 1000,
          columns: ['col1', 'col2'],
        },
      });

      const result = await dataApi.getDatasetInfo('dataset1');

      expect(mockGet).toHaveBeenCalledWith('/data/datasets/dataset1/info');
      expect(result.name).toBe('dataset1');
      expect(result.rows).toBe(1000);
    });
  });

  describe('queryDataset', () => {
    it('should query dataset with parameters', async () => {
      mockGet.mockResolvedValue({
        data: {
          results: [{ id: 1, value: 'test' }],
        },
      });

      const params = {
        query: 'SELECT * FROM table',
        limit: 10,
      };

      const result = await dataApi.queryDataset('dataset1', params);

      expect(mockGet).toHaveBeenCalledWith('/data/datasets/dataset1/query', { params });
      expect(result.results).toHaveLength(1);
    });

    it('should query dataset without parameters', async () => {
      mockGet.mockResolvedValue({
        data: { results: [] },
      });

      await dataApi.queryDataset('dataset1');

      expect(mockGet).toHaveBeenCalledWith('/data/datasets/dataset1/query', { params: undefined });
    });
  });
});
