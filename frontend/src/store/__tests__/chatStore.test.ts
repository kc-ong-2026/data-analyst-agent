import { renderHook, act, waitFor } from '@testing-library/react';
import { useChatStore } from '../chatStore';
import { chatApi } from '../../api/client';

// Mock the API client
jest.mock('../../api/client', () => ({
  chatApi: {
    sendMessage: jest.fn(),
    streamMessage: jest.fn(),
    clearHistory: jest.fn().mockResolvedValue({ success: true }),
  },
}));

describe('chatStore', () => {
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();

    // Reset the store state manually
    act(() => {
      useChatStore.setState({
        messages: [],
        conversationId: null,
        isLoading: false,
        isStreaming: false,
        error: null,
        currentVisualization: null,
        currentAgent: null,
        streamingMessage: '',
        useStreaming: true,
      });
    });
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const { result } = renderHook(() => useChatStore());

      expect(result.current.messages).toEqual([]);
      expect(result.current.conversationId).toBeNull();
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.error).toBeNull();
      expect(result.current.currentVisualization).toBeNull();
      expect(result.current.currentAgent).toBeNull();
      expect(result.current.streamingMessage).toBe('');
      expect(result.current.useStreaming).toBe(true);
    });
  });

  describe('sendMessage (non-streaming)', () => {
    it('should send message and update state correctly', async () => {
      const mockResponse = {
        message: 'This is a test response',
        conversation_id: 'conv-123',
        visualization: undefined,
      };

      (chatApi.sendMessage as jest.Mock).mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useChatStore());

      // Disable streaming for this test
      act(() => {
        result.current.toggleStreaming();
      });

      expect(result.current.useStreaming).toBe(false);

      // Send a message
      await act(async () => {
        await result.current.sendMessage('Hello');
      });

      // Verify user message was added
      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].role).toBe('user');
      expect(result.current.messages[0].content).toBe('Hello');

      // Verify assistant message was added
      expect(result.current.messages[1].role).toBe('assistant');
      expect(result.current.messages[1].content).toBe('This is a test response');

      // Verify conversation ID was set
      expect(result.current.conversationId).toBe('conv-123');

      // Verify loading state is false
      expect(result.current.isLoading).toBe(false);
    });

    it('should handle errors correctly', async () => {
      (chatApi.sendMessage as jest.Mock).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useChatStore());

      // Disable streaming
      act(() => {
        result.current.toggleStreaming();
      });

      // Send a message that will fail
      await act(async () => {
        await result.current.sendMessage('Hello');
      });

      // Verify error state
      expect(result.current.error).toBe('Network error');
      expect(result.current.isLoading).toBe(false);

      // User message should still be added
      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0].role).toBe('user');
    });
  });

  describe('sendMessage (streaming)', () => {
    it('should handle streaming message correctly', async () => {
      (chatApi.streamMessage as jest.Mock).mockImplementation(async (_request, callbacks) => {
        // Simulate streaming events
        callbacks.onStart('conv-456');
        callbacks.onAgent('coordinator', 'running', 'Processing query');
        callbacks.onToken('Hello');
        callbacks.onToken(' from');
        callbacks.onToken(' streaming!');
        callbacks.onComplete('Hello from streaming!');
      });

      const { result } = renderHook(() => useChatStore());

      expect(result.current.useStreaming).toBe(true);

      // Send a message with streaming
      await act(async () => {
        await result.current.sendMessage('Test streaming');
      });

      // Verify user message
      expect(result.current.messages[0].role).toBe('user');
      expect(result.current.messages[0].content).toBe('Test streaming');

      // Verify assistant message was built progressively
      await waitFor(() => {
        expect(result.current.messages).toHaveLength(2);
      });

      expect(result.current.messages[1].role).toBe('assistant');
      expect(result.current.messages[1].content).toBe('Hello from streaming!');

      // Verify conversation ID
      expect(result.current.conversationId).toBe('conv-456');

      // Verify final state
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.streamingMessage).toBe('');
    });

    it('should handle streaming errors', async () => {
      (chatApi.streamMessage as jest.Mock).mockImplementation(async (_request, callbacks) => {
        callbacks.onStart('conv-789');
        callbacks.onError('Streaming failed');
      });

      const { result } = renderHook(() => useChatStore());

      await act(async () => {
        await result.current.sendMessage('Test error');
      });

      // Verify error state
      expect(result.current.error).toBe('Streaming failed');
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isStreaming).toBe(false);
    });

    it('should handle visualization in streaming', async () => {
      const mockVisualization = {
        chart_type: 'bar' as const,
        title: 'Test Chart',
        data: [{ x: '2023', y: 100 }],
        x_axis: 'x',
        y_axis: 'y',
      };

      (chatApi.streamMessage as jest.Mock).mockImplementation(async (_request, callbacks) => {
        callbacks.onStart('conv-viz');
        callbacks.onToken('Chart data:');
        callbacks.onVisualization(mockVisualization);
        callbacks.onComplete('Chart data:');
      });

      const { result } = renderHook(() => useChatStore());

      await act(async () => {
        await result.current.sendMessage('Show chart');
      });

      // Verify visualization was set
      await waitFor(() => {
        expect(result.current.currentVisualization).toEqual(mockVisualization);
      });

      // Verify message has visualization
      const assistantMessage = result.current.messages.find(m => m.role === 'assistant');
      expect(assistantMessage?.visualization).toEqual(mockVisualization);
    });
  });

  describe('clearMessages', () => {
    it('should clear all messages and reset state', async () => {
      (chatApi.sendMessage as jest.Mock).mockResolvedValue({
        message: 'Response',
        conversation_id: 'conv-clear',
      });

      const { result } = renderHook(() => useChatStore());

      // Disable streaming and send a message
      act(() => {
        result.current.toggleStreaming();
      });

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.conversationId).toBe('conv-clear');

      // Clear messages
      act(() => {
        result.current.clearMessages();
      });

      // Verify state is reset
      expect(result.current.messages).toEqual([]);
      expect(result.current.conversationId).toBeNull();
      expect(result.current.currentVisualization).toBeNull();
      expect(result.current.error).toBeNull();
      expect(result.current.streamingMessage).toBe('');
      expect(result.current.currentAgent).toBeNull();
    });
  });

  describe('setVisualization', () => {
    it('should set visualization data', () => {
      const mockVisualization = {
        chart_type: 'line' as const,
        title: 'Line Chart',
        data: [{ month: 'Jan', value: 100 }],
      };

      const { result } = renderHook(() => useChatStore());

      act(() => {
        result.current.setVisualization(mockVisualization);
      });

      expect(result.current.currentVisualization).toEqual(mockVisualization);
    });

    it('should clear visualization when null', () => {
      const mockVisualization = {
        chart_type: 'pie' as const,
        title: 'Pie Chart',
        data: [{ category: 'A', value: 50 }],
      };

      const { result } = renderHook(() => useChatStore());

      act(() => {
        result.current.setVisualization(mockVisualization);
      });

      expect(result.current.currentVisualization).toEqual(mockVisualization);

      act(() => {
        result.current.setVisualization(null);
      });

      expect(result.current.currentVisualization).toBeNull();
    });
  });

  describe('toggleStreaming', () => {
    it('should toggle streaming mode', () => {
      const { result } = renderHook(() => useChatStore());

      expect(result.current.useStreaming).toBe(true);

      act(() => {
        result.current.toggleStreaming();
      });

      expect(result.current.useStreaming).toBe(false);

      act(() => {
        result.current.toggleStreaming();
      });

      expect(result.current.useStreaming).toBe(true);
    });
  });
});
