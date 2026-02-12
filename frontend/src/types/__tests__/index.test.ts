import type { Message, VisualizationData, ChatRequest, ChatResponse } from '../index';

describe('Type Definitions', () => {
  describe('Message', () => {
    it('should create a valid user message', () => {
      const message: Message = {
        id: 'msg-1',
        role: 'user',
        content: 'Hello',
        timestamp: new Date(),
      };

      expect(message.role).toBe('user');
      expect(message.content).toBe('Hello');
    });

    it('should create a valid assistant message with visualization', () => {
      const visualization: VisualizationData = {
        chart_type: 'bar',
        title: 'Test Chart',
        data: [{ x: '2023', y: 100 }],
      };

      const message: Message = {
        id: 'msg-2',
        role: 'assistant',
        content: 'Here is your chart',
        timestamp: new Date(),
        visualization,
      };

      expect(message.role).toBe('assistant');
      expect(message.visualization).toBeDefined();
      expect(message.visualization?.chart_type).toBe('bar');
    });
  });

  describe('VisualizationData', () => {
    it('should create a bar chart visualization', () => {
      const viz: VisualizationData = {
        chart_type: 'bar',
        title: 'Monthly Sales',
        data: [
          { month: 'Jan', sales: 100 },
          { month: 'Feb', sales: 150 },
        ],
        x_axis: 'month',
        y_axis: 'sales',
        x_label: 'Month',
        y_label: 'Sales ($)',
      };

      expect(viz.chart_type).toBe('bar');
      expect(viz.data).toHaveLength(2);
    });

    it('should create a line chart visualization', () => {
      const viz: VisualizationData = {
        chart_type: 'line',
        title: 'Temperature Trend',
        data: [
          { time: '00:00', temp: 20 },
          { time: '12:00', temp: 25 },
        ],
      };

      expect(viz.chart_type).toBe('line');
    });

    it('should create a pie chart visualization', () => {
      const viz: VisualizationData = {
        chart_type: 'pie',
        title: 'Market Share',
        data: [
          { category: 'A', value: 40 },
          { category: 'B', value: 60 },
        ],
      };

      expect(viz.chart_type).toBe('pie');
    });

    it('should create a scatter chart visualization', () => {
      const viz: VisualizationData = {
        chart_type: 'scatter',
        title: 'Correlation',
        data: [
          { x: 1, y: 2 },
          { x: 2, y: 4 },
        ],
      };

      expect(viz.chart_type).toBe('scatter');
    });

    it('should create a table visualization', () => {
      const viz: VisualizationData = {
        chart_type: 'table',
        title: 'Data Table',
        data: [
          { name: 'Alice', age: 30 },
          { name: 'Bob', age: 25 },
        ],
      };

      expect(viz.chart_type).toBe('table');
    });

    it('should support HTML chart', () => {
      const viz: VisualizationData = {
        chart_type: 'bar',
        title: 'Custom HTML Chart',
        data: [],
        html_chart: '<div>Custom HTML</div>',
      };

      expect(viz.html_chart).toBe('<div>Custom HTML</div>');
    });
  });

  describe('ChatRequest', () => {
    it('should create a basic chat request', () => {
      const request: ChatRequest = {
        message: 'What is the weather?',
      };

      expect(request.message).toBe('What is the weather?');
      expect(request.conversation_id).toBeUndefined();
      expect(request.include_visualization).toBeUndefined();
    });

    it('should create a chat request with conversation ID', () => {
      const request: ChatRequest = {
        message: 'Follow-up question',
        conversation_id: 'conv-123',
      };

      expect(request.conversation_id).toBe('conv-123');
    });

    it('should create a chat request with visualization flag', () => {
      const request: ChatRequest = {
        message: 'Show me a chart',
        include_visualization: true,
      };

      expect(request.include_visualization).toBe(true);
    });
  });

  describe('ChatResponse', () => {
    it('should create a basic chat response', () => {
      const response: ChatResponse = {
        message: 'The weather is sunny',
        conversation_id: 'conv-456',
      };

      expect(response.message).toBe('The weather is sunny');
      expect(response.conversation_id).toBe('conv-456');
    });

    it('should create a chat response with visualization', () => {
      const visualization: VisualizationData = {
        chart_type: 'line',
        title: 'Temperature',
        data: [{ time: '12:00', temp: 25 }],
      };

      const response: ChatResponse = {
        message: 'Here is the temperature chart',
        conversation_id: 'conv-789',
        visualization,
      };

      expect(response.visualization).toBeDefined();
      expect(response.visualization?.chart_type).toBe('line');
    });

    it('should create a chat response with metadata', () => {
      const response: ChatResponse = {
        message: 'Response with metadata',
        conversation_id: 'conv-meta',
        metadata: {
          agent: 'coordinator',
          execution_time: 1.5,
        },
      };

      expect(response.metadata).toBeDefined();
      expect(response.metadata?.agent).toBe('coordinator');
    });

    it('should create a chat response with sources', () => {
      const response: ChatResponse = {
        message: 'Based on the data...',
        conversation_id: 'conv-sources',
        sources: ['dataset1.csv', 'dataset2.xlsx'],
      };

      expect(response.sources).toHaveLength(2);
      expect(response.sources).toContain('dataset1.csv');
    });
  });
});
