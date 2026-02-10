export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  visualization?: VisualizationData;
}

export interface VisualizationData {
  chart_type: 'bar' | 'line' | 'pie' | 'scatter' | 'table';
  title: string;
  data: Record<string, unknown>[];
  x_axis?: string;
  y_axis?: string;
  description?: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  include_visualization?: boolean;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  visualization?: VisualizationData;
  sources?: string[];
  metadata?: Record<string, unknown>;
}

