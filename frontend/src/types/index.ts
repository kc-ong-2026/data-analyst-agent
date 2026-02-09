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
  llm_provider?: string;
  llm_model?: string;
  include_visualization?: boolean;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  visualization?: VisualizationData;
  sources?: string[];
  metadata?: Record<string, unknown>;
}

export interface ModelConfig {
  provider: string;
  model: string;
  available_models: string[];
  temperature: number;
  max_tokens: number;
}

export interface ConfigResponse {
  llm_providers: string[];
  embedding_providers: string[];
  current_llm: ModelConfig;
  current_embedding: {
    provider: string;
    model: string;
    dimensions: number;
  };
}

export interface ProvidersResponse {
  llm_providers: Record<string, {
    models: string[];
    default_model: string;
    has_api_key: boolean;
  }>;
  embedding_providers: Record<string, {
    models: string[];
    default_model: string;
    has_api_key: boolean;
  }>;
}
