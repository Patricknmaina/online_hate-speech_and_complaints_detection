// Prefer environment-provided API base URL, fallback to localhost for dev
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export interface TweetRequest {
  text: string;
  user_id?: string;
}

export interface TweetResponse {
  text: string;
  prediction: string;
  confidence: number;
  probabilities: { [key: string]: number };
  user_id?: string;
  model_used?: string;
}

export interface HealthResponse {
  status: string;
  message: string;
  model_info: {
    sklearn_loaded: boolean;
    openai_available: boolean;
    openai_model: string | null;
    sklearn_model_type: string | null;
    memory_usage: string;
    available_memory_mb: number;
  };
  system_info: {
    memory_usage_percent: number;
    available_memory_gb: number;
    cpu_usage_percent: number;
    timestamp: string;
  };
}

export interface ModelInfo {
  model_info: {
    sklearn_loaded: boolean;
    openai_available: boolean;
    openai_model: string | null;
    sklearn_model_type: string | null;
  };
  config: {
    openai_model: string;
    use_sklearn_only: boolean;
    categories: string[];
  };
}

// Enhanced response wrapper for better error handling
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  status?: number;
}

// Chatbot interfaces
export interface ChatRequest {
  message: string;
  sender_id?: string;
  conversation_history?: Array<{ role: string; content: string }>;
}

export interface ChatMessage {
  text: string;
  image?: string;
  buttons?: Array<{
    title: string;
    payload: string;
  }>;
}

export interface ChatResponse {
  responses: ChatMessage[];
  sender_id: string;
  timestamp: string;
  model_used?: string;
}

export interface ChatStatus {
  openai_available: boolean;
  model: string;
  fallback_mode: boolean;
  status: string;
}

// Model types for the frontend
export type ModelType = 'OpenAI' | 'Sklearn';

// API functions
export const checkApiHealth = async (): Promise<ApiResponse<HealthResponse>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        data,
        status: response.status
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `API returned status ${response.status}: ${errorText}`,
        status: response.status
      };
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to connect to FastAPI backend'
    };
  }
};

export const predictTweet = async (
  request: TweetRequest,
  modelType: ModelType = 'OpenAI'
): Promise<ApiResponse<TweetResponse>> => {
  try {
    // Map model type to endpoint
    const endpoint = modelType === 'OpenAI' ? '/predict/openai' : '/predict';
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        data,
        status: response.status
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `Prediction failed (${response.status}): ${errorText}`,
        status: response.status
      };
    }
  } catch (error) {
    console.error('Error predicting tweet:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error - ensure FastAPI backend is running'
    };
  }
};

export const predictBatchTweets = async (
  tweets: TweetRequest[],
  modelType: ModelType = 'OpenAI'
): Promise<ApiResponse<{ predictions: TweetResponse[] }>> => {
  try {
    // Map model type to endpoint
    const endpoint = modelType === 'OpenAI' ? '/predict/openai/batch' : '/predict/batch';
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(tweets),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        data,
        status: response.status
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `Batch prediction failed (${response.status}): ${errorText}`,
        status: response.status
      };
    }
  } catch (error) {
    console.error('Error predicting batch tweets:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error - ensure FastAPI backend is running'
    };
  }
};

export const getModelInfo = async (): Promise<ApiResponse<ModelInfo>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/model/info`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        data,
        status: response.status
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `Failed to get model info (${response.status}): ${errorText}`,
        status: response.status
      };
    }
  } catch (error) {
    console.error('Error getting model info:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error - ensure FastAPI backend is running'
    };
  }
};

// Chatbot API functions
export const sendChatMessage = async (request: ChatRequest): Promise<ApiResponse<ChatResponse>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (response.ok) {
      const data: ChatResponse = await response.json();
      return {
        success: true,
        data,
        status: response.status
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `Chat request failed (${response.status}): ${errorText}`,
        status: response.status
      };
    }
  } catch (error) {
    console.error('Error sending chat message:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error - ensure FastAPI backend is running'
    };
  }
};

export const getChatStatus = async (): Promise<ApiResponse<ChatStatus>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/status`);

    if (response.ok) {
      const data: ChatStatus = await response.json();
      return {
        success: true,
        data,
        status: response.status
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        error: `Failed to get chat status (${response.status}): ${errorText}`,
        status: response.status
      };
    }
  } catch (error) {
    console.error('Error getting chat status:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error - ensure FastAPI backend is running'
    };
  }
};
