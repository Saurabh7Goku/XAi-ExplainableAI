/**
 * API client for Mango Leaf Disease Detection System
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PredictionResponse {
  success: boolean;
  prediction: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  lime_explanation: string;
  disease_info: DiseaseInfo;
  report: string;
  processing_time_ms: number;
  prediction_id?: number;
  warning?: string;
}

export interface DiseaseInfo {
  disease_name: string;
  symptoms: string[];
  treatments: string[];
  description?: string;
  severity?: string;
  prevention_methods?: string[];
}

export interface PredictionHistory {
  id: number;
  image_filename: string;
  predicted_class: string;
  confidence: number;
  created_at: string;
  processing_time_ms?: number;
}

export interface PredictionStatistics {
  total_predictions: number;
  class_distribution: Record<string, number>;
  average_confidence: number;
  period_days: number;
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  version: string;
  checks: {
    database: { status: string; message: string };
    model: { status: string; message: string; device?: string };
    filesystem: { status: string; message: string; stats?: any };
    memory: { status: string; message: string; details?: any };
    gpu?: { status: string; message: string; details?: any };
  };
}

export interface TrainingRequest {
  model_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  data_dir: string;
  validation_split?: number;
  save_best_only?: boolean;
  early_stopping_patience?: number;
}

export interface TrainingResponse {
  success: boolean;
  message: string;
  training_run_id?: number;
  error?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  // Prediction endpoints
  async predictDisease(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/predict/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Prediction failed: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async predictDiseaseStream(
    file: File,
    onUpdate: (data: any) => void
  ): Promise<void> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/predict/stream`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Prediction failed: ${response.status} - ${errorText}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) return;

    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          try {
            onUpdate(JSON.parse(line));
          } catch (e) {
            console.error('Failed to parse stream chunk:', e);
          }
        }
      }
    }
  }

  async batchPredict(files: File[]): Promise<any> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const response = await fetch(`${this.baseUrl}/predict/batch`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Batch prediction failed: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async getPredictionHistory(limit: number = 50): Promise<{ success: boolean; predictions: PredictionHistory[] }> {
    return this.request(`/predict/history?limit=${limit}`);
  }

  async getPredictionStatistics(days: number = 30): Promise<{ success: boolean; statistics: PredictionStatistics }> {
    return this.request(`/predict/statistics?days=${days}`);
  }

  // Health endpoints
  async getHealthCheck(): Promise<SystemHealth> {
    return this.request('/health/');
  }

  async getSystemInfo(): Promise<any> {
    return this.request('/health/system');
  }

  async getModelInfo(): Promise<any> {
    return this.request('/health/model');
  }

  // Training endpoints
  async startTraining(trainingRequest: TrainingRequest): Promise<TrainingResponse> {
    return this.request('/training/start', {
      method: 'POST',
      body: JSON.stringify(trainingRequest),
    });
  }

  async getTrainingStatus(runId: number): Promise<any> {
    return this.request(`/training/status/${runId}`);
  }

  async getTrainingRuns(limit: number = 50): Promise<any> {
    return this.request(`/training/runs?limit=${limit}`);
  }

  async getModelVersions(): Promise<any> {
    return this.request('/training/models');
  }

  async activateModel(version: string): Promise<any> {
    return this.request(`/training/models/${version}/activate`, {
      method: 'POST',
    });
  }

  // Utility methods
  async testConnection(): Promise<boolean> {
    try {
      await this.getHealthCheck();
      return true;
    } catch (error) {
      console.error('API connection test failed:', error);
      return false;
    }
  }
}

// Create and export singleton instance
export const apiClient = new ApiClient();

// Export types
export { ApiClient };
