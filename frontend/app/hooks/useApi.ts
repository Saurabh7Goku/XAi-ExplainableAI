import { useState, useEffect, useCallback } from 'react';
import { apiClient, PredictionResponse, SystemHealth, PredictionHistory } from '../lib/api';

// Generic hook for API calls with loading and error states
export function useApiCall<T, P extends any[] = []>(
  apiFunction: (...args: P) => Promise<T>
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async (...args: P) => {
      setLoading(true);
      setError(null);

      try {
        const result = await apiFunction(...args);
        setData(result);
        return result;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An error occurred';
        setError(errorMessage);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [apiFunction]
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

// Hook for prediction
export function usePrediction() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);

    try {
      await apiClient.predictDiseaseStream(file, (update) => {
        if (update.type === 'prediction') {
          setPrediction({
            success: true,
            prediction: update.prediction,
            confidence: update.confidence,
            class_probabilities: update.class_probabilities,
            disease_info: update.disease_info,
            lime_explanation: '',
            report: 'Generating expert report and diagnosis...',
            processing_time_ms: 0,
            prediction_id: update.prediction_id
          });
          // Stop global loading so result components can show up
          setLoading(false);
        } else if (update.type === 'report') {
          setPrediction(prev => prev ? { ...prev, report: update.report } : null);
        } else if (update.type === 'explanation') {
          setPrediction(prev => prev ? {
            ...prev,
            lime_explanation: update.lime_explanation,
            processing_time_ms: update.processing_time_ms
          } : null);
        } else if (update.type === 'error') {
          setError(update.message);
          setLoading(false);
        }
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Prediction failed';
      setError(errorMessage);
      setLoading(false);
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setPrediction(null);
    setError(null);
    setLoading(false);
  }, []);

  return { prediction, loading, error, predict, reset };
}

// Hook for health check
export function useHealthCheck(interval: number = 30000) {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const result = await apiClient.getHealthCheck();
      setHealth(result);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Health check failed';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();

    if (interval > 0) {
      const intervalId = setInterval(checkHealth, interval);
      return () => clearInterval(intervalId);
    }
  }, [checkHealth, interval]);

  return { health, loading, error, refetch: checkHealth };
}

// Hook for prediction history
export function usePredictionHistory(limit: number = 50) {
  const [history, setHistory] = useState<PredictionHistory[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.getPredictionHistory(limit);
      setHistory(result.predictions);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch history';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return { history, loading, error, refetch: fetchHistory };
}

// Hook for system info
export function useSystemInfo() {
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSystemInfo = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.getSystemInfo();
      setSystemInfo(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch system info';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSystemInfo();
  }, [fetchSystemInfo]);

  return { systemInfo, loading, error, refetch: fetchSystemInfo };
}

// Hook for model info
export function useModelInfo() {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModelInfo = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.getModelInfo();
      setModelInfo(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch model info';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModelInfo();
  }, [fetchModelInfo]);

  return { modelInfo, loading, error, refetch: fetchModelInfo };
}
