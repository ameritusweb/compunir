import { useState, useEffect } from 'react';
import useNodeWebSocket from './useNodeWebSocket';

const useMetricsStream = (nodeUrl) => {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const { socket, connected, subscribe } = useNodeWebSocket(nodeUrl);

  useEffect(() => {
    if (!connected) return;

    // Subscribe to different metric channels
    const unsubscribeGPU = subscribe('gpu_metrics', (data) => {
      setMetrics(prev => ({
        ...prev,
        gpu: data
      }));
      
      // Update history
      setHistory(prev => [...prev.slice(-50), {
        timestamp: new Date(),
        ...data
      }]);
    });

    const unsubscribeJobs = subscribe('job_metrics', (data) => {
      setMetrics(prev => ({
        ...prev,
        jobs: data
      }));
    });

    const unsubscribeEarnings = subscribe('earnings_metrics', (data) => {
      setMetrics(prev => ({
        ...prev,
        earnings: data
      }));
    });

    const unsubscribeVerification = subscribe('verification_metrics', (data) => {
      setMetrics(prev => ({
        ...prev,
        verification: data
      }));
    });

    // Error handling
    const unsubscribeErrors = subscribe('errors', (error) => {
      setError(error);
    });

    // Cleanup subscriptions
    return () => {
      unsubscribeGPU();
      unsubscribeJobs();
      unsubscribeEarnings();
      unsubscribeVerification();
      unsubscribeErrors();
    };
  }, [connected, subscribe]);

  // Helper functions for data transformation
  const getMetricsByTimeframe = (timeframe) => {
    const now = new Date();
    const timeframes = {
      '1h': 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000
    };

    return history.filter(metric => 
      (now - new Date(metric.timestamp)) <= timeframes[timeframe]
    );
  };

  const getAggregatedMetrics = (timeframe) => {
    const metrics = getMetricsByTimeframe(timeframe);
    return {
      averageUtilization: metrics.reduce((acc, m) => acc + m.utilization, 0) / metrics.length,
      maxTemperature: Math.max(...metrics.map(m => m.temperature)),
      averageMemoryUsage: metrics.reduce((acc, m) => acc + m.memory.used, 0) / metrics.length
    };
  };

  return {
    metrics,
    history,
    error,
    connected,
    getMetricsByTimeframe,
    getAggregatedMetrics
  };
};

export default useMetricsStream;