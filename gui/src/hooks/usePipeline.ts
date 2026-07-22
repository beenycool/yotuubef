import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';

export function usePipeline(projectName: string) {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    if (!projectName) return;
    try {
      setError(null);
      const data = await api.getPipelineStatus(projectName);
      setStatus(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch status');
    }
  }, [projectName]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const startPipeline = async (params: { reddit_url?: string; no_upload?: boolean; no_auto_research?: boolean }) => {
    setLoading(true);
    try {
      await api.startPipeline(projectName, params);
      await fetchStatus();
    } finally {
      setLoading(false);
    }
  };

  const resumePipeline = async (params: { phase_override?: string; no_upload?: boolean }) => {
    setLoading(true);
    try {
      await api.resumePipeline(projectName, params);
      await fetchStatus();
    } finally {
      setLoading(false);
    }
  };

  const overridePhase = async (phase: string) => {
    setLoading(true);
    try {
      await api.overridePhase(projectName, phase);
      await fetchStatus();
    } finally {
      setLoading(false);
    }
  };

  const stopPipeline = async () => {
    setLoading(true);
    try {
      await api.stopPipeline(projectName);
      await fetchStatus();
    } finally {
      setLoading(false);
    }
  };

  return {
    status,
    loading,
    error,
    refreshStatus: fetchStatus,
    startPipeline,
    resumePipeline,
    overridePhase,
    stopPipeline,
  };
}
