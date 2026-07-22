import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Project } from '../api/types';


export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProjects = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.listProjects();
      setProjects(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const createProject = async (name: string, reddit_url?: string) => {
    const newProj = await api.createProject(name, reddit_url);
    await fetchProjects();
    return newProj;
  };

  const deleteProject = async (name: string) => {
    await api.deleteProject(name);
    await fetchProjects();
  };

  return { projects, loading, error, refresh: fetchProjects, createProject, deleteProject };
}
