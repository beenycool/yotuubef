import type { Project, ScriptData, ArtifactFile, MediaAsset, ConfigData, PreflightResult } from './types';


const getApiBase = () => {
  if (typeof window === 'undefined') return 'http://localhost:8420/api';
  if (window.location.port === '5173') {
    return `${window.location.protocol}//${window.location.hostname}:8420/api`;
  }
  return `${window.location.origin}/api`;
};

const getMediaBase = () => {
  if (typeof window === 'undefined') return 'http://localhost:8420';
  if (window.location.port === '5173') {
    return `${window.location.protocol}//${window.location.hostname}:8420`;
  }
  return window.location.origin;
};


async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const errorBody = await res.text();
    throw new Error(`HTTP ${res.status}: ${errorBody}`);
  }
  return res.json();
}

export const api = {
  // Projects
  async listProjects(): Promise<Project[]> {
    return fetchJSON<Project[]>(`${getApiBase()}/projects`);
  },

  async getProject(name: string): Promise<any> {
    return fetchJSON<any>(`${getApiBase()}/projects/${name}`);
  },

  async createProject(name: string, reddit_url?: string): Promise<Project> {
    return fetchJSON<Project>(`${getApiBase()}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, reddit_url }),
    });
  },

  async deleteProject(name: string): Promise<void> {
    await fetchJSON(`${getApiBase()}/projects/${name}`, { method: 'DELETE' });
  },

  // Pipeline
  async startPipeline(project: string, params: { reddit_url?: string; no_upload?: boolean; no_auto_research?: boolean }): Promise<any> {
    return fetchJSON(`${getApiBase()}/pipeline/${project}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
  },

  async resumePipeline(project: string, params: { phase_override?: string; no_upload?: boolean }): Promise<any> {
    return fetchJSON(`${getApiBase()}/pipeline/${project}/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
  },

  async overridePhase(project: string, phase: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/pipeline/${project}/phase`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phase }),
    });
  },

  async stopPipeline(project: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/pipeline/${project}/stop`, { method: 'POST' });
  },

  async getPipelineStatus(project: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/pipeline/${project}/status`);
  },

  // Scripts
  async getScript(project: string): Promise<ScriptData> {
    return fetchJSON<ScriptData>(`${getApiBase()}/scripts/${project}`);
  },

  async updateScript(project: string, scriptData: ScriptData): Promise<any> {
    return fetchJSON(`${getApiBase()}/scripts/${project}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scriptData),
    });
  },

  // Artifacts
  async listArtifacts(project: string): Promise<ArtifactFile[]> {
    return fetchJSON<ArtifactFile[]>(`${getApiBase()}/artifacts/${project}`);
  },

  async readArtifactFile(project: string, relativePath: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/artifacts/${project}/file/${relativePath}`);
  },

  async listMediaAssets(project: string): Promise<MediaAsset[]> {
    return fetchJSON<MediaAsset[]>(`${getApiBase()}/artifacts/${project}/media`);
  },

  async submitReport(project: string, content: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/artifacts/${project}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
  },


  // Config
  async getConfig(): Promise<ConfigData> {
    return fetchJSON<ConfigData>(`${getApiBase()}/config`);
  },

  async updateConfigYaml(yamlContent: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml_content: yamlContent }),
    });
  },

  async updateEnvVars(envVars: Record<string, string>): Promise<any> {
    return fetchJSON(`${getApiBase()}/config/env`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ env_vars: envVars }),
    });
  },

  async runPreflight(): Promise<PreflightResult> {
    return fetchJSON<PreflightResult>(`${getApiBase()}/config/preflight`);
  },

  // Videos
  async getRenderedVideo(project: string): Promise<any> {
    return fetchJSON(`${getApiBase()}/videos/${project}`);
  },

  async listBackgroundVideos(): Promise<any[]> {
    return fetchJSON<any[]>(`${getApiBase()}/videos/backgrounds`);
  },
};
export { getMediaBase };

