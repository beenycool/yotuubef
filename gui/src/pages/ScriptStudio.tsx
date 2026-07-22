import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { api } from '../api/client';
import type { ScriptData } from '../api/types';

import { ScriptEditor } from '../components/ScriptEditor';

export const ScriptStudio: React.FC = () => {
  const { name } = useParams<{ name: string }>();
  const projectName = name || '';
  const navigate = useNavigate();

  const [script, setScript] = useState<ScriptData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectName) return;
    setLoading(true);
    api.getScript(projectName)
      .then(setScript)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [projectName]);

  const handleSave = async (updatedScript: ScriptData) => {
    await api.updateScript(projectName, updatedScript);
    setScript(updatedScript);
  };

  const handleSaveAndRender = async (updatedScript: ScriptData) => {
    await api.updateScript(projectName, updatedScript);
    await api.resumePipeline(projectName, { phase_override: 'VIDEO_RENDER', no_upload: true });
    navigate(`/project/${projectName}`);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <Link to={`/project/${projectName}`} style={{ textDecoration: 'none', color: 'var(--text-secondary)' }}>
          ← Back to Project
        </Link>
        <h1 style={{ fontSize: '1.6rem', fontWeight: 700 }}>Director's Chair: {projectName}</h1>
      </div>

      {loading ? (
        <div style={{ color: 'var(--text-muted)' }}>Loading script data...</div>
      ) : error ? (
        <div className="glass-card" style={{ padding: '32px', textAlign: 'center', color: 'var(--accent-danger)' }}>
          <h3>Script Not Ready</h3>
          <p style={{ marginTop: '8px', color: 'var(--text-secondary)' }}>{error}</p>
          <div style={{ marginTop: '16px' }}>
            <Link to={`/project/${projectName}`} className="btn btn-primary">
              Run SCRIPTING Phase
            </Link>
          </div>
        </div>
      ) : script ? (
        <ScriptEditor
          initialScript={script}
          onSave={handleSave}
          onSaveAndRender={handleSaveAndRender}
        />
      ) : null}
    </div>
  );
};
