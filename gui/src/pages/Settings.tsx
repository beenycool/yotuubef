import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ConfigData, PreflightResult } from '../api/types';

import { ConfigForm } from '../components/ConfigForm';

export const Settings: React.FC = () => {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAll();
  }, []);

  const loadAll = async () => {
    setLoading(true);
    try {
      const cfg = await api.getConfig();
      setConfig(cfg);
      const pf = await api.runPreflight();
      setPreflight(pf);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveYaml = async (yamlContent: string) => {
    await api.updateConfigYaml(yamlContent);
    await loadAll();
  };

  const handleSaveEnv = async (envVars: Record<string, string>) => {
    await api.updateEnvVars(envVars);
    await loadAll();
  };

  const handleRunPreflight = async () => {
    const pf = await api.runPreflight();
    setPreflight(pf);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '1.8rem', fontWeight: 700 }}>Studio Configuration & Diagnostics</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          Tune application parameters, manage API keys, and run preflight checks
        </p>
      </div>

      {loading || !config ? (
        <div style={{ color: 'var(--text-muted)' }}>Loading configuration...</div>
      ) : (
        <ConfigForm
          config={config}
          preflight={preflight}
          onSaveYaml={handleSaveYaml}
          onSaveEnv={handleSaveEnv}
          onRunPreflight={handleRunPreflight}
        />
      )}
    </div>
  );
};
