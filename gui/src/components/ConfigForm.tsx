import React, { useState } from 'react';
import type { ConfigData, PreflightResult } from '../api/types';
import { useToast } from './Toast';

interface ConfigFormProps {
  config: ConfigData;
  preflight: PreflightResult | null;
  onSaveYaml: (yamlContent: string) => Promise<void>;
  onSaveEnv: (envVars: Record<string, string>) => Promise<void>;
  onRunPreflight: () => Promise<void>;
}

export const ConfigForm: React.FC<ConfigFormProps> = ({
  config,
  preflight,
  onSaveYaml,
  onSaveEnv,
  onRunPreflight,
}) => {
  const toast = useToast();
  const [activeTab, setActiveTab] = useState<'env' | 'yaml' | 'preflight'>('env');

  const [yamlContent, setYamlContent] = useState(config.yaml_content);
  const [envVars, setEnvVars] = useState<Record<string, string>>(config.env_vars);
  const [showMasked, setShowMasked] = useState<Record<string, boolean>>({});
  const [newKey, setNewKey] = useState('');
  const [newVal, setNewVal] = useState('');

  const [savingYaml, setSavingYaml] = useState(false);
  const [savingEnv, setSavingEnv] = useState(false);
  const [runningPreflight, setRunningPreflight] = useState(false);

  const handleSaveYaml = async () => {
    setSavingYaml(true);
    try {
      await onSaveYaml(yamlContent);
      toast.success('config.yaml saved successfully!');
    } catch (err: any) {
      toast.error(err.message, 'YAML Save Error');
    } finally {
      setSavingYaml(false);
    }
  };

  const handleSaveEnv = async () => {
    setSavingEnv(true);
    try {
      await onSaveEnv(envVars);
      toast.success('.env variables saved successfully!');
    } catch (err: any) {
      toast.error(err.message, 'Environment Save Error');
    } finally {
      setSavingEnv(false);
    }
  };

  const handlePreflight = async () => {
    setRunningPreflight(true);
    try {
      await onRunPreflight();
      toast.success('Preflight diagnostics complete');
    } catch (err: any) {
      toast.error(err.message, 'Preflight Check Error');
    } finally {
      setRunningPreflight(false);
    }
  };

  const handleAddEnvVar = () => {
    if (!newKey.trim()) return;
    const key = newKey.trim().toUpperCase();
    setEnvVars((prev) => ({ ...prev, [key]: newVal.trim() }));
    setNewKey('');
    setNewVal('');
    toast.info(`Added variable ${key}`);
  };

  const handleRemoveEnvVar = (key: string) => {
    setEnvVars((prev) => {
      const copy = { ...prev };
      delete copy[key];
      return copy;
    });
  };

  const toggleMask = (key: string) => {
    setShowMasked((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Tabs bar */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
        <button
          className={`btn ${activeTab === 'env' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('env')}
        >
          🔑 Environment & API Keys
        </button>
        <button
          className={`btn ${activeTab === 'yaml' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('yaml')}
        >
          ⚙️ YAML Configuration
        </button>
        <button
          className={`btn ${activeTab === 'preflight' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('preflight')}
        >
          🔍 System Diagnostics ({preflight ? (preflight.all_passed ? '✅ Passed' : '❌ Issues') : 'Run Check'})
        </button>
      </div>

      {/* Tab 1: Environment Variables */}
      {activeTab === 'env' && (
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>🔑 Environment Variables (.env)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Configure API keys for Gemini, Reddit, OpenAI, ElevenLabs, etc.
              </p>
            </div>
            <button className="btn btn-primary" onClick={handleSaveEnv} disabled={savingEnv}>
              {savingEnv ? <><span className="spinner" /> Saving...</> : '💾 Save Environment'}
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {Object.entries(envVars).map(([key, val]) => {
              const isSensitive = key.includes('KEY') || key.includes('SECRET') || key.includes('PASSWORD') || key.includes('TOKEN');
              const isVisible = showMasked[key];

              return (
                <div
                  key={key}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '220px 1fr 40px 40px',
                    gap: '12px',
                    alignItems: 'center',
                  }}
                >
                  <label style={{ fontSize: '0.85rem', fontWeight: 600, fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {key}
                  </label>

                  <input
                    type={isSensitive && !isVisible ? 'password' : 'text'}
                    value={val}
                    onChange={(e) => setEnvVars({ ...envVars, [key]: e.target.value })}
                  />

                  {isSensitive ? (
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '8px', justifyContent: 'center' }}
                      title={isVisible ? 'Hide key' : 'Show key'}
                      onClick={() => toggleMask(key)}
                    >
                      {isVisible ? '🙈' : '👁️'}
                    </button>
                  ) : (
                    <div />
                  )}

                  <button
                    className="btn btn-danger"
                    style={{ padding: '8px', justifyContent: 'center' }}
                    title="Remove variable"
                    onClick={() => handleRemoveEnvVar(key)}
                  >
                    🗑️
                  </button>
                </div>
              );
            })}
          </div>

          {/* Add New Variable Box */}
          <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
            <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '12px' }}>➕ Add New Environment Variable</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr 120px', gap: '12px', alignItems: 'center' }}>
              <input
                type="text"
                placeholder="VARIABLE_NAME"
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                style={{ fontFamily: 'var(--font-mono)' }}
              />
              <input
                type="text"
                placeholder="Value..."
                value={newVal}
                onChange={(e) => setNewVal(e.target.value)}
              />
              <button className="btn btn-secondary" onClick={handleAddEnvVar} disabled={!newKey.trim()}>
                Add Variable
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: YAML Config */}
      {activeTab === 'yaml' && (
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>⚙️ Configuration YAML (config.yaml)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Fine-tune processing limits, model selection, timing parameters, and directory paths.
              </p>
            </div>
            <button className="btn btn-primary" onClick={handleSaveYaml} disabled={savingYaml}>
              {savingYaml ? <><span className="spinner" /> Saving...</> : '💾 Save YAML'}
            </button>
          </div>

          <textarea
            rows={18}
            value={yamlContent}
            onChange={(e) => setYamlContent(e.target.value)}
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', lineHeight: '1.6' }}
          />
        </div>
      )}

      {/* Tab 3: System Diagnostics */}
      {activeTab === 'preflight' && (
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>🔍 System Diagnostics & Preflight Check</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Validates API key connectivity, background video library, and system dependencies.
              </p>
            </div>

            <button className="btn btn-primary" onClick={handlePreflight} disabled={runningPreflight}>
              {runningPreflight ? <><span className="spinner" /> Checking...</> : 'Run Preflight Check'}
            </button>
          </div>

          {preflight ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '16px' }}>
              {preflight.checks.map((check, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 16px',
                    borderRadius: 'var(--radius-sm)',
                    background: check.passed ? 'rgba(0, 210, 160, 0.08)' : 'rgba(255, 71, 87, 0.08)',
                    border: `1px solid ${check.passed ? 'rgba(0, 210, 160, 0.25)' : 'rgba(255, 71, 87, 0.25)'}`,
                    fontSize: '0.85rem',
                  }}
                >
                  <span style={{ fontSize: '1.1rem' }}>{check.passed ? '✅' : '❌'}</span>
                  <span style={{ fontWeight: 600, width: '200px' }}>{check.name}:</span>
                  <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{check.message}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
              Click "Run Preflight Check" to scan environment status.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
