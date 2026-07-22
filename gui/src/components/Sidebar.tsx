import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { api } from '../api/client';
import type { Project } from '../api/types';

export const Sidebar: React.FC = () => {
  const [recentProjects, setRecentProjects] = useState<Project[]>([]);
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    // Fetch recent projects list for sidebar quick nav
    api.listProjects()
      .then((projs) => setRecentProjects(projs.slice(0, 5)))
      .catch(() => setRecentProjects([]));

    // Health ping
    fetch('/api/health')
      .then((res) => setHealthy(res.ok))
      .catch(() => setHealthy(false));
  }, []);

  const getPhaseDot = (phase: string) => {
    switch (phase) {
      case 'IDEA_GENERATION': return '🟡';
      case 'WAIT_FOR_GEMINI_REPORT': return '🔵';
      case 'SYNTHESIS': return '🟣';
      case 'EVIDENCE_GATHERING': return '🔴';
      case 'SCRIPTING': return '🟤';
      case 'VIDEO_RENDER': return '🟢';
      default: return '⚪';
    }
  };

  return (
    <aside
      className="sidebar-aside"
      style={{
        width: '240px',
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 16px',
        gap: '24px',
        flexShrink: 0,
      }}
    >
      {/* Brand Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'var(--accent-gradient)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '18px',
            fontWeight: 'bold',
            boxShadow: '0 4px 14px rgba(108, 92, 231, 0.4)',
          }}
        >
          🎬
        </div>
        <div>
          <h1 style={{ fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.5px' }}>Yotuubef</h1>
          <span style={{ fontSize: '0.75rem', color: 'var(--accent-secondary)' }}>Studio v1.0</span>
        </div>
      </div>

      {/* Primary Nav */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <NavLink
          to="/"
          className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
          style={{ justifyContent: 'flex-start' }}
        >
          📊 Dashboard
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) => `btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
          style={{ justifyContent: 'flex-start' }}
        >
          ⚙️ Settings
        </NavLink>
      </nav>

      {/* Recent Projects List */}
      {recentProjects.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Recent Projects
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {recentProjects.map((p) => (
              <NavLink
                key={p.name}
                to={`/project/${p.name}`}
                className={({ isActive }) => (isActive ? 'sidebar-proj-link active' : 'sidebar-proj-link')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 10px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.825rem',
                  color: 'var(--text-primary)',
                  textDecoration: 'none',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  transition: 'background 0.2s ease',
                }}
              >
                <span style={{ fontSize: '0.65rem' }}>{getPhaseDot(p.current_phase)}</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{p.name}</span>
              </NavLink>
            ))}
          </div>
        </div>
      )}

      {/* System Status Footer */}
      <div
        style={{
          marginTop: 'auto',
          padding: '12px 14px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
        }}
      >
        <div style={{ color: healthy ? 'var(--accent-success)' : healthy === false ? 'var(--accent-danger)' : 'var(--accent-warning)', fontWeight: 600, marginBottom: '4px' }}>
          ● {healthy ? 'Studio API Connected' : healthy === false ? 'API Offline' : 'Checking Server...'}
        </div>
        FastAPI + Vite connected
      </div>
    </aside>
  );
};
