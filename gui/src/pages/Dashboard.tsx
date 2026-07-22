import React, { useState, useEffect } from 'react';
import { useProjects } from '../hooks/useProjects';
import { ProjectCard } from '../components/ProjectCard';
import { useToast } from '../components/Toast';

export const Dashboard: React.FC = () => {
  const { projects, loading, error, createProject, deleteProject } = useProjects();
  const toast = useToast();
  const [showModal, setShowModal] = useState(false);
  const [newProjName, setNewProjName] = useState('');
  const [newRedditUrl, setNewRedditUrl] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showModal) {
        setShowModal(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showModal]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjName.trim()) return;
    setCreating(true);
    try {
      await createProject(newProjName.trim(), newRedditUrl.trim() || undefined);
      toast.success(`Project '${newProjName.trim()}' created!`);
      setNewProjName('');
      setNewRedditUrl('');
      setShowModal(false);
    } catch (err: any) {
      toast.error(err.message, 'Failed to Create Project');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, letterSpacing: '-0.5px' }}>Project Studio Dashboard</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            Manage and run documentary YouTube Shorts workflows
          </p>
        </div>

        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
          ➕ New Documentary Project
        </button>
      </div>

      {/* Stats Summary Bar */}
      <div className="dashboard-stats" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Total Projects</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-primary)' }}>{projects.length}</div>
        </div>
        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Completed Videos</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--accent-success)' }}>
            {projects.filter((p) => p.has_video).length}
          </div>
        </div>
        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Ready for Script Review</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--phase-scripting)' }}>
            {projects.filter((p) => p.status === 'paused_for_script_review' || p.current_phase === 'SCRIPTING').length}
          </div>
        </div>
        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Active / In Progress</span>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--accent-secondary)' }}>
            {projects.filter((p) => p.status === 'active').length}
          </div>
        </div>
      </div>

      {/* Projects Grid or Onboarding Empty State */}
      {loading ? (
        <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px', padding: '40px' }}>
          <span className="spinner" /> Loading project studio workspaces...
        </div>
      ) : error ? (
        <div className="glass-card" style={{ padding: '24px', borderColor: 'var(--accent-danger)', color: 'var(--accent-danger)' }}>
          ⚠️ Error loading projects: {error}
        </div>
      ) : projects.length === 0 ? (
        <div
          className="glass-card"
          style={{
            padding: '48px 32px',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '20px',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(108,92,231,0.05) 100%)',
          }}
        >
          <div
            style={{
              fontSize: '48px',
              width: '80px',
              height: '80px',
              borderRadius: '20px',
              background: 'var(--accent-gradient)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 8px 32px rgba(108, 92, 231, 0.4)',
            }}
          >
            🎬
          </div>

          <div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '8px' }}>
              Create Your First Documentary
            </h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '540px', margin: '0 auto', fontSize: '0.95rem' }}>
              Yotuubef Studio automates deep research, story synthesis, script narration, evidence gathering, and video editing for viral YouTube Shorts.
            </p>
          </div>

          <button className="btn btn-primary" style={{ padding: '12px 24px', fontSize: '1rem' }} onClick={() => setShowModal(true)}>
            🚀 Start a New Project
          </button>

          {/* Quick 3-Step Guide */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px',
              width: '100%',
              maxWidth: '720px',
              marginTop: '16px',
              textAlign: 'left',
            }}
          >
            <div style={{ padding: '16px', borderRadius: 'var(--radius-sm)', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--phase-idea)', fontWeight: 700, fontSize: '0.85rem' }}>1. IDEA & SEED</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>Provide a topic name or optional Reddit link as initial seed.</div>
            </div>

            <div style={{ padding: '16px', borderRadius: 'var(--radius-sm)', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--phase-research)', fontWeight: 700, fontSize: '0.85rem' }}>2. RESEARCH & SCRIPT</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>Deep research runs automatically or accepts custom report input.</div>
            </div>

            <div style={{ padding: '16px', borderRadius: 'var(--radius-sm)', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ color: 'var(--phase-render)', fontWeight: 700, fontSize: '0.85rem' }}>3. EDIT & RENDER</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>Tweak cues in Director's Chair and render high-impact vertical video.</div>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
          {projects.map((proj) => (
            <ProjectCard key={proj.name} project={proj} onDelete={deleteProject} />
          ))}
        </div>
      )}

      {/* New Project Modal */}
      {showModal && (
        <div
          onClick={() => setShowModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.75)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            animation: 'fadeIn 0.15s ease',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="glass-card"
            style={{ width: '460px', maxWidth: '90vw', padding: '32px', background: 'var(--bg-secondary)' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 700 }}>Create New Project</h2>
              <button
                onClick={() => setShowModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Project Name *
                </label>
                <input
                  type="text"
                  required
                  autoFocus
                  placeholder="e.g. dream_speedrun_scandal"
                  value={newProjName}
                  onChange={(e) => setNewProjName(e.target.value)}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Optional Reddit URL Seed
                </label>
                <input
                  type="url"
                  placeholder="https://reddit.com/r/speedrun/comments/..."
                  value={newRedditUrl}
                  onChange={(e) => setNewRedditUrl(e.target.value)}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '12px' }}>
                <button type="button" className="btn btn-secondary" onClick={() => setShowModal(false)} disabled={creating}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={creating || !newProjName.trim()}>
                  {creating ? <><span className="spinner" /> Creating...</> : 'Create Project'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
