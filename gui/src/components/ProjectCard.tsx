import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import type { Project } from '../api/types';
import { ConfirmDialog } from './ConfirmDialog';
import { useToast } from './Toast';
import { use3DTilt } from '../hooks/useMotion';

interface ProjectCardProps {
  project: Project;
  onDelete: (name: string) => Promise<void> | void;
}

export const ProjectCard: React.FC<ProjectCardProps> = ({ project, onDelete }) => {
  const [showConfirm, setShowConfirm] = useState(false);
  const toast = useToast();
  const cardRef = use3DTilt<HTMLDivElement>(8);

  const getPhaseBadgeColor = (phase: string) => {
    switch (phase) {
      case 'IDEA_GENERATION': return 'var(--phase-idea)';
      case 'WAIT_FOR_GEMINI_REPORT': return 'var(--phase-research)';
      case 'SYNTHESIS': return 'var(--phase-synthesis)';
      case 'EVIDENCE_GATHERING': return 'var(--phase-evidence)';
      case 'SCRIPTING': return 'var(--phase-scripting)';
      case 'VIDEO_RENDER': return 'var(--phase-render)';
      default: return 'var(--text-muted)';
    }
  };

  const getStatusColor = (status: string) => {
    if (status === 'completed') return 'var(--accent-success)';
    if (status.startsWith('paused')) return 'var(--accent-warning)';
    return 'var(--accent-primary)';
  };

  const handleDelete = async () => {
    try {
      await onDelete(project.name);
      toast.success(`Project '${project.name}' deleted`);
    } catch (err: any) {
      toast.error(`Delete failed: ${err.message}`);
    } finally {
      setShowConfirm(false);
    }
  };

  return (
    <>
      <div
        ref={cardRef}
        className="glass-card interactive-card-spring"
        style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '4px' }}>{project.name}</h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Updated: {new Date(project.updated_at).toLocaleString()}
            </span>
          </div>
          <span
            className="badge badge-pulse"
            style={{
              background: 'rgba(0,0,0,0.3)',
              color: getPhaseBadgeColor(project.current_phase),
              border: `1px solid ${getPhaseBadgeColor(project.current_phase)}`,
            }}
          >
            {project.current_phase}
          </span>
        </div>

        {project.reddit_url && (
          <div style={{ fontSize: '0.8rem', color: 'var(--accent-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            🔗 {project.reddit_url}
          </div>
        )}

        <div style={{ display: 'flex', gap: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <span>Status: <strong style={{ color: getStatusColor(project.status) }}>{project.status}</strong></span>
          <span>•</span>
          <span>Script: {project.has_script ? '✅' : '❌'}</span>
          <span>•</span>
          <span>Video: {project.has_video ? '🎬' : '❌'}</span>
        </div>

        <div style={{ display: 'flex', gap: '10px', marginTop: 'auto', paddingTop: '12px', borderTop: '1px solid var(--border-subtle)' }}>
          <Link to={`/project/${project.name}`} className="btn btn-primary btn-ripple" style={{ flex: 1, justifyContent: 'center' }}>
            Open Studio
          </Link>

          {project.has_script && (
            <Link to={`/project/${project.name}/script`} className="btn btn-secondary btn-ripple" title="Director's Chair">
              ✏️ Edit Script
            </Link>
          )}

          <button
            className="btn btn-danger btn-ripple"
            style={{ padding: '8px 12px' }}
            title="Delete Project"
            onClick={(e) => {
              e.stopPropagation();
              setShowConfirm(true);
            }}
          >
            🗑️
          </button>
        </div>
      </div>

      <ConfirmDialog
        isOpen={showConfirm}
        title="Delete Project"
        message={`Are you sure you want to permanently delete '${project.name}' and all its research artifacts? This action cannot be undone.`}
        confirmLabel="Delete Project"
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setShowConfirm(false)}
      />
    </>
  );
};
