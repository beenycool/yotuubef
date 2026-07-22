import React, { useEffect, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { usePipeline } from '../hooks/usePipeline';
import { useWebSocket } from '../hooks/useWebSocket';
import { PhaseStepper } from '../components/PhaseStepper';
import { LogStream } from '../components/LogStream';
import { MediaBrowser } from '../components/MediaBrowser';
import { VideoPlayer } from '../components/VideoPlayer';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { useToast } from '../components/Toast';
import { api } from '../api/client';
import type { ArtifactFile, MediaAsset } from '../api/types';

export const ProjectView: React.FC = () => {
  const { name } = useParams<{ name: string }>();
  const projectName = name || '';
  const toast = useToast();

  const { status, loading, startPipeline, resumePipeline, overridePhase, stopPipeline } = usePipeline(projectName);
  const { logs, clearLogs } = useWebSocket(projectName);

  const [artifacts, setArtifacts] = useState<ArtifactFile[]>([]);
  const [mediaAssets, setMediaAssets] = useState<MediaAsset[]>([]);
  const [videoInfo, setVideoInfo] = useState<any>(null);
  const [selectedFileContent, setSelectedFileContent] = useState<any>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);

  const rawStatus = status?.status || 'idle';
  const isRunning = status?.is_running || false;
  const currentPhase = status?.current_phase || 'IDEA_GENERATION';

  // Eliminating Waterfalls (async-parallel) & Re-render Optimization (rerender-dependencies):
  // Fetch artifacts, assets & video in parallel using Promise.allSettled with primitive dependencies
  useEffect(() => {
    if (!projectName) return;
    let isCancelled = false;

    Promise.allSettled([
      api.listArtifacts(projectName),
      api.listMediaAssets(projectName),
      api.getRenderedVideo(projectName),
    ]).then(([artifactsRes, mediaRes, videoRes]) => {
      if (isCancelled) return;
      if (artifactsRes.status === 'fulfilled') setArtifacts(artifactsRes.value);
      if (mediaRes.status === 'fulfilled') setMediaAssets(mediaRes.value);
      if (videoRes.status === 'fulfilled') setVideoInfo(videoRes.value);
      else setVideoInfo(null);
    });

    return () => {
      isCancelled = true;
    };
  }, [projectName, rawStatus, isRunning, currentPhase]);

  useEffect(() => {
    if (!selectedFileContent) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedFileContent(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedFileContent]);

  const handleSelectFile = async (file: ArtifactFile) => {
    try {
      const data = await api.readArtifactFile(projectName, file.relative_path);
      setSelectedFileContent({ file: file.relative_path, data });
    } catch (err: any) {
      toast.error(err.message, 'Error Loading File');
    }
  };

  const [reportText, setReportText] = useState('');
  const [submittingReport, setSubmittingReport] = useState(false);

  const handleSubmitReport = async () => {
    if (!reportText.trim()) return;
    setSubmittingReport(true);
    try {
      const res = await api.submitReport(projectName, reportText.trim());
      toast.success(res.message || 'Report submitted successfully!');
      setReportText('');
      await resumePipeline({ no_upload: true });
    } catch (err: any) {
      toast.error(err.message, 'Report Submit Error');
    } finally {
      setSubmittingReport(false);
    }
  };

  const handleStart = async () => {
    try {
      await startPipeline({ no_upload: true });
      toast.success('Pipeline started!');
    } catch (err: any) {
      toast.error(err.message, 'Failed to start pipeline');
    }
  };

  const handleResume = async () => {
    try {
      await resumePipeline({ no_upload: true });
      toast.success('Pipeline resumed!');
    } catch (err: any) {
      toast.error(err.message, 'Failed to resume pipeline');
    }
  };

  const handleStop = async () => {
    try {
      await stopPipeline();
      toast.warning('Pipeline stopped');
    } catch (err: any) {
      toast.error(err.message, 'Failed to stop pipeline');
    } finally {
      setShowStopConfirm(false);
    }
  };

  // Re-render Optimization (rerender-memo): Memoize status banner details
  const banner = useMemo(() => {
    if (isRunning) {
      return {
        bg: 'rgba(0, 210, 160, 0.1)',
        border: 'var(--accent-success)',
        title: '⚡ Pipeline Running',
        desc: `Orchestrator active in phase: ${currentPhase}. Streaming live console logs below...`,
        icon: '⚙️',
      };
    }
    if (rawStatus === 'paused_waiting_for_gemini_report') {
      return {
        bg: 'rgba(255, 193, 7, 0.12)',
        border: 'var(--accent-warning)',
        title: '📄 Paused — Research Report Required',
        desc: 'Please paste your deep research report below to advance the pipeline to Synthesis.',
        icon: '⏸️',
      };
    }
    if (rawStatus === 'paused_for_script_review') {
      return {
        bg: 'rgba(162, 155, 254, 0.15)',
        border: 'var(--phase-scripting)',
        title: '✏️ Paused — Script Ready for Review',
        desc: 'Open Director\'s Chair to tweak narration and visual cues before rendering.',
        icon: '🎬',
      };
    }
    if (rawStatus === 'completed') {
      return {
        bg: 'rgba(0, 210, 160, 0.1)',
        border: 'var(--accent-success)',
        title: '🎉 Documentary Production Complete',
        desc: 'Final vertical Short video has been generated and ready for preview/download.',
        icon: '✅',
      };
    }
    return {
      bg: 'var(--bg-glass)',
      border: 'var(--border-subtle)',
      title: `Status: ${rawStatus}`,
      desc: 'Click "Start Pipeline" to begin documentary generation.',
      icon: 'ℹ️',
    };
  }, [isRunning, rawStatus, currentPhase]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-secondary)' }}>
              ← Back
            </Link>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 700 }}>Project: {projectName}</h1>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '12px' }}>
          {isRunning ? (
            <button className="btn btn-danger" onClick={() => setShowStopConfirm(true)} disabled={loading}>
              {loading ? <span className="spinner" /> : '⏹️ Stop Execution'}
            </button>
          ) : (
            <>
              <button className="btn btn-primary" onClick={handleStart} disabled={loading}>
                {loading ? <span className="spinner" /> : '▶️ Start Pipeline'}
              </button>
              <button className="btn btn-secondary" onClick={handleResume} disabled={loading}>
                {loading ? <span className="spinner" /> : '⏩ Resume'}
              </button>
            </>
          )}

          <Link to={`/project/${projectName}/script`} className="btn btn-secondary">
            ✏️ Director's Chair
          </Link>
        </div>
      </div>

      {/* Status Banner */}
      <div
        className="glass-card"
        style={{
          padding: '16px 20px',
          background: banner.bg,
          borderLeft: `4px solid ${banner.border}`,
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
        }}
      >
        <span style={{ fontSize: '1.4rem' }}>{banner.icon}</span>
        <div>
          <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>{banner.title}</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{banner.desc}</p>
        </div>
      </div>

      {/* Phase Stepper */}
      <PhaseStepper currentPhase={currentPhase} onSelectPhase={(phase) => overridePhase(phase)} />

      {/* Submit Deep Research Report Card */}
      {(status?.status === 'paused_waiting_for_gemini_report' || currentPhase === 'WAIT_FOR_GEMINI_REPORT') && (
        <div className="glass-card" style={{ padding: '24px', borderLeft: '4px solid var(--phase-research)' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '8px' }}>📄 Submit Deep Research Report</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '14px' }}>
            Paste your deep research report text or dossier below to continue the pipeline to Synthesis & Evidence gathering.
          </p>

          <textarea
            rows={6}
            placeholder="Paste your Deep Research report text here..."
            value={reportText}
            onChange={(e) => setReportText(e.target.value)}
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', marginBottom: '12px' }}
          />

          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
            <button className="btn btn-primary" onClick={handleSubmitReport} disabled={submittingReport || !reportText.trim()}>
              {submittingReport ? <><span className="spinner" /> Submitting...</> : '📥 Submit Report & Continue Pipeline'}
            </button>
          </div>
        </div>
      )}

      {/* Main Grid: Logs + Media Browser */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <LogStream logs={logs} onClear={clearLogs} />

          {videoInfo && <VideoPlayer videoUrl={videoInfo.url} projectName={projectName} />}
        </div>

        <div>
          <MediaBrowser
            artifacts={artifacts}
            mediaAssets={mediaAssets}
            onSelectFile={handleSelectFile}
          />
        </div>
      </div>

      {/* File Inspector Modal */}
      {selectedFileContent && (
        <div
          onClick={() => setSelectedFileContent(null)}
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
            style={{
              width: '720px',
              maxWidth: '90vw',
              maxHeight: '80vh',
              padding: '24px',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
              background: 'var(--bg-secondary)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>📄 {selectedFileContent.file}</h3>
              <button className="btn btn-secondary" onClick={() => setSelectedFileContent(null)}>
                ✕ Close
              </button>
            </div>

            <pre
              style={{
                flex: 1,
                overflowY: 'auto',
                background: '#07070a',
                padding: '16px',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem',
                fontFamily: 'var(--font-mono)',
                color: '#e8e8ed',
              }}
            >
              {typeof selectedFileContent.data === 'object'
                ? JSON.stringify(selectedFileContent.data, null, 2)
                : selectedFileContent.data.content}
            </pre>
          </div>
        </div>
      )}

      {/* Confirm Stop Dialog */}
      <ConfirmDialog
        isOpen={showStopConfirm}
        title="Stop Execution?"
        message="Are you sure you want to terminate the background pipeline worker?"
        confirmLabel="Stop Execution"
        variant="danger"
        onConfirm={handleStop}
        onCancel={() => setShowStopConfirm(false)}
      />
    </div>
  );
};
