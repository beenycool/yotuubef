import React, { useState, useEffect } from 'react';
import type { ScriptData, ScriptSegment } from '../api/types';
import { SegmentCard } from './SegmentCard';
import { useToast } from './Toast';

interface ScriptEditorProps {
  initialScript: ScriptData;
  onSave: (script: ScriptData) => Promise<void>;
  onSaveAndRender?: (script: ScriptData) => Promise<void>;
}

export const ScriptEditor: React.FC<ScriptEditorProps> = ({ initialScript, onSave, onSaveAndRender }) => {
  const [script, setScript] = useState<ScriptData>(initialScript);
  const [saving, setSaving] = useState(false);
  const toast = useToast();

  const handleSegmentChange = (index: number, updated: ScriptSegment) => {
    const newSegments = [...script.segments];
    newSegments[index] = updated;
    setScript({ ...script, segments: newSegments });
  };

  const handleAddSegment = () => {
    const lastSeg = script.segments[script.segments.length - 1];
    const startTime = lastSeg ? lastSeg.time_seconds + lastSeg.intended_duration_seconds : 0;
    const newSeg: ScriptSegment = {
      time_seconds: startTime,
      intended_duration_seconds: 6.0,
      narration: '',
      evidence_refs: [],
      pace: 'fast',
      emotion: 'dramatic',
    };
    setScript({ ...script, segments: [...script.segments, newSeg] });
  };

  const handleDeleteSegment = (index: number) => {
    const newSegments = script.segments.filter((_, i) => i !== index);
    setScript({ ...script, segments: newSegments });
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(script);
      toast.success('Script saved successfully!');
    } catch (err: any) {
      toast.error(err.message, 'Save Failed');
    } finally {
      setSaving(false);
    }
  };

  const handleSaveAndRender = async () => {
    setSaving(true);
    try {
      await onSave(script);
      toast.success('Script saved! Initiating video render...');
      if (onSaveAndRender) {
        await onSaveAndRender(script);
      }
    } catch (err: any) {
      toast.error(err.message, 'Render Trigger Error');
    } finally {
      setSaving(false);
    }
  };

  // Keyboard shortcut Ctrl+S / Ctrl+Enter
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleSaveAndRender();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [script]);

  const totalDuration = script.segments.reduce((acc, s) => acc + (s.intended_duration_seconds || 0), 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header controls */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>🎬 Director's Chair: Script Editor</h2>
          <div className="badge" style={{ background: 'rgba(255, 255, 255, 0.1)', color: 'var(--accent-secondary)' }}>
            Total Est. Duration: {totalDuration.toFixed(1)}s
          </div>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>Short Title</label>
            <input
              type="text"
              value={script.title}
              onChange={(e) => setScript({ ...script, title: e.target.value })}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>Hook (Under 3 seconds)</label>
            <textarea
              rows={2}
              value={script.hook}
              onChange={(e) => setScript({ ...script, hook: e.target.value })}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>Loop Bridge Text</label>
            <input
              type="text"
              value={script.loop_bridge || ''}
              onChange={(e) => setScript({ ...script, loop_bridge: e.target.value })}
              placeholder="Text connecting end of short back to hook..."
            />
          </div>
        </div>
      </div>

      {/* Segment List */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Narrative Segments ({script.segments.length})</h3>
          <button className="btn btn-secondary" onClick={handleAddSegment}>
            ➕ Add Segment
          </button>
        </div>

        {script.segments.map((seg, idx) => (
          <SegmentCard
            key={idx}
            index={idx}
            segment={seg}
            onChange={(updated) => handleSegmentChange(idx, updated)}
            onDelete={() => handleDeleteSegment(idx)}
          />
        ))}
      </div>

      {/* Sticky Action Bar */}
      <div
        className="glass-card"
        style={{
          position: 'sticky',
          bottom: '24px',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(18, 18, 26, 0.95)',
          zIndex: 10,
        }}
      >
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Tip: Press <kbd style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px' }}>Ctrl+S</kbd> to save, <kbd style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px' }}>Ctrl+Enter</kbd> to save & render
        </span>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn-secondary" onClick={handleSave} disabled={saving}>
            {saving ? <><span className="spinner" /> Saving...</> : '💾 Save Script Changes'}
          </button>

          {onSaveAndRender && (
            <button className="btn btn-primary" onClick={handleSaveAndRender} disabled={saving}>
              {saving ? <><span className="spinner" /> Saving...</> : '🎥 Save & Render Video'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
