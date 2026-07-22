import React from 'react';
import type { ScriptSegment } from '../api/types';


interface SegmentCardProps {
  index: number;
  segment: ScriptSegment;
  onChange: (updated: ScriptSegment) => void;
  onDelete: () => void;
}

export const SegmentCard: React.FC<SegmentCardProps> = ({ index, segment, onChange, onDelete }) => {
  const wordCount = segment.narration.trim() ? segment.narration.trim().split(/\s+/).length : 0;

  return (
    <div className="glass-card" style={{ padding: '20px', marginBottom: '16px', borderLeft: '4px solid var(--accent-primary)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--accent-secondary)' }}>
            Segment #{index + 1}
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Start: {segment.time_seconds}s | Est. Duration: {segment.intended_duration_seconds}s
          </span>
        </div>

        <button className="btn btn-danger" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={onDelete}>
          Delete
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {/* Narration */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Narration Text</label>
            <span style={{ fontSize: '0.75rem', color: wordCount > 25 ? 'var(--accent-warning)' : 'var(--text-muted)' }}>
              {wordCount} words
            </span>
          </div>
          <textarea
            rows={3}
            value={segment.narration}
            onChange={(e) => onChange({ ...segment, narration: e.target.value })}
            placeholder="Narrator voiceover script for this segment..."
          />
        </div>

        {/* Cues grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Emotion</label>
            <select
              value={segment.emotion}
              onChange={(e) => onChange({ ...segment, emotion: e.target.value as any })}
            >
              <option value="dramatic">dramatic</option>
              <option value="excited">excited</option>
              <option value="calm">calm</option>
              <option value="neutral">neutral</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Pace</label>
            <select
              value={segment.pace}
              onChange={(e) => onChange({ ...segment, pace: e.target.value as any })}
            >
              <option value="fast">fast</option>
              <option value="normal">normal</option>
              <option value="slow">slow</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Text Overlay</label>
            <input
              type="text"
              value={segment.text_overlay || ''}
              onChange={(e) => onChange({ ...segment, text_overlay: e.target.value })}
              placeholder="On-screen text"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Duration (s)</label>
            <input
              type="number"
              step="0.5"
              value={segment.intended_duration_seconds}
              onChange={(e) => onChange({ ...segment, intended_duration_seconds: parseFloat(e.target.value) || 3.0 })}
            />
          </div>
        </div>

        {/* Visual directive & Asset */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Visual Directive</label>
            <input
              type="text"
              value={segment.visual_directive || ''}
              onChange={(e) => onChange({ ...segment, visual_directive: e.target.value })}
              placeholder="e.g. Zoom in on date, highlight document"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Visual Asset Path</label>
            <input
              type="text"
              value={segment.visual_asset_path || ''}
              onChange={(e) => onChange({ ...segment, visual_asset_path: e.target.value })}
              placeholder="e.g. research/media_images/evidence1.png"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
