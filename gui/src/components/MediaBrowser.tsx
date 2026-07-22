import React from 'react';
import type { MediaAsset, ArtifactFile } from '../api/types';
import { getMediaBase } from '../api/client';



interface MediaBrowserProps {
  artifacts: ArtifactFile[];
  mediaAssets: MediaAsset[];
  onSelectFile?: (file: ArtifactFile) => void;
}

export const MediaBrowser: React.FC<MediaBrowserProps> = ({ artifacts, mediaAssets, onSelectFile }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Media Gallery */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px' }}>🖼️ Downloaded Media Assets ({mediaAssets.length})</h3>
        
        {mediaAssets.length === 0 ? (
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No media assets downloaded for this project yet.</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' }}>
            {mediaAssets.map((asset, idx) => (
              <div
                key={idx}
                style={{
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.75rem',
                }}
              >
                {asset.type === 'image' ? (
                  <img
                    src={`${getMediaBase()}${asset.url}`}
                    alt={asset.name}
                    style={{ width: '100%', height: '100px', objectFit: 'cover' }}
                  />
                ) : (

                  <div style={{ height: '100px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
                    🎬 Video
                  </div>
                )}
                <div style={{ padding: '6px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {asset.name}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Artifact Files List */}
      <div className="glass-card" style={{ padding: '20px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px' }}>📁 Research Artifacts ({artifacts.length})</h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {artifacts.map((art, idx) => (
            <div
              key={idx}
              onClick={() => onSelectFile && onSelectFile(art)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-card)',
                cursor: 'pointer',
                fontSize: '0.85rem',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>📄</span>
                <span style={{ fontWeight: 500 }}>{art.relative_path}</span>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {(art.byte_size / 1024).toFixed(1)} KB
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
