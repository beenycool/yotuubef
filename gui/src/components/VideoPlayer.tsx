import React from 'react';
import { getMediaBase } from '../api/client';

interface VideoPlayerProps {
  videoUrl: string;
  projectName: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl, projectName }) => {
  const fullUrl = `${getMediaBase()}${videoUrl}`;


  return (
    <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>🎬 Final Rendered Video</h3>

      <div style={{ width: '300px', height: '533px', background: '#000', borderRadius: 'var(--radius-md)', overflow: 'hidden', boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }}>
        <video
          controls
          src={fullUrl}
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
      </div>

      <a href={fullUrl} download={`${projectName}.mp4`} className="btn btn-primary">
        📥 Download MP4 Video
      </a>
    </div>
  );
};
