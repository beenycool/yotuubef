import React, { useEffect, useRef, useState } from 'react';
import type { WSLogMessage } from '../api/types';


interface LogStreamProps {
  logs: WSLogMessage[];
  onClear?: () => void;
}

export const LogStream: React.FC<LogStreamProps> = ({ logs, onClear }) => {
  const endRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    if (autoScroll) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter((log) =>
    log.message.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '420px', overflow: 'hidden' }}>
      {/* Header bar */}
      <div
        style={{
          padding: '12px 16px',
          background: 'rgba(0, 0, 0, 0.4)',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
            💻 Live Console Output
          </span>
          <span className="badge" style={{ background: 'rgba(255, 255, 255, 0.1)', color: 'var(--text-secondary)' }}>
            {logs.length} lines
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <input
            type="text"
            placeholder="Filter logs..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: '160px', padding: '4px 8px', fontSize: '0.8rem' }}
          />

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>

          {onClear && (
            <button className="btn btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={onClear}>
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Logs container */}
      <div
        style={{
          flex: 1,
          padding: '16px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.85rem',
          overflowY: 'auto',
          background: '#07070a',
          color: '#d4d4d4',
          lineHeight: '1.6',
        }}
      >
        {filteredLogs.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No logs captured yet. Start or resume pipeline to view live stdout/stderr stream...
          </div>
        ) : (
          filteredLogs.map((log, index) => {
            let color = '#d4d4d4';
            if (log.level === 'error') color = '#ff4757';
            else if (log.level === 'warning') color = '#ffc107';
            else if (log.message.includes('[Hybrid]')) color = '#a29bfe';

            return (
              <div key={index} style={{ color, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {log.message}
              </div>
            );
          })
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
};
