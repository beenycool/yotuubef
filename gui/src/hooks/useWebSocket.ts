import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSLogMessage, WSEventMessage } from '../api/types';


export function useWebSocket(projectName?: string) {
  const [logs, setLogs] = useState<WSLogMessage[]>([]);
  const [lastEvent, setLastEvent] = useState<WSEventMessage | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const isHttps = typeof window !== 'undefined' && window.location.protocol === 'https:';
    const wsProto = isHttps ? 'wss:' : 'ws:';
    const wsHost = typeof window !== 'undefined'
      ? (window.location.port === '5173' ? `${window.location.hostname}:8420` : window.location.host)
      : 'localhost:8420';

    const wsUrl = projectName
      ? `${wsProto}//${wsHost}/ws/${projectName}`
      : `${wsProto}//${wsHost}/ws`;

    const ws = new WebSocket(wsUrl);

    socketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          setLogs((prev) => (prev.length >= 1000 ? [...prev.slice(-999), data as WSLogMessage] : [...prev, data as WSLogMessage]));
        } else {
          setLastEvent(data as WSEventMessage);
        }
      } catch (err) {
        console.error('WS parse error:', err);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WS error:', error);
    };

    return () => {
      ws.close();
    };
  }, [projectName]);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  return { logs, lastEvent, isConnected, clearLogs };
}
