"""WebSocket connection manager for real-time pipeline log streaming."""

import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections per project."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.global_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, project_name: str = None):
        await websocket.accept()
        if project_name:
            if project_name not in self.active_connections:
                self.active_connections[project_name] = set()
            self.active_connections[project_name].add(websocket)
        else:
            self.global_connections.add(websocket)
        logger.info("WebSocket client connected (project=%s)", project_name)

    def disconnect(self, websocket: WebSocket, project_name: str = None):
        if project_name and project_name in self.active_connections:
            self.active_connections[project_name].discard(websocket)
            if not self.active_connections[project_name]:
                del self.active_connections[project_name]
        self.global_connections.discard(websocket)
        logger.info("WebSocket client disconnected (project=%s)", project_name)

    async def broadcast_log(
        self,
        project_name: str,
        message: str,
        level: str = "info",
        phase: str = None,
    ):
        """Broadcast log message to project listeners and global listeners."""
        payload = {
            "type": "log",
            "project": project_name,
            "message": message,
            "level": level,
            "phase": phase,
        }
        text_data = json.dumps(payload)
        
        targets = set(self.global_connections)
        if project_name in self.active_connections:
            targets.update(self.active_connections[project_name])

        for connection in targets:
            try:
                await connection.send_text(text_data)
            except Exception as e:
                logger.warning("Failed to send WS log message: %s", e)

    async def broadcast_event(self, project_name: str, event_type: str, data: dict):
        """Broadcast arbitrary event to project listeners."""
        payload = {
            "type": event_type,
            "project": project_name,
            "data": data,
        }
        text_data = json.dumps(payload)
        targets = set(self.global_connections)
        if project_name in self.active_connections:
            targets.update(self.active_connections[project_name])

        for connection in targets:
            try:
                await connection.send_text(text_data)
            except Exception as e:
                logger.warning("Failed to send WS event message: %s", e)


ws_manager = ConnectionManager()
