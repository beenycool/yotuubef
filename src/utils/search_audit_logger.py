"""Search audit logger for capturing HTTP requests and responses."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Truncate response bodies to avoid huge log files
BODY_PREVIEW_MAX_CHARS = 2000

# Patterns to redact API keys
REDACT_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|auth(orization)?|token|bearer)\s*[:=]\s*['\"]?[^\s'\"]+"), "***REDACTED***"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer ***REDACTED***"),
    (re.compile(r"X-Subscription-Token:\s*[^\s]+"), "X-Subscription-Token: ***REDACTED***"),
]


def _redact_sensitive(value: str) -> str:
    """Redact API keys and tokens from headers/params."""
    if not value or not isinstance(value, str):
        return str(value)

    result = value
    for pattern, replacement in REDACT_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Redact Authorization and token headers."""
    sanitized = {}
    for k, v in headers.items():
        if k.lower() in ("authorization", "x-subscription-token", "api-key"):
            sanitized[k] = "***REDACTED***"
        else:
            sanitized[k] = v
    return sanitized


class SearchAuditLogger:
    """
    Logs HTTP search requests and responses to a JSONL file.
    Redacts API keys; truncates response bodies.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Append a JSONL entry to the log file."""
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except OSError as exc:
            logger.warning("SearchAuditLogger write failed: %s", exc)

    def log_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers_sanitized: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log an outgoing HTTP request."""
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "type": "request",
            "method": method,
            "url": str(url),
            "params": params or {},
            "headers_sanitized": _sanitize_headers(headers_sanitized or {}),
        }
        self._write_entry(entry)

    def log_response(
        self,
        status: int,
        body_preview: str,
        duration_ms: float,
        url: Optional[str] = None,
    ) -> None:
        """Log an HTTP response."""
        if isinstance(body_preview, str) and len(body_preview) > BODY_PREVIEW_MAX_CHARS:
            body_preview = body_preview[:BODY_PREVIEW_MAX_CHARS] + "...[truncated]"

        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "type": "response",
            "status": status,
            "body_preview": body_preview,
            "duration_ms": round(duration_ms, 2),
            "url": url,
        }
        self._write_entry(entry)
