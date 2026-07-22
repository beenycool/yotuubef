"""
Sync bridge to the async documentary pipeline.

Gradio callbacks are synchronous, so this module wraps the async
orchestrator with `asyncio.run()` and provides helper functions for
listing projects, loading state, and reading results without an event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)


def _validate_project_name(project_name: str) -> Path:
    """Return resolved project dir under findings/, or raise ValueError."""
    base = (REPO_ROOT / "findings").resolve()
    project_dir = (base / project_name).resolve()
    if project_dir != base and base not in project_dir.parents:
        raise ValueError(f"Invalid project name: {project_name!r}")
    return project_dir


# Pipeline phases sorted in order
PHASE_ORDER: List[str] = [
    "IDEA_GENERATION",
    "WAIT_FOR_GEMINI_REPORT",
    "SYNTHESIS",
    "EVIDENCE_GATHERING",
    "SCRIPTING",
    "VIDEO_RENDER",
]

PHASE_LABELS: Dict[str, str] = {
    "IDEA_GENERATION": "\U0001F9EC Idea Generation",
    "WAIT_FOR_GEMINI_REPORT": "\u23F3 Waiting for Deep Research",
    "SYNTHESIS": "\U0001F9E9 Synthesis",
    "EVIDENCE_GATHERING": "\U0001F50D Evidence Gathering",
    "SCRIPTING": "\U0001F4DD Scripting",
    "VIDEO_RENDER": "\U0001F3AC Video Render",
}


# ── Project listing ──────────────────────────────────────────────────


def list_projects() -> List[Dict[str, Any]]:
    """Scan findings/ and return a list of project summaries."""
    findings_dir = REPO_ROOT / "findings"
    if not findings_dir.exists():
        return []
    out = []
    for entry in sorted(findings_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not entry.is_dir():
            continue
        state_path = entry / "research" / "state.json"
        info = {
            "name": entry.name,
            "dir": str(entry),
            "phase": "—",
            "status": "—",
            "reddit_url": "",
            "created_at": "",
            "updated_at": "",
            "findings": 0,
        }
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                info["phase"] = state.get("current_phase", "—")
                info["status"] = state.get("status", "—")
                info["created_at"] = state.get("created_at", "")
                info["updated_at"] = state.get("updated_at", "")
                info["findings"] = len(state.get("findings", []))
                info["reddit_url"] = state.get("metadata", {}).get("reddit_url", "")
            except Exception as exc:
                logger.warning("Failed to read state for %s: %s", entry.name, exc)
        out.append(info)
    return out


def get_project_state(project_name: str) -> Optional[Dict[str, Any]]:
    """Return raw state.json as a dict, or None if missing."""
    project_dir = _validate_project_name(project_name)
    state_path = project_dir / "research" / "state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def get_project_files(project_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """List research artefacts (ideas, scripts, media, etc.) for a project."""
    project_dir = _validate_project_name(project_name)
    research_dir = project_dir / "research"
    folders = [
        "ideas",
        "reports",
        "synthesis",
        "evidence",
        "scripts",
        "transcripts",
        "summaries",
        "logs",
        "media_images",
        "media_videos",
    ]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for folder in folders:
        d = research_dir / folder
        items: List[Dict[str, Any]] = []
        if d.exists():
            for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.is_file():
                    items.append(
                        {
                            "name": f.name,
                            "path": str(f),
                            "size_kb": round(f.stat().st_size / 1024, 1),
                            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                        }
                    )
        out[folder] = items
    return out


def get_project_video(project_name: str) -> Optional[str]:
    """Return the local video path if render finished."""
    project_dir = _validate_project_name(project_name)
    meta_path = project_dir / "research" / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        if meta.get("final_video_path"):
            p = Path(meta["final_video_path"])
            if p.exists():
                return str(p)
    result_files = list((REPO_ROOT / "data" / "results").glob("results_hybrid_*.json"))
    for rp in result_files:
        try:
            rdata = json.loads(rp.read_text())
        except Exception:
            continue
        if rdata.get("project_name") == project_name:
            vp = rdata.get("final_video_path")
            if vp and Path(vp).exists():
                return vp
    processed = REPO_ROOT / "processed"
    if processed.exists():
        videos = sorted(
            processed.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if videos:
            return str(videos[0])
    return None


# ── Script loading / saving ──────────────────────────


def load_script(project_name: str) -> Optional[Dict[str, Any]]:
    """Load final_script.json for a project."""
    project_dir = _validate_project_name(project_name)
    script_path = project_dir / "research" / "scripts" / "final_script.json"
    if not script_path.exists():
        return None
    return json.loads(script_path.read_text(encoding="utf-8"))


def save_script(project_name: str, script_data: Dict[str, Any]) -> str:
    """Overwrite final_script.json for a project."""
    project_dir = _validate_project_name(project_name)
    script_path = project_dir / "research" / "scripts" / "final_script.json"
    script_path.write_text(json.dumps(script_data, indent=2, ensure_ascii=True), encoding="utf-8")
    return f"Saved to {script_path}"


# ── Pipeline execution (subprocess for isolation) ──────────────


class PipelineRunner:
    """Run a pipeline in a background subprocess and stream stdout lines."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._log_lines: List[str] = []
        self._done = False

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def done(self) -> bool:
        return self._done

    def start(
        self,
        project_name: str,
        reddit_url: str = "",
        resume: bool = False,
        phase_override: str = "",
        no_upload: bool = True,
        no_auto_research: bool = False,
        gemini_report_path: str = "",
    ) -> None:
        cmd = [sys.executable, "main.py", project_name]
        if resume:
            cmd.append("--resume")
        if phase_override:
            cmd.extend(["--phase", phase_override])
        if no_upload:
            cmd.append("--no-upload")
        if no_auto_research:
            cmd.append("--no-auto-research")
        if gemini_report_path:
            cmd.extend(["--gemini-report", gemini_report_path])
        if reddit_url.strip():
            cmd.extend(["--reddit-url", reddit_url.strip()])

        self._log_lines = [f"$ {' '.join(cmd)}\n"]
        self._done = False

        def _run():
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self._log_lines.append(line)
                self._proc.wait()
                code = self._proc.returncode
                self._log_lines.append(f"\n[exit code {code}]\n")
            except Exception as exc:
                self._log_lines.append(f"\n[ERROR: {exc}]\n")
            finally:
                self._done = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def get_logs(self) -> str:
        return "".join(self._log_lines)

    def stop(self) -> None:
        if self.is_running and self._proc is not None:
            self._proc.terminate()


# Singleton runner for the dashboard
RUNNER = PipelineRunner()


# ── History ─────────────────────────────────────────────────────────────


def get_result_files() -> List[Dict[str, Any]]:
    """Read all results_hybrid_*.json files into summary dicts."""
    results_dir = REPO_ROOT / "data" / "results"
    if not results_dir.exists():
        return []
    out = []
    for rp in sorted(results_dir.glob("results_hybrid_*.json"), reverse=True):
        try:
            data = json.loads(rp.read_text())
            out.append(
                {
                    "file": rp.name,
                    "project": data.get("project_name", ""),
                    "phase": data.get("current_phase", ""),
                    "success": data.get("success", False),
                    "paused": data.get("paused", False),
                    "video": data.get("final_video_path", ""),
                    "error": data.get("error", ""),
                    "timestamp": rp.stem.replace("results_hybrid_", ""),
                }
            )
        except Exception:
            continue
    return out


# ── Config loading / saving ──────────────────────────────────────────


def load_config_yaml() -> Dict[str, Any]:
    """Return config.yaml as a dict."""
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def save_config_yaml(data: Dict[str, Any]) -> str:
    """Write back to config.yaml."""
    config_path = REPO_ROOT / "config.yaml"
    backup = config_path.with_suffix(".yaml.bak")
    if config_path.exists():
        backup.write_text(config_path.read_text(encoding="utf-8"))
    config_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return f"Saved config.yaml (backup at {backup.name})"


# ── Database queries (sync sqlite) ─────────────────────────────────


def get_upload_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Query the SQLite uploads table synchronously."""
    import sqlite3

    db_path = REPO_ROOT / "data" / "databases" / "youtube_shorts.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM uploads ORDER BY upload_timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        logger.warning("DB query failed: %s", exc)
        return []


# ── Background monitor ──────────────────────────────────────────────────


def get_monitor_status() -> Dict[str, Any]:
    """Return current pipeline + project status for the dashboard."""
    running = RUNNER.is_running
    logs = RUNNER.get_logs()
    projects = list_projects()
    active: Optional[Dict[str, Any]] = None
    for p in projects:
        if p["status"] in ("active", "paused_for_script_review", "paused"):
            active = p
            break
    return {
        "running": running,
        "done": RUNNER.done,
        "logs": logs,
        "projects": projects,
        "active": active,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
