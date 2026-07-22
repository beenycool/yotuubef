"""Pipeline execution routes with WebSocket log streaming."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.enhanced_orchestrator import EnhancedVideoOrchestrator
from src.hybrid_documentary_state_machine import load_run_state, PipelinePhase
from src.api.ws import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Active background running tasks
active_tasks: Dict[str, asyncio.Task] = {}


class PipelineStartRequest(BaseModel):
    reddit_url: Optional[str] = None
    no_upload: bool = True
    no_auto_research: bool = False
    gemini_report_path: Optional[str] = None


class PipelineResumeRequest(BaseModel):
    phase_override: Optional[str] = None
    gemini_report_path: Optional[str] = None
    no_upload: bool = True


class PipelinePhaseOverrideRequest(BaseModel):
    phase: str


class WSLogHandler(logging.Handler):
    """Custom logging handler to send logs over WebSocket."""

    def __init__(self, project_name: str, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.project_name = project_name
        self.loop = loop

    def emit(self, record):
        log_entry = self.format(record)
        level = record.levelname.lower()
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast_log(
                    project_name=self.project_name,
                    message=log_entry,
                    level=level,
                ),
                self.loop,
            )


async def _run_orchestrator_task(
    project_name: str,
    reddit_url: Optional[str] = None,
    resume: bool = False,
    phase_override: Optional[str] = None,
    gemini_report_path: Optional[str] = None,
    no_upload: bool = True,
    no_auto_research: bool = False,
):
    """Worker task executing pipeline with logging intercept."""
    loop = asyncio.get_running_loop()
    log_handler = WSLogHandler(project_name, loop)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        await ws_manager.broadcast_event(
            project_name, "pipeline_start", {"status": "running", "resume": resume}
        )
        orchestrator = EnhancedVideoOrchestrator()
        result = await orchestrator.process_hybrid_documentary_studio(
            project_name=project_name,
            reddit_url=reddit_url,
            resume=resume,
            phase_override=phase_override,
            gemini_report_path=gemini_report_path,
            no_upload=no_upload,
            no_auto_research=no_auto_research,
        )
        await ws_manager.broadcast_event(
            project_name, "pipeline_complete", {"result": result}
        )
        return result
    except Exception as exc:
        logger.exception("Pipeline task failed for project %s", project_name)
        await ws_manager.broadcast_event(
            project_name, "pipeline_error", {"error": str(exc)}
        )
    finally:
        root_logger.removeHandler(log_handler)
        active_tasks.pop(project_name, None)


@router.post("/{project}/start")
async def start_pipeline(project: str, body: PipelineStartRequest):
    """Start pipeline for a project."""
    if project in active_tasks and not active_tasks[project].done():
        raise HTTPException(
            status_code=400, detail=f"Pipeline is already running for project '{project}'"
        )

    task = asyncio.create_task(
        _run_orchestrator_task(
            project_name=project,
            reddit_url=body.reddit_url,
            resume=False,
            gemini_report_path=body.gemini_report_path,
            no_upload=body.no_upload,
            no_auto_research=body.no_auto_research,
        )
    )
    active_tasks[project] = task
    return {"status": "started", "project": project}


@router.post("/{project}/resume")
async def resume_pipeline(project: str, body: PipelineResumeRequest):
    """Resume a paused pipeline."""
    if project in active_tasks and not active_tasks[project].done():
        raise HTTPException(
            status_code=400, detail=f"Pipeline is already running for project '{project}'"
        )


    task = asyncio.create_task(
        _run_orchestrator_task(
            project_name=project,
            resume=True,
            phase_override=body.phase_override,
            gemini_report_path=body.gemini_report_path,
            no_upload=body.no_upload,
        )
    )
    active_tasks[project] = task
    return {"status": "resumed", "project": project}


@router.post("/{project}/phase")
async def override_phase(project: str, body: PipelinePhaseOverrideRequest):
    """Override pipeline phase."""
    project_dir = Path("findings") / project
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

    valid_phases = [p.value for p in PipelinePhase]
    if body.phase not in valid_phases:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid phase '{body.phase}'. Must be one of {valid_phases}",
        )

    state = load_run_state(project_dir)
    from src.hybrid_documentary_state_machine import set_phase
    set_phase(state, PipelinePhase(body.phase), "Manual API override")
    return {"status": "phase_updated", "current_phase": body.phase}


@router.get("/{project}/status")
async def get_pipeline_status(project: str):
    """Get active status and current state of pipeline."""
    project_dir = Path("findings") / project
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

    state = load_run_state(project_dir)
    is_running = project in active_tasks and not active_tasks[project].done()
    return {
        "project": project,
        "is_running": is_running,
        "status": state.status,
        "current_phase": state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
        "updated_at": state.updated_at,
        "metadata": state.metadata,
    }


@router.post("/{project}/stop")
async def stop_pipeline(project: str):
    """Stop active pipeline task."""
    if project in active_tasks and not active_tasks[project].done():
        active_tasks[project].cancel()
        active_tasks.pop(project, None)
        await ws_manager.broadcast_event(project, "pipeline_stopped", {})
        return {"status": "stopped", "project": project}
    return {"status": "not_running", "project": project}
