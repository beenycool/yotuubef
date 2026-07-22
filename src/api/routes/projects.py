"""Projects CRUD endpoints."""

import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.hybrid_documentary_state_machine import (
    load_run_state,
    setup_project_workspace,
)

router = APIRouter()


class ProjectCreateRequest(BaseModel):
    name: str
    reddit_url: Optional[str] = None


class ProjectResponse(BaseModel):
    name: str
    current_phase: str
    status: str
    created_at: str
    updated_at: str
    has_script: bool
    has_video: bool
    transition_count: int
    reddit_url: Optional[str] = None


@router.get("", response_model=List[ProjectResponse])
async def list_projects():
    """List all projects in findings/ directory."""
    findings_dir = Path("findings")
    if not findings_dir.exists():
        return []

    projects = []
    for item in findings_dir.iterdir():
        if item.is_dir():
            state_file = item / "research" / "state.json"
            if state_file.exists():
                try:
                    state = load_run_state(item)
                    script_file = item / "research" / "scripts" / "final_script.json"
                    has_video = bool(state.metadata.get("final_video_path") and Path(state.metadata["final_video_path"]).exists())
                    projects.append(
                        ProjectResponse(
                            name=state.project_name,
                            current_phase=state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
                            status=state.status,
                            created_at=state.created_at,
                            updated_at=state.updated_at,
                            has_script=script_file.exists(),
                            has_video=has_video,
                            transition_count=len(state.transitions),
                            reddit_url=state.metadata.get("reddit_url"),
                        )
                    )
                except Exception:
                    continue

    projects.sort(key=lambda p: p.updated_at, reverse=True)
    return projects


@router.get("/{name}")
async def get_project(name: str):
    """Get project details and complete state."""
    project_dir = Path("findings") / name
    state_file = project_dir / "research" / "state.json"
    if not state_file.exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    
    state = load_run_state(project_dir)
    return state.model_dump()


@router.post("", response_model=ProjectResponse)
async def create_project(body: ProjectCreateRequest):
    """Create a new project workspace."""
    clean_name = body.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    
    project_dir = setup_project_workspace(clean_name)
    state = load_run_state(project_dir)
    
    if body.reddit_url:
        state.metadata["reddit_url"] = body.reddit_url.strip()
        state.context_snapshot = f"Reddit URL: {body.reddit_url.strip()}"
        from src.hybrid_documentary_state_machine import save_run_state
        save_run_state(state)

    script_file = project_dir / "research" / "scripts" / "final_script.json"
    return ProjectResponse(
        name=state.project_name,
        current_phase=state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
        status=state.status,
        created_at=state.created_at,
        updated_at=state.updated_at,
        has_script=script_file.exists(),
        has_video=False,
        transition_count=len(state.transitions),
        reddit_url=state.metadata.get("reddit_url"),
    )


@router.delete("/{name}")
async def delete_project(name: str):
    """Delete project workspace."""
    project_dir = Path("findings") / name
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    
    shutil.rmtree(project_dir, ignore_errors=True)
    return {"success": True, "message": f"Project '{name}' deleted"}
