"""Artifacts and research data browser endpoints."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/{project}")
async def list_artifacts(project: str):
    """List all research artifacts for a project."""
    research_dir = Path("findings") / project / "research"
    if not research_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project}' research dir not found")

    artifacts = []
    for path in research_dir.glob("**/*"):
        if path.is_file() and not path.name.startswith("."):
            rel_path = path.relative_to(research_dir)
            artifacts.append(
                {
                    "name": path.name,
                    "folder": str(rel_path.parent),
                    "relative_path": str(rel_path),
                    "byte_size": path.stat().st_size,
                    "updated_at": path.stat().st_mtime,
                }
            )

    return sorted(artifacts, key=lambda a: a["relative_path"])


@router.get("/{project}/file/{relative_path:path}")
async def read_artifact_file(project: str, relative_path: str):
    """Read contents of an artifact file (text/json)."""
    file_path = Path("findings") / project / "research" / relative_path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact file not found: {relative_path}")

    # Return JSON or raw text
    try:
        if file_path.suffix.lower() == ".json":
            return json.loads(file_path.read_text(encoding="utf-8"))
        else:
            return {"content": file_path.read_text(encoding="utf-8", errors="replace")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")


@router.get("/{project}/media")
async def list_media_assets(project: str):
    """List downloaded image and video assets for project."""
    raw_dir = Path("findings") / project / "raw_media"
    images_dir = Path("findings") / project / "research" / "media_images"
    videos_dir = Path("findings") / project / "research" / "media_videos"

    media_items = []
    for dir_path, media_type in [(raw_dir, "raw"), (images_dir, "image"), (videos_dir, "video")]:
        if dir_path.exists():
            for p in dir_path.glob("*"):
                    rel = p.relative_to(Path("findings"))
                    media_items.append(
                        {
                            "name": p.name,
                            "type": "image" if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] else "video",
                            "category": media_type,
                            "url": f"/media/{rel}",
                            "byte_size": p.stat().st_size,
                        }
                    )

    return media_items


from pydantic import BaseModel


class ReportSubmitRequest(BaseModel):
    content: str


@router.post("/{project}/report")
async def submit_research_report(project: str, body: ReportSubmitRequest):
    """Save pasted/uploaded research report and set state ready for synthesis."""
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Report content cannot be empty")

    project_dir = Path("findings") / project
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")

    from src.hybrid_documentary_state_machine import save_finding, load_run_state, set_phase, PipelinePhase
    report_file = save_finding(project_dir, "reports", "gemini_report.txt", body.content.strip())

    state = load_run_state(project_dir)
    state.gemini_report_path = str(report_file)
    state.status = "active"
    set_phase(state, PipelinePhase.SYNTHESIS, "Gemini report supplied via Web GUI")

    return {
        "success": True,
        "message": f"Report saved ({len(body.content)} chars). Ready for synthesis.",
        "report_path": str(report_file),
    }

