"""Video management and streaming endpoints."""

import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/backgrounds")
async def list_background_videos():
    """List available background video files."""
    bg_folder = Path(os.getenv("BACKGROUND_FOLDER", "data/backgrounds"))
    if not bg_folder.exists():
        return []

    videos = []
    for p in bg_folder.glob("*.mp4"):
        videos.append(
            {
                "name": p.name,
                "path": str(p),
                "byte_size": p.stat().st_size,
                "updated_at": p.stat().st_mtime,
            }
        )
    return videos


@router.get("/{project}")
async def get_rendered_video(project: str):
    """Get metadata and URL for project's rendered final video."""
    processed_dir = Path("processed")
    state_file = Path("findings") / project / "research" / "state.json"

    video_path = None
    if state_file.exists():
        import json
        state = json.loads(state_file.read_text(encoding="utf-8"))
        video_path = state.get("metadata", {}).get("final_video_path")

    if not video_path or not Path(video_path).exists():
        # Fallback to finding latest mp4 matching project name in processed/
        candidates = sorted(
            processed_dir.glob(f"*{project}*.mp4") if processed_dir.exists() else [],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            video_path = str(candidates[0])

    if not video_path or not Path(video_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Rendered video not found for project '{project}'",
        )

    p = Path(video_path)
    rel = p.relative_to(Path("processed")) if "processed" in str(p) else p.name
    return {
        "project": project,
        "name": p.name,
        "path": str(p),
        "url": f"/processed/{rel}",
        "byte_size": p.stat().st_size,
        "created_at": p.stat().st_mtime,
    }
