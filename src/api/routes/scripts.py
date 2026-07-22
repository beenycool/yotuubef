"""Script editor endpoints."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class SegmentUpdate(BaseModel):
    time_seconds: float = 0.0
    intended_duration_seconds: float = 6.0
    narration: str
    expression_cue: Optional[str] = None
    visual_asset_path: Optional[str] = None
    visual_directive: Optional[str] = None
    text_overlay: Optional[str] = None
    evidence_refs: List[str] = []
    pace: str = "fast"
    emotion: str = "dramatic"


class ScriptPayload(BaseModel):
    phase: str = "SCRIPTING"
    title: str
    hook: str
    loop_bridge: Optional[str] = ""
    segments: List[SegmentUpdate]
    sources_to_check: List[str] = []
    hashtags: List[str] = []


@router.get("/{project}")
async def get_script(project: str):
    """Read final_script.json for project."""
    script_file = Path("findings") / project / "research" / "scripts" / "final_script.json"
    if not script_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Script not found for project '{project}'. Run SCRIPTING phase first.",
        )

    try:
        data = json.loads(script_file.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read script JSON: {exc}")


@router.put("/{project}")
async def update_script(project: str, payload: ScriptPayload):
    """Save edited script back to final_script.json."""
    script_file = Path("findings") / project / "research" / "scripts" / "final_script.json"
    script_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        script_dict = payload.model_dump()
        script_file.write_text(json.dumps(script_dict, indent=2), encoding="utf-8")
        return {"success": True, "message": "Script updated successfully", "data": script_dict}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write script JSON: {exc}")
