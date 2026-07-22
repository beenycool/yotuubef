"""Configuration management endpoints."""

import os
import yaml
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ConfigUpdateRequest(BaseModel):
    yaml_content: str


class EnvUpdateRequest(BaseModel):
    env_vars: Dict[str, str]


@router.get("")
async def get_configuration():
    """Read current configuration YAML and environment status."""
    config_file = Path("config.yaml")
    yaml_raw = ""
    if config_file.exists():
        yaml_raw = config_file.read_text(encoding="utf-8")

    env_file = Path(".env")
    env_vars_masked = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if any(sec in key.lower() for sec in ["key", "secret", "token"]):
                    env_vars_masked[key] = f"***{val[-4:]}" if len(val) >= 4 else "***"
                else:
                    env_vars_masked[key] = val

    return {
        "yaml_content": yaml_raw,
        "env_vars": env_vars_masked,
    }


@router.put("")
async def update_configuration(body: ConfigUpdateRequest):
    """Save config.yaml content."""
    config_file = Path("config.yaml")
    try:
        # Validate YAML syntax
        parsed = yaml.safe_load(body.yaml_content)
        if not isinstance(parsed, dict):
            raise ValueError("Configuration must be a valid YAML dictionary")

        config_file.write_text(body.yaml_content, encoding="utf-8")
        return {"success": True, "message": "config.yaml updated successfully"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")


@router.put("/env")
async def update_env_vars(body: EnvUpdateRequest):
    """Update .env file key-value pairs."""
    env_file = Path(".env")
    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    existing_keys = {}
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s and not line_s.startswith("#") and "=" in line_s:
            key = line_s.split("=", 1)[0].strip()
            existing_keys[key] = i

    for k, v in body.env_vars.items():
        if k in existing_keys:
            lines[existing_keys[k]] = f"{k}={v}"
        else:
            lines.append(f"{k}={v}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"success": True, "message": ".env file updated successfully"}


@router.get("/preflight")
async def run_preflight_check():
    """Execute preflight check and return diagnostic report."""
    checks = []

    # API keys check
    req_keys = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "NVIDIA_NIM_API_KEY"]
    for k in req_keys:
        val = os.getenv(k, "").strip()
        checks.append(
            {
                "name": k,
                "passed": bool(val),
                "message": f"{k} is set" if val else f"Missing {k} in environment",
            }
        )

    # Background videos check
    bg_folder = Path(os.getenv("BACKGROUND_FOLDER", "data/backgrounds"))
    bg_videos = list(bg_folder.glob("*.mp4")) if bg_folder.exists() else []
    checks.append(
        {
            "name": "BACKGROUND_VIDEOS",
            "passed": len(bg_videos) > 0,
            "message": f"Found {len(bg_videos)} background MP4 video(s) in {bg_folder}",
        }
    )

    all_passed = all(c["passed"] for c in checks)
    return {"all_passed": all_passed, "checks": checks}
