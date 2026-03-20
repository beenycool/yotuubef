"""Hybrid documentary state-machine utilities for research workflows."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from uuid import uuid4

import aiohttp

if TYPE_CHECKING:
    from src.utils.search_audit_logger import SearchAuditLogger
from pydantic import BaseModel, ConfigDict, Field

OpenAI = None
try:
    from openai import OpenAI as _OpenAI

    OpenAI = _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")
DEFAULT_SUMMARY_MODEL = os.getenv("NVIDIA_SUMMARY_MODEL", "qwen/qwen3.5-397b-a17b")
DEFAULT_TRANSCRIBE_MODEL = os.getenv(
    "NVIDIA_TRANSCRIBE_MODEL", "openai/whisper-large-v3"
)


class PipelinePhase(str, Enum):
    IDEA_GENERATION = "IDEA_GENERATION"
    WAIT_FOR_GEMINI_REPORT = "WAIT_FOR_GEMINI_REPORT"
    SYNTHESIS = "SYNTHESIS"
    EVIDENCE_GATHERING = "EVIDENCE_GATHERING"
    SCRIPTING = "SCRIPTING"
    VIDEO_RENDER = "VIDEO_RENDER"


class ResearchArtifact(BaseModel):
    folder: str
    filename: str
    relative_path: str
    created_at: str
    byte_size: int


class PhaseTransition(BaseModel):
    from_phase: Optional[PipelinePhase] = None
    to_phase: PipelinePhase
    reason: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    project_name: str
    project_dir: str
    current_phase: PipelinePhase = PipelinePhase.IDEA_GENERATION
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "active"
    context_snapshot: str = ""
    gemini_report_path: Optional[str] = None
    findings: List[ResearchArtifact] = Field(default_factory=list)
    transitions: List[PhaseTransition] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


@dataclass(frozen=True)
class MediaSearchResult:
    title: str
    url: str
    description: str
    source: str
    thumbnail_url: str = ""


class SummaryResult(BaseModel):
    context: str
    was_summarized: bool
    estimated_tokens_before: int
    estimated_tokens_after: int


class TranscriptArtifacts(BaseModel):
    transcript_text_path: str
    transcript_json_path: str
    transcript_srt_path: Optional[str] = None
    text: str


PHASE_JSON_CONTRACTS: Dict[PipelinePhase, str] = {
    PipelinePhase.IDEA_GENERATION: json.dumps(
        {
            "phase": "IDEA_GENERATION",
            "angles": [
                {
                    "id": "A1",
                    "title": "string",
                    "hook": "string",
                    "viability_score": 85,
                    "source_urls": ["https://example.com/source1"],
                }
            ],
            "gemini_deep_research_prompt": "string",
            "next_phase": "WAIT_FOR_GEMINI_REPORT",
        },
        indent=2,
    ),
    PipelinePhase.WAIT_FOR_GEMINI_REPORT: json.dumps(
        {
            "phase": "WAIT_FOR_GEMINI_REPORT",
            "status": "waiting|ready",
            "required_artifacts": ["gemini_report"],
            "missing": ["string"],
            "next_phase": "SYNTHESIS",
        },
        indent=2,
    ),
    PipelinePhase.SYNTHESIS: json.dumps(
        {
            "phase": "SYNTHESIS",
            "chosen_angle": "string",
            "reasoning": "string",
            "image_queries": ["string"],
            "video_queries": ["string"],
            "evidence_questions": ["string"],
            "next_phase": "EVIDENCE_GATHERING",
        },
        indent=2,
    ),
    PipelinePhase.EVIDENCE_GATHERING: json.dumps(
        {
            "phase": "EVIDENCE_GATHERING",
            "evidence_plan": [
                {
                    "claim": "string",
                    "evidence_needed": ["string"],
                    "priority": "high|medium|low",
                }
            ],
            "media_queries": ["string"],
            "next_phase": "SCRIPTING",
        },
        indent=2,
    ),
    PipelinePhase.SCRIPTING: json.dumps(
        {
            "phase": "SCRIPTING",
            "title": "string",
            "hook": "string",
            "loop_bridge": "string",
            "segments": [
                {
                    "time_seconds": 0.0,
                    "intended_duration_seconds": 6.0,
                    "narration": "string",
                    "visual_asset_path": "research/media_images/example.png",
                    "visual_directive": "string",
                    "text_overlay": "string",
                    "evidence_refs": ["string"],
                    "pace": "fast|normal|slow",
                    "emotion": "excited|calm|dramatic|neutral",
                }
            ],
            "sources_to_check": ["string"],
            "hashtags": ["string"],
        },
        indent=2,
    ),
    PipelinePhase.VIDEO_RENDER: json.dumps(
        {
            "phase": "VIDEO_RENDER",
            "status": "ready|rendered|failed",
            "final_script_path": "research/scripts/final_script.json",
            "output_video_path": "processed/hybrid_project_20260101_120000.mp4",
            "notes": "string",
        },
        indent=2,
    ),
}


STATE_MACHINE_PROMPT_TEMPLATE = """You are an elite YouTube Shorts Documentary Producer.

Guiding principle: COMPREHENSION AND CLARITY.
The audience must be able to follow the story easily. Avoid disjointed or overly dense narration.
Prefer clear receipts over generic b-roll: exact dates, quotes, handles, URLs, and clip timestamps.
For SCRIPTING, match a conversational rabbit-hole investigation tone:
- Open with a spoken question hook in Segment 1 narration.
- Escalate clue-by-clue logically.
- Keep language sharp, human, and perfectly paced (do not cram too many words into a short time).

CURRENT_PIPELINE_PHASE: {current_phase}
PROJECT: {project_name}
RUN_ID: {run_id}

Context:
{context}

Hard rules:
1) Return ONLY valid JSON.
2) Return ONLY the schema for CURRENT_PIPELINE_PHASE.
3) Do not include keys outside that phase contract.
4) If uncertain, use empty strings/arrays while preserving schema shape.
5) Never invent sources, dates, or quotes.
6) In SCRIPTING, map each segment to a concrete local asset path when available.

Phase contracts:
{contracts}

Return contract for CURRENT_PIPELINE_PHASE only.
"""


def _sanitize_project_name(project_name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", project_name.strip())
    value = value.strip("-._")
    return value or "untitled-project"


def _state_path(project_dir: Union[str, Path]) -> Path:
    return Path(project_dir) / "research" / "state.json"


def _ensure_text(
    value: Union[str, bytes, Mapping[str, Any], Sequence[Any]],
) -> Tuple[str, bool]:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace"), False
    if isinstance(value, str):
        return value, False
    if isinstance(value, Mapping) or isinstance(value, Sequence):
        return json.dumps(value, indent=2, ensure_ascii=True), True
    return str(value), False


def _write_research_readme(research_dir: Path) -> None:
    """Write a README explaining the research folder structure."""
    readme = research_dir / "README.md"
    content = """# Research Folder Structure

| Folder | Purpose |
|--------|---------|
| `ideas/` | Idea generation output, raw scout data |
| `reports/` | Gemini deep research report, deep research prompt |
| `synthesis/` | Synthesis JSON (chosen angle, queries) |
| `evidence/` | Media search results, evidence index |
| `scripts/` | Final script JSON |
| `transcripts/` | Transcripts from downloaded media |
| `summaries/` | Script context summary |
| `logs/` | Search audit logs |
| `media_images/` | Downloaded broll images |
| `media_videos/` | Reserved for video assets |
"""
    readme.write_text(content, encoding="utf-8")


def setup_project_workspace(project_name: str) -> Path:
    """Create findings workspace and initialize run state if absent."""
    sanitized_project = _sanitize_project_name(project_name)
    project_dir = Path("findings") / sanitized_project
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

    research_dir.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        (research_dir / folder).mkdir(parents=True, exist_ok=True)
    (project_dir / "raw_media").mkdir(parents=True, exist_ok=True)
    _write_research_readme(research_dir)

    state_path = _state_path(project_dir)
    if not state_path.exists():
        state = RunState(
            project_name=project_name,
            project_dir=str(project_dir),
            current_phase=PipelinePhase.IDEA_GENERATION,
        )
        save_run_state(state)
        logger.info("Initialized workspace state at %s", state_path)
    else:
        logger.debug("Workspace state already exists at %s", state_path)

    return project_dir


def load_run_state(project_dir: Union[str, Path]) -> RunState:
    state_path = _state_path(project_dir)
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return RunState(**payload)


def save_run_state(state: RunState) -> Path:
    state.updated_at = datetime.now(UTC).isoformat()
    state_path = _state_path(state.project_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    logger.debug("Saved run state to %s", state_path)
    return state_path


def set_phase(state: RunState, next_phase: PipelinePhase, reason: str) -> RunState:
    previous_phase = state.current_phase
    state.current_phase = next_phase
    state.transitions.append(
        PhaseTransition(from_phase=previous_phase, to_phase=next_phase, reason=reason)
    )
    save_run_state(state)
    logger.info("Phase transition: %s -> %s", previous_phase, next_phase)
    return state


def save_finding(
    project_dir: Union[str, Path],
    folder: str,
    filename: str,
    content: Union[str, bytes, Mapping[str, Any], Sequence[Any]],
) -> Path:
    """Save finding artifact under findings/<project>/research/<folder>/<filename>."""
    normalized_folder = folder.strip().strip("/")
    clean_filename = filename.strip().replace("..", "")
    if not clean_filename:
        raise ValueError("filename must not be empty")

    target_path = Path(project_dir) / "research" / normalized_folder / clean_filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    text_content, is_json_payload = _ensure_text(content)
    write_mode = "w"
    if isinstance(content, bytes):
        write_mode = "wb"

    if write_mode == "wb":
        if not isinstance(content, bytes):
            raise TypeError("Binary writes require bytes content")
        target_path.write_bytes(content)
    else:
        target_path.write_text(text_content, encoding="utf-8")

    logger.info(
        "Saved finding artifact %s (json_payload=%s)",
        target_path,
        is_json_payload,
    )
    return target_path


class HackclubMediaSearchClient:
    """Hackclub search helper for web, image, and video endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://search.hackclub.com/res/v1",
        timeout_seconds: int = 20,
        audit_logger: Optional["SearchAuditLogger"] = None,
    ):
        self.api_key = api_key or os.getenv("HACKCLUB_SEARCH_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.audit_logger = audit_logger
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "HackclubMediaSearchClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.close()

    async def search_web(self, query: str, count: int = 10) -> List[MediaSearchResult]:
        return await self._search("web/search", query, count)

    async def search_images(
        self, query: str, count: int = 10
    ) -> List[MediaSearchResult]:
        return await self._search("images/search", query, count)

    async def search_videos(
        self, query: str, count: int = 10
    ) -> List[MediaSearchResult]:
        return await self._search("videos/search", query, count)

    async def _search(
        self,
        endpoint: str,
        query: str,
        count: int,
    ) -> List[MediaSearchResult]:
        if not self.api_key:
            logger.warning("HACKCLUB_SEARCH_API_KEY is not configured")
            return []

        session = await self._ensure_session()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        params = {"q": query, "count": max(1, min(count, 50))}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        if self.audit_logger:
            self.audit_logger.log_request("GET", url, params, headers)

        start = time.perf_counter()
        try:
            async with session.get(url, headers=headers, params=params) as response:
                body = await response.text()
                duration_ms = (time.perf_counter() - start) * 1000
                if self.audit_logger:
                    self.audit_logger.log_response(
                        response.status,
                        body,
                        duration_ms,
                        url=url,
                    )
                if response.status >= 400:
                    logger.warning(
                        "Hackclub search failed %s status=%s body=%s",
                        endpoint,
                        response.status,
                        body[:300],
                    )
                    return []
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    return []
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            if self.audit_logger:
                self.audit_logger.log_response(500, str(exc), duration_ms, url=url)
            logger.error("Hackclub search request failed for %s: %s", endpoint, exc)
            return []

        return self._parse_search_results(endpoint, payload)

    def _parse_search_results(
        self,
        endpoint: str,
        payload: Any,
    ) -> List[MediaSearchResult]:
        candidates: List[Dict[str, Any]] = []

        if isinstance(payload, list):
            candidates = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            key_prefix = endpoint.split("/", 1)[0]
            key_bucket = payload.get(key_prefix)

            nested_candidates: List[Any] = []
            if isinstance(key_bucket, dict):
                nested_candidates.extend(
                    [
                        key_bucket.get("results"),
                        key_bucket.get("items"),
                        key_bucket.get("data"),
                    ]
                )

            nested_candidates.extend(
                [payload.get("results"), payload.get("items"), payload.get("data")]
            )

            for entry in nested_candidates:
                if isinstance(entry, list):
                    candidates.extend(
                        [item for item in entry if isinstance(item, dict)]
                    )

        parsed: List[MediaSearchResult] = []
        for item in candidates:
            url = (
                item.get("url")
                or item.get("link")
                or item.get("contentUrl")
                or item.get("source")
                or ""
            )
            if not isinstance(url, str) or not url:
                continue

            # FIX: Filter out logos and icons by checking filename only
            url_path = url.split("?")[0].split("#")[0]
            filename = url_path.rsplit("/", 1)[-1] if "/" in url_path else url_path
            filename_lower = filename.lower()
            if any(kw in filename_lower for kw in LOGO_FILTER_KEYWORDS):
                continue

            title = item.get("title") or item.get("name") or ""
            description = item.get("description") or item.get("snippet") or ""
            source = item.get("source") or item.get("domain") or "unknown"
            thumbnail_url = (
                item.get("thumbnail")
                or item.get("thumbnail_url")
                or item.get("image")
                or ""
            )

            # Strict filtering for logos, icons, and avatars
            if item.get("logo") is True:
                continue
            title_lower = str(title).lower()
            if any(bad in title_lower for bad in LOGO_FILTER_KEYWORDS):
                continue

            # FIX: Filter out results where thumbnail_url indicates it's a logo
            if (
                isinstance(thumbnail_url, str)
                and "'logo': true" in thumbnail_url.lower()
            ):
                try:
                    import ast

                    if isinstance(ast.literal_eval(thumbnail_url), dict):
                        parsed_thumb = ast.literal_eval(thumbnail_url)
                        if parsed_thumb.get("logo"):
                            continue
                except Exception:
                    pass

            parsed.append(
                MediaSearchResult(
                    title=str(title),
                    url=url,
                    description=str(description),
                    source=str(source),
                    thumbnail_url=str(thumbnail_url),
                )
            )

        logger.debug("Parsed %d %s results", len(parsed), endpoint)
        return parsed


def transcribe_media_file(
    media_path: Union[str, Path],
    project_dir: Union[str, Path],
    *,
    api_key: Optional[str] = None,
    base_url: str = NVIDIA_BASE_URL,
    model: str = DEFAULT_TRANSCRIBE_MODEL,
    language: Optional[str] = None,
) -> TranscriptArtifacts:
    """Transcribe a local media file using OpenAI client against NVIDIA base URL."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not available")
    if OpenAI is None:
        raise RuntimeError("OpenAI client is unavailable")

    media_file = Path(media_path)
    if not media_file.exists() or not media_file.is_file():
        raise FileNotFoundError(f"Media file not found: {media_file}")

    client = OpenAI(api_key=api_key or NVIDIA_API_KEY, base_url=base_url)

    request_kwargs: Dict[str, Any] = {"model": model}
    if language:
        request_kwargs["language"] = language
    request_kwargs["response_format"] = "verbose_json"
    request_kwargs["timestamp_granularities"] = ["segment"]

    with media_file.open("rb") as handle:
        transcript = client.audio.transcriptions.create(file=handle, **request_kwargs)

    transcript_text = _extract_transcript_text(transcript)
    transcript_payload = _to_serializable(transcript)

    transcript_dir = Path(project_dir) / "research" / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    stem = media_file.stem
    text_path = transcript_dir / f"{stem}.txt"
    json_path = transcript_dir / f"{stem}.json"

    text_path.write_text(transcript_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(transcript_payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    srt_path: Optional[Path] = None
    segments = _extract_segments(transcript_payload)
    if segments:
        srt_path = transcript_dir / f"{stem}.srt"
        srt_path.write_text(_build_srt_from_segments(segments), encoding="utf-8")

    logger.info("Saved transcript artifacts for %s", media_file)
    return TranscriptArtifacts(
        transcript_text_path=str(text_path),
        transcript_json_path=str(json_path),
        transcript_srt_path=str(srt_path) if srt_path else None,
        text=transcript_text,
    )


def _extract_transcript_text(transcript: Any) -> str:
    if isinstance(transcript, dict):
        return str(transcript.get("text", ""))

    text_value = getattr(transcript, "text", None)
    if isinstance(text_value, str):
        return text_value

    model_dump = getattr(transcript, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        if isinstance(payload, dict):
            return str(payload.get("text", ""))

    return str(transcript)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_serializable(model_dump())

    dict_value = getattr(value, "__dict__", None)
    if isinstance(dict_value, dict):
        return _to_serializable(dict_value)

    return str(value)


def _extract_segments(transcript_payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(transcript_payload, dict):
        return []
    segments = transcript_payload.get("segments")
    if isinstance(segments, list):
        return [item for item in segments if isinstance(item, dict)]
    return []


def _build_srt_from_segments(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        text = str(segment.get("text", "")).strip()

        lines.append(str(idx))
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_srt_time(value: float) -> str:
    total_ms = int(round(max(0.0, value) * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def estimate_tokens_conservative(text: str) -> int:
    """Conservative estimate that intentionally overestimates token usage."""
    if not text:
        return 0
    return max(1, int((len(text) / 3.2) + 32))


def summarize_if_needed(
    context: str,
    max_tokens: int,
    *,
    api_key: Optional[str] = None,
    base_url: str = NVIDIA_BASE_URL,
    model: str = DEFAULT_SUMMARY_MODEL,
) -> SummaryResult:
    """Summarize context only when estimated token count exceeds threshold."""
    token_estimate = estimate_tokens_conservative(context)
    if token_estimate <= max_tokens:
        return SummaryResult(
            context=context,
            was_summarized=False,
            estimated_tokens_before=token_estimate,
            estimated_tokens_after=token_estimate,
        )

    if not OPENAI_AVAILABLE:
        logger.warning("openai package not available; returning original context")
        return SummaryResult(
            context=context,
            was_summarized=False,
            estimated_tokens_before=token_estimate,
            estimated_tokens_after=token_estimate,
        )
    if OpenAI is None:
        logger.warning("OpenAI client unavailable; returning original context")
        return SummaryResult(
            context=context,
            was_summarized=False,
            estimated_tokens_before=token_estimate,
            estimated_tokens_after=token_estimate,
        )

    prompt = (
        "Summarize this research context for downstream planning. "
        "Preserve exact quotes, dates, URLs, and names exactly as written when included. "
        "Do not fabricate details.\n\n"
        "Return plain text with short sections: Facts, Claims, Gaps, Sources.\n\n"
        f"Context:\n{context}"
    )

    try:
        client = OpenAI(api_key=api_key or NVIDIA_API_KEY, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful research summarizer. Preserve exact entity strings "
                        "for names, dates, URLs, and direct quotes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max(200, int(max_tokens * 0.6)),
        )
        summarized = response.choices[0].message.content or context
    except Exception as exc:
        logger.error("Context summarization failed: %s", exc)
        summarized = context

    summarized_estimate = estimate_tokens_conservative(summarized)
    return SummaryResult(
        context=summarized,
        was_summarized=summarized != context,
        estimated_tokens_before=token_estimate,
        estimated_tokens_after=summarized_estimate,
    )


def build_state_machine_prompt(
    state: RunState,
    context: str,
) -> str:
    contracts = "\n\n".join(
        f"{phase.value}:\n{PHASE_JSON_CONTRACTS[phase]}" for phase in PipelinePhase
    )
    return STATE_MACHINE_PROMPT_TEMPLATE.format(
        current_phase=state.current_phase,
        project_name=state.project_name,
        run_id=state.run_id,
        context=context,
        contracts=contracts,
    )


__all__ = [
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_TRANSCRIBE_MODEL",
    "HackclubMediaSearchClient",
    "MediaSearchResult",
    "NVIDIA_BASE_URL",
    "PHASE_JSON_CONTRACTS",
    "PipelinePhase",
    "ResearchArtifact",
    "RunState",
    "STATE_MACHINE_PROMPT_TEMPLATE",
    "SummaryResult",
    "TranscriptArtifacts",
    "build_state_machine_prompt",
    "estimate_tokens_conservative",
    "load_run_state",
    "save_finding",
    "save_run_state",
    "set_phase",
    "setup_project_workspace",
    "summarize_if_needed",
    "transcribe_media_file",
]
