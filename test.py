#!/usr/bin/env python3
import argparse
import base64
import cv2
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from openai import APITimeoutError, NotFoundError, OpenAI

from src.hybrid_documentary_state_machine import (
    DEFAULT_SUMMARY_MODEL,
    HackclubMediaSearchClient,
    PipelinePhase,
    ResearchArtifact,
    RunState,
    load_run_state,
    save_finding,
    save_run_state,
    set_phase,
    setup_project_workspace,
    summarize_if_needed,
    transcribe_media_file,
)

# -----------------------------
# Config
# -----------------------------
NVIDIA_API_KEY = (
    os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY") or ""
).strip()
HACKCLUB_SEARCH_KEY = (
    os.getenv("HACKCLUB_SEARCH_KEY") or os.getenv("HACKCLUB_SEARCH_API_KEY") or ""
).strip()

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")
MODEL_PRIMARY = os.getenv("MODEL_QWEN", "qwen/qwen3.5-397b-a17b")
MODEL_FALLBACK = os.getenv("MODEL_KIMI", "qwen/qwen3.5-397b-a17b")
NVIDIA_CHAT_RETRIES = int(os.getenv("NVIDIA_CHAT_RETRIES", "3"))

SEARCH_COUNT = int(os.getenv("SEARCH_COUNT", "6"))
DOWNLOAD_MEDIA = os.getenv("DOWNLOAD_MEDIA", "1").strip() not in {"0", "false", "False"}
MAX_MEDIA_DOWNLOADS = int(os.getenv("MAX_MEDIA_DOWNLOADS", "8"))
MAX_MEDIA_FILE_SIZE_MB = int(os.getenv("MAX_MEDIA_FILE_SIZE_MB", "50"))
MEDIA_DOWNLOAD_TIMEOUT = int(os.getenv("MEDIA_DOWNLOAD_TIMEOUT", "20"))
TRANSCRIBE_LOCAL_MEDIA = os.getenv("TRANSCRIBE_LOCAL_MEDIA", "1").strip() not in {
    "0",
    "false",
    "False",
}
MAX_TRANSCRIBE_FILES = int(os.getenv("MAX_TRANSCRIBE_FILES", "6"))
MAX_SCRIPT_CONTEXT_TOKENS = int(os.getenv("MAX_SCRIPT_CONTEXT_TOKENS", "12000"))

LOG_FILE = os.getenv("LOG_FILE", "run_log.txt")


SCRIPT_SYSTEM_PROMPT = (
    "You are an elite YouTube Shorts documentary editor for hyper-niche internet/gaming history. "
    "Primary mode: fast, conversational internet-investigation voiceover. "
    "Guiding principle: INFORMATION DENSITY. "
    "1. In SCRIPTING, start Segment 1 narration with a compelling question hook, not a flat declarative line. "
    "2. Write like a live rabbit-hole investigation: use transitions such as 'you might think... but...' and 'when people checked...'. "
    "3. Every sentence must add a concrete receipt-backed detail (exact numbers, usernames, IDs, dates, posts, PDFs, quotes, timestamps). "
    "4. Prefer progression over summary: reveal how the community discovered each clue step-by-step, escalating stakes each beat. "
    "5. Avoid formal documentary phrasing and dry report tone. Keep wording sharp, human, and conversational while factual. "
    "6. Visuals dictate script: map each narration beat to specific evidence in provided context/assets. "
    "7. Create a perfect loop: final sentence must flow grammatically back into the first hook line. "
    "Return strict JSON only. Never invent facts, sources, dates, quotes, or numbers."
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def log_event(title: str, payload: Optional[Any] = None) -> None:
    logger = logging.getLogger("hybrid_test")
    if payload is None:
        logger.info(title)
        return
    if isinstance(payload, str):
        logger.info("%s %s", title, payload)
        return
    try:
        logger.info("%s %s", title, json.dumps(payload, ensure_ascii=True, default=str))
    except Exception:
        logger.info("%s %s", title, str(payload))


def die(message: str, code: int = 2) -> None:
    log_event("FATAL", {"message": message, "exit_code": code})
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


def safe_json_extract(text: str) -> Any | None:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None


def require_keys(payload: Dict[str, Any], required: List[str], context: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{context}: missing keys: {missing}")


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return cleaned or "file"


def infer_media_extension(url: str, content_type: str) -> str:
    lower_url = url.lower()
    if ".jpg" in lower_url or ".jpeg" in lower_url:
        return ".jpg"
    if ".png" in lower_url:
        return ".png"
    if ".gif" in lower_url:
        return ".gif"
    if ".webp" in lower_url:
        return ".webp"
    if ".mp4" in lower_url:
        return ".mp4"
    if ".webm" in lower_url:
        return ".webm"
    if ".mov" in lower_url:
        return ".mov"
    if content_type.startswith("image/"):
        subtype = content_type.split("/", 1)[1].split(";", 1)[0]
        return f".{subtype or 'jpg'}"
    if content_type.startswith("video/"):
        subtype = content_type.split("/", 1)[1].split(";", 1)[0]
        return f".{subtype or 'mp4'}"
    return ".bin"


def as_phase(value: str) -> PipelinePhase:
    try:
        return PipelinePhase(value)
    except ValueError as exc:
        valid = ", ".join(phase.value for phase in PipelinePhase)
        raise argparse.ArgumentTypeError(
            f"Invalid phase '{value}'. Use one of: {valid}"
        ) from exc


def nvidia_chat(
    model: str, messages: List[Dict[str, str]], temperature: float = 0.4
) -> str:
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    retries = max(1, NVIDIA_CHAT_RETRIES)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            log_event(
                "NVIDIA_CHAT_REQUEST",
                {
                    "attempt": attempt + 1,
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                },
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=2200,
                timeout=120,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            log_event(
                "NVIDIA_CHAT_RESPONSE",
                {"attempt": attempt + 1, "model": model, "length": len(content)},
            )
            return content
        except APITimeoutError as exc:
            last_err = exc
            log_event(
                "NVIDIA_CHAT_TIMEOUT", {"attempt": attempt + 1, "error": str(exc)}
            )
        except NotFoundError as exc:
            raise RuntimeError(f"Model not found: {model}: {exc}") from exc
        except Exception as exc:
            last_err = exc
            log_event("NVIDIA_CHAT_ERROR", {"attempt": attempt + 1, "error": str(exc)})
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NVIDIA chat failed for model {model}: {last_err}")


def chat_with_fallback(
    model: str,
    fallback_model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
    try:
        return nvidia_chat(model, messages, temperature=temperature)
    except Exception as first_err:
        if not fallback_model or fallback_model == model:
            raise
        log_event(
            "MODEL_FALLBACK",
            {"from_model": model, "to_model": fallback_model, "reason": str(first_err)},
        )
        return nvidia_chat(fallback_model, messages, temperature=temperature)


def chat_json(
    user_prompt: str,
    *,
    validator: Callable[[Dict[str, Any]], None],
    temperature: float = 0.4,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = chat_with_fallback(MODEL_PRIMARY, MODEL_FALLBACK, messages, temperature)
    payload = safe_json_extract(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Model did not return valid JSON object")
    validator(payload)
    return payload


def add_artifact_to_state(state: RunState, path: Path, folder: str) -> None:
    relative = path.relative_to(Path(state.project_dir))
    artifact = ResearchArtifact(
        folder=folder,
        filename=path.name,
        relative_path=str(relative),
        created_at=str(int(path.stat().st_mtime)),
        byte_size=path.stat().st_size,
    )
    state.findings.append(artifact)
    save_run_state(state)


async def phase_idea_generation(state: RunState) -> None:
    print("AI is planning a scouting mission for niche lore...")

    # STEP 1: AI generates its own search queries
    planning_prompt = """
    You are a Niche Lore Scout. Your goal is to find obscure, high-stakes internet mysteries
    or gaming scandals that have "receipts" (archived forum posts, player IDs, specific dates).

    Generate 5 search queries that target specific places where "lore" is documented but not widely known.
    Think: "removed speedrun world record", "Geometry Dash impossible level controversy",
    "deleted forum threads internet mystery", "site:speedrun.com/forums controversy".

    Return strict JSON: {"scouting_queries": ["query 1", "query 2", ...]}
    """

    planning_raw = chat_json(
        planning_prompt, validator=lambda x: "scouting_queries" in x
    )
    scouting_queries = planning_raw["scouting_queries"]
    log_event("AI_PLANNED_QUERIES", scouting_queries)

    # STEP 2: Execute the AI's queries
    search_context = ""
    async with HackclubMediaSearchClient() as client:
        for q in scouting_queries:
            print(f"Searching: {q}")
            hits = await client.search_web(q, count=4)
            search_context += f"\n--- RESULTS FOR QUERY: {q} ---\n"
            for hit in hits:
                search_context += f"TITLE: {hit.title}\nURL: {hit.url}\nSNIPPET: {hit.description}\n\n"

    # Save the raw data for reference
    save_finding(state.project_dir, "ideas", "raw_scout_data.txt", search_context)

    # STEP 3: Summarize Context if it's huge (using your 120k threshold)
    # Note: summarize_if_needed is imported from your state machine
    processed_context = summarize_if_needed(
        search_context,
        max_tokens=120000,
        api_key=NVIDIA_API_KEY,
        model="qwen/qwen3.5-397b-a17b",  # Or your gpt-oss-120b equivalent
    ).context

    # STEP 4: Generate the 3 Mid-Tier Angles + the Master Gemini Prompt
    ideation_prompt = f"""
    Based on this scouted web data, find 3 HIGH-STAKES, NICHE story angles.

    CONTEXT DATA:
    {processed_context}

    CRITICAL:
    1. Avoid mainstream topics.
    2. Focus on "The First X", "The Deleted Y", "The Impossible Z".
    3. For the 'gemini_deep_research_prompt', write a prompt that forces Gemini to
       be a forensic archivist. It must find the EXACT timestamps, forum post authors,
       and archived links in the provided context to build a bulletproof timeline.

    Return strict JSON:
    {{
      "phase": "IDEA_GENERATION",
      "angles": [
        {{"id": "A1", "title": "...", "hook": "...", "scouted_evidence_summary": "..."}}
      ],
      "gemini_deep_research_prompt": "..."
    }}
    """

    def _validate_ideation(payload):
        if len(payload.get("angles", [])) != 3:
            raise ValueError("Need 3 angles")

    final_ideas = chat_json(ideation_prompt, validator=_validate_ideation)

    # Save results to project folder
    saved_path = save_finding(
        state.project_dir, "ideas", "idea_generation.json", final_ideas
    )
    add_artifact_to_state(state, saved_path, "ideas")

    # Update State
    state.metadata["idea_generation_path"] = str(saved_path)
    state.context_snapshot = final_ideas.get("gemini_deep_research_prompt", "")
    save_run_state(state)
    set_phase(state, PipelinePhase.WAIT_FOR_GEMINI_REPORT, "Idea package generated")
    log_event("IDEA_GENERATION_DONE", {"path": str(saved_path)})

    print(f"Scouting complete. Ideas saved to {saved_path}. Check the Gemini prompt!")


def print_waiting_instructions(project_dir: Path, state: RunState) -> None:
    report_dir = project_dir / "research" / "reports"
    print("\n=== WAIT_FOR_GEMINI_REPORT ===")
    print("1) Run Gemini Deep Research using prompt from:")
    print(f"   {project_dir / 'research' / 'ideas' / 'idea_generation.json'}")
    print("2) Save Gemini report as plain text or markdown under:")
    print(f"   {report_dir}")
    print("3) Resume with:")
    print(
        f'   python3 test.py --project "{state.project_name}" --resume --gemini-report "{report_dir / "gemini_report.txt"}"'
    )


def phase_wait_for_gemini_report(
    state: RunState, gemini_report_path: Optional[str]
) -> bool:
    report_path = gemini_report_path or state.gemini_report_path
    if not report_path:
        save_run_state(state)
        print_waiting_instructions(Path(state.project_dir), state)
        return False

    report_file = Path(report_path)
    if not report_file.exists() or not report_file.is_file():
        save_run_state(state)
        print_waiting_instructions(Path(state.project_dir), state)
        print(f"\nProvided report path not found: {report_file}", file=sys.stderr)
        return False

    text = report_file.read_text(encoding="utf-8", errors="replace")
    copied = save_finding(state.project_dir, "reports", "gemini_report.txt", text)
    add_artifact_to_state(state, copied, "reports")
    state.gemini_report_path = str(copied)
    save_run_state(state)
    set_phase(state, PipelinePhase.SYNTHESIS, "Gemini report supplied")
    log_event("GEMINI_REPORT_READY", {"path": str(copied), "chars": len(text)})
    return True


def phase_synthesis(state: RunState) -> None:
    if not state.gemini_report_path:
        set_phase(state, PipelinePhase.WAIT_FOR_GEMINI_REPORT, "Missing report path")
        return

    report_text = Path(state.gemini_report_path).read_text(
        encoding="utf-8", errors="replace"
    )
    ideas_path = state.metadata.get("idea_generation_path", "")
    ideas_text = ""
    if ideas_path and Path(ideas_path).exists():
        ideas_text = Path(ideas_path).read_text(encoding="utf-8", errors="replace")

    prompt = (
        "Phase: SYNTHESIS\n"
        "Choose one idea angle and convert Gemini report into concrete evidence/search plan.\n"
        "Return strict JSON with keys:\n"
        "{\n"
        '  "phase": "SYNTHESIS",\n'
        '  "chosen_angle_id": "A1|A2|A3",\n'
        '  "chosen_angle_title": "...",\n'
        '  "reason": "...",\n'
        '  "image_queries": ["..."],\n'
        '  "video_queries": ["..."],\n'
        '  "evidence_questions": ["..."],\n'
        '  "next_phase": "EVIDENCE_GATHERING"\n'
        "}\n"
        "Provide at least 3 image queries and 3 video queries.\n\n"
        f"Idea JSON:\n{ideas_text}\n\nGemini report raw text:\n{report_text}"
    )

    def _validate(payload: Dict[str, Any]) -> None:
        require_keys(
            payload,
            [
                "phase",
                "chosen_angle_id",
                "chosen_angle_title",
                "reason",
                "image_queries",
                "video_queries",
                "evidence_questions",
                "next_phase",
            ],
            "SYNTHESIS",
        )
        if payload.get("phase") != "SYNTHESIS":
            raise ValueError("SYNTHESIS: invalid phase")
        if payload.get("chosen_angle_id") not in {"A1", "A2", "A3"}:
            raise ValueError("SYNTHESIS: chosen_angle_id must be A1/A2/A3")
        image_queries = payload.get("image_queries")
        video_queries = payload.get("video_queries")
        if not isinstance(image_queries, list) or len(image_queries) < 3:
            raise ValueError("SYNTHESIS: image_queries must have at least 3 entries")
        if not isinstance(video_queries, list) or len(video_queries) < 3:
            raise ValueError("SYNTHESIS: video_queries must have at least 3 entries")

    payload = chat_json(prompt, validator=_validate, temperature=0.3)
    saved_path = save_finding(state.project_dir, "synthesis", "synthesis.json", payload)
    add_artifact_to_state(state, saved_path, "synthesis")
    state.metadata["synthesis_path"] = str(saved_path)
    save_run_state(state)
    set_phase(
        state, PipelinePhase.EVIDENCE_GATHERING, "Synthesis produced media queries"
    )
    log_event("SYNTHESIS_DONE", {"path": str(saved_path)})


async def run_media_queries(
    image_queries: List[str],
    video_queries: List[str],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"image": {}, "video": {}, "web": {}}
    async with HackclubMediaSearchClient(api_key=HACKCLUB_SEARCH_KEY) as client:
        for query in image_queries:
            image_hits = await client.search_images(query, count=SEARCH_COUNT)
            web_hits = await client.search_web(query, count=min(SEARCH_COUNT, 4))
            results["image"][query] = [hit.__dict__ for hit in image_hits]
            results["web"][query] = [hit.__dict__ for hit in web_hits]
        for query in video_queries:
            video_hits = await client.search_videos(query, count=SEARCH_COUNT)
            web_hits = await client.search_web(query, count=min(SEARCH_COUNT, 4))
            results["video"][query] = [hit.__dict__ for hit in video_hits]
            if query in results["web"]:
                results["web"][query].extend([hit.__dict__ for hit in web_hits])
            else:
                results["web"][query] = [hit.__dict__ for hit in web_hits]
    return results


def maybe_download_media(state: RunState, search_payload: Dict[str, Any]) -> List[str]:
    if not DOWNLOAD_MEDIA:
        return []

    download_targets: List[tuple[str, str]] = []
    for query, hits in (search_payload.get("image", {}) or {}).items():
        if isinstance(hits, list):
            for item in hits[:2]:
                if isinstance(item, dict) and item.get("url"):
                    download_targets.append((str(item["url"]), "images"))
    for query, hits in (search_payload.get("video", {}) or {}).items():
        if isinstance(hits, list):
            for item in hits[:2]:
                if isinstance(item, dict) and item.get("url"):
                    download_targets.append((str(item["url"]), "videos"))

    downloaded_paths: List[str] = []
    for idx, (url, media_kind) in enumerate(
        download_targets[:MAX_MEDIA_DOWNLOADS], start=1
    ):
        try:
            response = requests.get(url, timeout=MEDIA_DOWNLOAD_TIMEOUT, stream=True)
            if response.status_code >= 400:
                continue

            content_length = response.headers.get("content-length")
            if content_length:
                file_size_mb = int(content_length) / (1024 * 1024)
                if file_size_mb > MAX_MEDIA_FILE_SIZE_MB:
                    log_event(
                        "MEDIA_SKIP_TOO_LARGE",
                        {
                            "url": url,
                            "size_mb": round(file_size_mb, 2),
                            "limit_mb": MAX_MEDIA_FILE_SIZE_MB,
                        },
                    )
                    continue

            content_type = (response.headers.get("content-type") or "").lower()
            extension = infer_media_extension(url, content_type)
            if media_kind == "images" and not (
                extension in {".jpg", ".jpeg", ".png", ".gif", ".webp"}
                or content_type.startswith("image/")
            ):
                continue
            if media_kind == "videos" and not (
                extension in {".mp4", ".webm", ".mov"}
                or content_type.startswith("video/")
            ):
                continue

            filename = f"{idx:03d}-{sanitize_filename(media_kind)}{extension}"
            target = (
                Path(state.project_dir) / "research" / "media" / media_kind / filename
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            add_artifact_to_state(state, target, f"media/{media_kind}")
            downloaded_paths.append(str(target.relative_to(Path(state.project_dir))))
        except Exception as exc:
            log_event("MEDIA_DOWNLOAD_ERROR", {"url": url, "error": str(exc)})
    return downloaded_paths


def maybe_transcribe_local_media(state: RunState) -> List[Dict[str, Any]]:
    if not TRANSCRIBE_LOCAL_MEDIA:
        return []
    if not NVIDIA_API_KEY:
        log_event("TRANSCRIBE_SKIPPED", "Missing NVIDIA API key")
        return []

    media_root = Path(state.project_dir) / "research" / "media"
    if not media_root.exists():
        return []

    candidates: List[Path] = []
    extensions = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".mov", ".mkv"}
    for path in media_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            candidates.append(path)

    outputs: List[Dict[str, Any]] = []
    for media_path in candidates[:MAX_TRANSCRIBE_FILES]:
        try:
            transcript = transcribe_media_file(
                media_path, state.project_dir, api_key=NVIDIA_API_KEY
            )
            outputs.append(
                {
                    "media": str(media_path.relative_to(Path(state.project_dir))),
                    "transcript_text_path": transcript.transcript_text_path,
                    "transcript_json_path": transcript.transcript_json_path,
                    "transcript_srt_path": transcript.transcript_srt_path,
                }
            )
            add_artifact_to_state(
                state, Path(transcript.transcript_text_path), "transcripts"
            )
            add_artifact_to_state(
                state, Path(transcript.transcript_json_path), "transcripts"
            )
            if transcript.transcript_srt_path:
                add_artifact_to_state(
                    state, Path(transcript.transcript_srt_path), "transcripts"
                )
        except Exception as exc:
            log_event("TRANSCRIBE_ERROR", {"media": str(media_path), "error": str(exc)})
    return outputs


def phase_evidence_gathering(state: RunState) -> None:
    synthesis_path = state.metadata.get("synthesis_path", "")
    if not synthesis_path or not Path(synthesis_path).exists():
        raise RuntimeError("Synthesis artifact missing; cannot run evidence phase")

    synthesis_payload = json.loads(Path(synthesis_path).read_text(encoding="utf-8"))
    image_queries = [
        str(item).strip()
        for item in synthesis_payload.get("image_queries", [])
        if str(item).strip()
    ]
    video_queries = [
        str(item).strip()
        for item in synthesis_payload.get("video_queries", [])
        if str(item).strip()
    ]
    if not image_queries and not video_queries:
        raise RuntimeError("No queries found in synthesis payload")

    search_payload = asyncio.run(run_media_queries(image_queries, video_queries))
    search_path = save_finding(
        state.project_dir, "evidence", "media_search_results.json", search_payload
    )
    add_artifact_to_state(state, search_path, "evidence")

    downloaded = maybe_download_media(state, search_payload)
    transcripts = maybe_transcribe_local_media(state)

    evidence_index = {
        "phase": "EVIDENCE_GATHERING",
        "media_search_results_path": str(search_path),
        "downloaded_media": downloaded,
        "transcripts": transcripts,
    }
    evidence_index_path = save_finding(
        state.project_dir, "evidence", "evidence_index.json", evidence_index
    )
    add_artifact_to_state(state, evidence_index_path, "evidence")

    state.metadata["evidence_search_path"] = str(search_path)
    state.metadata["evidence_index_path"] = str(evidence_index_path)
    save_run_state(state)
    set_phase(state, PipelinePhase.SCRIPTING, "Evidence artifacts captured")
    log_event("EVIDENCE_GATHERING_DONE", {"index": str(evidence_index_path)})


def gather_workspace_assets(state: RunState) -> List[str]:
    base = Path(state.project_dir)
    assets: List[str] = []
    for sub in [
        "research/media_images",
        "research/media_videos",
        "research/media/images",
        "research/media/videos",
        "research/transcripts",
        "research/evidence",
    ]:
        root = base / sub
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                assets.append(str(path.relative_to(base)))
    assets.sort()
    return assets


def remap_visual_asset_paths(script_payload: Dict[str, Any], assets: List[str], project_dir: str) -> None:
    if not assets:
        return
    segments = script_payload.get("segments",[])
    if not isinstance(segments, list):
        return

    # Use Qwen VLM to intelligently map assets to segments
    if NVIDIA_API_KEY:
        log_event("VLM_MAPPING", "Using Qwen VLM to map visual assets to script segments...")
        content_items =[
            {"type": "text", "text": "Map each script segment to the most visually appropriate asset from the provided images. Return ONLY a JSON object with a 'mapping' list containing objects with 'segment_index' and 'asset_index'."}
        ]

        valid_assets =[]
        for idx, asset in enumerate(assets):
            full_path = Path(project_dir) / asset
            if not full_path.exists(): continue

            try:
                img = None
                if full_path.suffix.lower() in['.mp4', '.webm', '.mov']:
                    cap = cv2.VideoCapture(str(full_path))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        img = frame
                else:
                    img = cv2.imread(str(full_path))

                if img is not None:
                    # resize to save tokens
                    img = cv2.resize(img, (512, 512))
                    _, buffer = cv2.imencode('.jpg', img)
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    content_items.append({
                        "type": "text",
                        "text": f"Asset {len(valid_assets)}: {asset}"
                    })
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                    valid_assets.append(asset)
            except Exception as e:
                log_event("VLM_ASSET_LOAD_ERROR", f"Failed to process asset {asset}: {e}")

        if valid_assets:
            seg_text = "\n".join([f"Segment {i}: {s.get('narration', '')} | Directive: {s.get('visual_directive', '')}" for i, s in enumerate(segments)])
            content_items.append({"type": "text", "text": f"Script Segments:\n{seg_text}\n\nReturn JSON format: {{\"mapping\": [{{\"segment_index\": 0, \"asset_index\": 2}}]}}"})

            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
            try:
                response = client.chat.completions.create(
                    model=MODEL_PRIMARY,
                    messages=[{"role": "user", "content": content_items}],
                    temperature=0.2,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                res_json = json.loads(response.choices[0].message.content)
                mapping = res_json.get("mapping",[])

                # Apply mapping
                for m in mapping:
                    s_idx = m.get("segment_index")
                    a_idx = m.get("asset_index")
                    if s_idx is not None and a_idx is not None:
                        if 0 <= s_idx < len(segments) and 0 <= a_idx < len(valid_assets):
                            segments[s_idx]["visual_asset_path"] = valid_assets[a_idx]

                # Fill any remaining missing with fallback
                for idx, seg in enumerate(segments):
                    path = str(seg.get("visual_asset_path", "")).strip()
                    if not path or path not in valid_assets:
                        seg["visual_asset_path"] = valid_assets[idx % len(valid_assets)]

                log_event("VLM_MAPPING_SUCCESS", "Assets successfully mapped via VLM.")
                return
            except Exception as e:
                log_event("VLM_MAPPING_FAILED", str(e))

    # Fallback if VLM fails or no API key
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        path = str(segment.get("visual_asset_path", "")).strip()
        if path and path in assets:
            continue
        segment["visual_asset_path"] = assets[idx % len(assets)]


def phase_scripting(state: RunState) -> Dict[str, Any]:
    idea_path = state.metadata.get("idea_generation_path", "")
    synthesis_path = state.metadata.get("synthesis_path", "")
    evidence_path = state.metadata.get("evidence_search_path", "")
    report_path = state.gemini_report_path or ""

    blocks: List[str] = []
    for label, path in [
        ("IDEAS", idea_path),
        ("GEMINI_REPORT", report_path),
        ("SYNTHESIS", synthesis_path),
        ("EVIDENCE", evidence_path),
    ]:
        if path and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            blocks.append(f"[{label}]\n{text}")
    context = "\n\n".join(blocks)

    summary_result = summarize_if_needed(
        context,
        max_tokens=MAX_SCRIPT_CONTEXT_TOKENS,
        api_key=NVIDIA_API_KEY or None,
        model=DEFAULT_SUMMARY_MODEL,
    )
    summary_path = save_finding(
        state.project_dir, "summaries", "script_context.txt", summary_result.context
    )
    add_artifact_to_state(state, summary_path, "summaries")

    assets = gather_workspace_assets(state)
    assets_json = json.dumps(assets, indent=2, ensure_ascii=True)
    prompt = (
        "Phase: SCRIPTING\n"
        "Write final 45-60 second investigative speedrunning short script from context.\n"
        "Return strict JSON with keys:\n"
        "{\n"
        '  "phase": "SCRIPTING",\n'
        '  "title": "...",\n'
        '  "hook": "...",\n'
        '  "script_full": "...",\n'
        '  "loop_line": "...",\n'
        '  "segments": [\n'
        "    {\n"
        '      "segment_id": "S1",\n'
        '      "time_seconds": 0,\n'
        '      "intended_duration_seconds": 8,\n'
        '      "narration": "...",\n'
        '      "on_screen_text": "...",\n'
        '      "visual_search_query": "...",\n'
        '      "visual_asset_path": "research/media_images/...",\n'
        '      "evidence_refs": ["..."],\n'
        '      "pace": "fast|normal|slow",\n'
        '      "emotion": "excited|calm|dramatic|neutral"\n'
        "    }\n"
        "  ],\n"
        '  "sources_to_check": ["..."],\n'
        '  "hashtags": ["..."]\n'
        "}\n"
        "Constraints: 6-10 segments, total intended duration 45-60 seconds.\n"
        "visual_asset_path must reference one of these existing findings workspace paths:\n"
        f"{assets_json}\n\n"
        f"Context:\n{summary_result.context}"
    )

    def _validate(payload: Dict[str, Any]) -> None:
        require_keys(
            payload,
            [
                "phase",
                "title",
                "hook",
                "script_full",
                "loop_line",
                "segments",
                "sources_to_check",
                "hashtags",
            ],
            "SCRIPTING",
        )
        if payload.get("phase") != "SCRIPTING":
            raise ValueError("SCRIPTING: invalid phase")
        segments = payload.get("segments")
        if not isinstance(segments, list) or not (6 <= len(segments) <= 10):
            raise ValueError("SCRIPTING: segments must contain 6-10 entries")
        duration = 0.0
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("SCRIPTING: each segment must be an object")
            require_keys(
                segment,
                [
                    "segment_id",
                    "time_seconds",
                    "intended_duration_seconds",
                    "narration",
                    "on_screen_text",
                    "visual_search_query",
                    "visual_asset_path",
                    "evidence_refs",
                    "pace",
                    "emotion",
                ],
                "SCRIPTING.segment",
            )
            duration += float(segment.get("intended_duration_seconds", 0))
        if duration < 45 or duration > 60:
            raise ValueError("SCRIPTING: segment durations must total 45-60 seconds")

    payload = chat_json(prompt, validator=_validate, temperature=0.4)
    remap_visual_asset_paths(payload, assets, state.project_dir)

    script_path = save_finding(
        state.project_dir, "scripts", "final_script.json", payload
    )
    add_artifact_to_state(state, script_path, "scripts")

    state.metadata["final_script_path"] = str(script_path)
    state.metadata["summary_result"] = summary_result.model_dump()
    state.status = "completed"
    save_run_state(state)
    log_event("SCRIPTING_DONE", {"final_script_path": str(script_path)})
    return payload


def resolve_project_name(args: argparse.Namespace) -> str:
    candidate = (
        args.project
        or args.project_positional
        or os.getenv("PROJECT_NAME")
        or os.getenv("PROJECT_ID")
    )
    if not candidate:
        die("Project not provided. Use --project or set PROJECT_NAME/PROJECT_ID.")
    return candidate.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid pause/resume documentary workflow driver"
    )
    parser.add_argument(
        "project_positional", nargs="?", help="Project name/id (optional positional)"
    )
    parser.add_argument("--project", help="Project name/id")
    parser.add_argument("--resume", action="store_true", help="Resume from state.json")
    parser.add_argument(
        "--phase",
        type=as_phase,
        help="Optional phase override",
    )
    parser.add_argument("--gemini-report", help="Path to Gemini report text/markdown")
    return parser.parse_args()


def initialize_state(project_name: str, resume: bool) -> RunState:
    project_dir = setup_project_workspace(project_name)
    state = load_run_state(project_dir)
    if not resume and state.status == "completed":
        state.status = "active"
        state.current_phase = PipelinePhase.IDEA_GENERATION
        save_run_state(state)
        log_event(
            "STATE_RESET",
            {"project": project_name, "reason": "completed run reset without --resume"},
        )
    return state


def run_workflow(
    state: RunState, gemini_report_path: Optional[str]
) -> Optional[Dict[str, Any]]:
    while True:
        phase = state.current_phase
        log_event("PHASE_START", {"phase": phase, "project": state.project_name})

        if phase == PipelinePhase.IDEA_GENERATION:
            asyncio.run(phase_idea_generation(state))
            continue

        if phase == PipelinePhase.WAIT_FOR_GEMINI_REPORT:
            is_ready = phase_wait_for_gemini_report(state, gemini_report_path)
            if not is_ready:
                return None
            gemini_report_path = None
            continue

        if phase == PipelinePhase.SYNTHESIS:
            phase_synthesis(state)
            continue

        if phase == PipelinePhase.EVIDENCE_GATHERING:
            phase_evidence_gathering(state)
            continue

        if phase == PipelinePhase.SCRIPTING:
            return phase_scripting(state)

        raise RuntimeError(f"Unhandled phase: {phase}")


def main() -> None:
    setup_logging()
    args = parse_args()

    project_name = resolve_project_name(args)
    state = initialize_state(project_name, resume=args.resume)

    if args.phase and args.phase != state.current_phase:
        state = set_phase(state, args.phase, "Manual phase override from CLI")

    if not NVIDIA_API_KEY:
        die("Set NVIDIA_API_KEY or NVIDIA_NIM_API_KEY.")

    if not HACKCLUB_SEARCH_KEY:
        log_event(
            "WARN",
            "HACKCLUB_SEARCH_KEY/HACKCLUB_SEARCH_API_KEY missing: evidence search may be empty",
        )

    log_event(
        "RUN_CONFIG",
        {
            "project": project_name,
            "resume": args.resume,
            "phase_override": args.phase.value if args.phase else None,
            "gemini_report": args.gemini_report,
            "nvidia_base_url": NVIDIA_BASE_URL,
            "model_primary": MODEL_PRIMARY,
            "model_fallback": MODEL_FALLBACK,
            "search_count": SEARCH_COUNT,
            "download_media": DOWNLOAD_MEDIA,
            "transcribe_local_media": TRANSCRIBE_LOCAL_MEDIA,
            "log_file": LOG_FILE,
        },
    )

    final_payload = run_workflow(state, args.gemini_report)
    if final_payload is not None:
        print(json.dumps(final_payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
