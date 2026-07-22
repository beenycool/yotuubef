"""
Enhanced AI-Powered Video Generation Orchestrator
Integrates all advanced features: cinematic editing, advanced audio, thumbnail optimization,
performance tracking, and proactive channel management.
"""

import asyncio
import ast
import base64
import json
import logging
import mimetypes
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlparse, urlunparse

import aiohttp
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

from src.config.settings import get_config
from src.models import (
    EmotionType,
    NarrativeSegment,
    PacingType,
    PositionType,
    TextOverlay,
    TextStyle,
)
from src.integrations.search_client import (
    DeepResearchClient,
    AgenticResearcher,
    AgenticExaResearcher,
    ExaSearchClient,
)
from src.integrations.ai_client import AIClient
from src.integrations.youtube_client import YouTubeClient
from src.processing.advanced_audio_processor import AdvancedAudioProcessor
from src.utils.gpu_memory_manager import GPUMemoryManager
from src.processing.video_processor import VideoProcessor
from src.processing.background_manager import BackgroundManager
from src.processing.video_processor_fixes import MoviePyCompat, ensure_shorts_format
from src.hybrid_documentary_state_machine import (
    PipelinePhase,
    build_state_machine_prompt,
    build_validation_feedback_prompt,
    load_run_state,
    save_finding,
    save_run_state,
    set_phase,
    setup_project_workspace,
    summarize_if_needed,
    transcribe_media_file,
    validate_phase_output,
)
from src.utils.search_audit_logger import SearchAuditLogger
from src.processing.asset_vector_store import AssetVectorStore
from src.processing.cinematic_editor import CinematicEditor


class EnhancedVideoOrchestrator:
    """
    Enhanced orchestrator with AI-powered cinematic editing, advanced audio processing,
    intelligent thumbnail optimization, and proactive channel management.
    """

    def __init__(self):
        # Basic setup
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.ai_client = AIClient()
        self.youtube_client = YouTubeClient()
        self.video_processor = VideoProcessor()

        self.advanced_audio_processor = AdvancedAudioProcessor()
        self.gpu_manager = GPUMemoryManager(max_vram_usage=0.85)

        # Local vector store for asset dedup + semantic retrieval
        self.asset_store = AssetVectorStore()

        # AI-powered Cinematic Editor for video scene analysis and pacing adjustment
        self.cinematic_editor = CinematicEditor()

        self.logger.info("Enhanced AI-powered video orchestrator initialized")

    def analyze_rendered_video_cinematically(
        self, video_path: Path, analysis: Optional[Any] = None
    ) -> Any:
        """
        Perform cinematic scene analysis on a video using CinematicEditor.
        """
        cinematic_editor = getattr(self, "cinematic_editor", None)
        if cinematic_editor is not None and hasattr(
            cinematic_editor, "analyze_video_cinematically"
        ):
            if analysis is None:
                from src.models import VideoAnalysisEnhanced

                analysis = VideoAnalysisEnhanced(
                    topic="Cinematic Analysis",
                    target_audience="General",
                    key_themes=["Documentary"],
                    recommended_style="Cinematic",
                    suggested_music_genre="Dramatic",
                )
            return cinematic_editor.analyze_video_cinematically(video_path, analysis)
        return analysis

    def adjust_cinematic_pacing(
        self, segments: List[Dict[str, Any]], target_style: str = "dynamic"
    ) -> List[Dict[str, Any]]:
        """
        Adjust segment pacing using CinematicEditor.
        """
        cinematic_editor = getattr(self, "cinematic_editor", None)
        if cinematic_editor is not None and hasattr(cinematic_editor, "adjust_pacing"):
            return cinematic_editor.adjust_pacing(segments, target_style=target_style)
        return segments

    async def process_hybrid_documentary_studio(
        self,
        project_name: str,
        reddit_url: Optional[str] = None,
        resume: bool = False,
        phase_override: Optional[str] = None,
        gemini_report_path: Optional[str] = None,
        no_upload: bool = False,
        no_auto_research: bool = False,
    ) -> Dict[str, Any]:
        """Run opt-in hybrid documentary workflow with human pause/resume."""
        try:
            self.logger.info(
                "Starting hybrid documentary workflow project=%s resume=%s",
                project_name,
                resume,
            )
            project_dir = setup_project_workspace(project_name)
            state = load_run_state(project_dir)

            if not resume:
                curr_phase = self._coerce_hybrid_phase(state.current_phase)
                needs_reset = state.status == "completed"
                if curr_phase == PipelinePhase.VIDEO_RENDER and not state.metadata.get(
                    "final_script_path"
                ):
                    needs_reset = True
                if needs_reset:
                    self.logger.info(
                        "Resetting hybrid run for project=%s (status=%s phase=%s)",
                        project_name,
                        state.status,
                        state.current_phase,
                    )
                    self._reset_hybrid_run_state(state)
                    save_run_state(state)

            if reddit_url:
                state.metadata["reddit_url"] = reddit_url
                if not state.context_snapshot:
                    state.context_snapshot = f"Reddit URL: {reddit_url}"

            state.metadata["hybrid_no_upload"] = bool(no_upload)
            state.metadata["hybrid_no_auto_research"] = bool(no_auto_research)

            if phase_override:
                if getattr(state, "status", "") == "paused_for_script_review":
                    state.status = "active"
                    save_run_state(state)
                target_phase = self._coerce_hybrid_phase(phase_override)
                current_phase = self._coerce_hybrid_phase(state.current_phase)
                if target_phase and target_phase != current_phase:
                    state = set_phase(
                        state, target_phase, "Manual hybrid phase override"
                    )

            workflow_result = await self._run_hybrid_state_machine(
                state,
                gemini_report_path=gemini_report_path,
                no_auto_research=no_auto_research,
            )
            workflow_result["workspace_path"] = str(project_dir)
            workflow_result["state_path"] = str(
                Path(project_dir) / "research" / "state.json"
            )
            if state.metadata.get("search_audit_path"):
                workflow_result["search_audit_path"] = state.metadata[
                    "search_audit_path"
                ]
            if state.metadata.get("deep_research_prompt_path"):
                workflow_result["deep_research_prompt_path"] = state.metadata[
                    "deep_research_prompt_path"
                ]
            return workflow_result
        except Exception as exc:
            self.logger.error(
                "Hybrid documentary workflow failed: %s", exc, exc_info=True
            )
            return {
                "success": False,
                "error": str(exc),
                "pipeline": "hybrid_documentary_studio",
                "project_name": project_name,
            }

    @staticmethod
    def _reset_hybrid_run_state(state: Any) -> None:
        """Clear stale downstream artifacts before restarting a completed run."""
        state.status = "active"
        state.current_phase = PipelinePhase.IDEA_GENERATION
        state.gemini_report_path = None

        metadata = getattr(state, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            state.metadata = metadata

        for key in (
            "synthesis_path",
            "evidence_index_path",
            "evidence_search_path",
            "raw_media_dir",
            "final_script_path",
            "final_video_path",
            "render_manifest_path",
        ):
            metadata.pop(key, None)

        # Wipe research subfolders and raw_media from disk
        research = Path(state.project_dir) / "research"
        for sub in (
            "synthesis",
            "evidence",
            "scripts",
            "summaries",
            "transcripts",
            "media_images",
            "media_videos",
            "reports",
        ):
            d = research / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
        raw = Path(state.project_dir) / "raw_media"
        if raw.exists():
            shutil.rmtree(raw, ignore_errors=True)
            raw.mkdir(parents=True, exist_ok=True)

    async def _handle_idea_generation_phase(self, state: Any) -> Any:
        idea_payload, audit_path = await self._run_idea_generation_search_first(
            state,
        )
        idea_payload = self._prepare_idea_generation_payload(
            idea_payload,
            project_name=state.project_name,
        )
        idea_path = save_finding(
            state.project_dir,
            "ideas",
            "idea_generation.json",
            idea_payload,
        )
        prompt_path = self._save_deep_research_prompt_artifact(
            state,
            idea_payload,
        )
        state.metadata["idea_generation_path"] = str(idea_path)
        if prompt_path:
            state.metadata["deep_research_prompt_path"] = str(prompt_path)
            print(
                f"[Hybrid] Deep research prompt saved: {prompt_path}",
                flush=True,
            )
        if audit_path:
            state.metadata["search_audit_path"] = str(audit_path)
        state.context_snapshot = json.dumps(idea_payload, ensure_ascii=True)
        save_run_state(state)
        return set_phase(
            state,
            PipelinePhase.WAIT_FOR_GEMINI_REPORT,
            "Idea package generated",
        )

    def _phase_result(self, state, **overrides) -> Dict[str, Any]:
        base = {
            "success": False,
            "paused": True,
            "status": state.status,
            "current_phase": state.current_phase.value
            if hasattr(state.current_phase, "value")
            else state.current_phase,
            "project_name": state.project_name,
            "pipeline": "hybrid_documentary_studio",
            "no_upload": bool(state.metadata.get("hybrid_no_upload", False)),
        }
        base.update(overrides)
        return base

    async def _handle_wait_for_gemini_report_phase(
        self, state: Any, gemini_report_path: Optional[str], no_auto_research: bool
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        report_candidate = gemini_report_path or state.gemini_report_path
        prompt_path = state.metadata.get("deep_research_prompt_path")
        if prompt_path:
            print(f"[Hybrid] Deep research prompt: {prompt_path}", flush=True)
        if not report_candidate and not no_auto_research:
            report_candidate, audit_path = await self._run_auto_gemini_research(
                state,
            )
            if audit_path:
                state.metadata["search_audit_path"] = str(audit_path)

        if not report_candidate:
            state.status = "paused_waiting_for_gemini_report"
            save_run_state(state)
            return state, self._phase_result(
                state,
                success=True,
                deep_research_prompt_path=state.metadata.get(
                    "deep_research_prompt_path"
                ),
                message="Waiting for Gemini report. Resume with --gemini-report.",
            )

        report_file = Path(report_candidate)
        if not report_file.exists() or not report_file.is_file():
            state.status = "paused_waiting_for_gemini_report"
            save_run_state(state)
            return state, self._phase_result(
                state,
                error=f"Gemini report not found: {report_file}",
            )

        report_text = report_file.read_text(encoding="utf-8", errors="replace")
        copied_report = save_finding(
            state.project_dir,
            "reports",
            "gemini_report.txt",
            report_text,
        )
        state.gemini_report_path = str(copied_report)
        state.status = "active"
        save_run_state(state)
        state = set_phase(state, PipelinePhase.SYNTHESIS, "Gemini report supplied")
        return state, None

    async def _handle_synthesis_phase(self, state: Any, phase: Any) -> Any:
        report_path = Path(state.gemini_report_path or "")
        if not report_path.exists():
            return set_phase(
                state,
                PipelinePhase.WAIT_FOR_GEMINI_REPORT,
                "Gemini report missing during synthesis",
            )

        idea_text = self._read_hybrid_artifact_text(
            state.metadata.get("idea_generation_path")
        )
        report_text = report_path.read_text(encoding="utf-8", errors="replace")
        synthesis_context = (
            f"Idea artifact:\n{idea_text}\n\nGemini report:\n{report_text}"
        )
        synthesis_payload = await self._generate_hybrid_phase_payload(
            state,
            phase,
            synthesis_context,
        )
        synthesis_payload["image_queries"] = self._normalize_query_list(
            synthesis_payload.get("image_queries"), minimum_count=3
        )
        synthesis_payload["video_queries"] = self._normalize_query_list(
            synthesis_payload.get("video_queries"), minimum_count=3
        )
        synthesis_path = save_finding(
            state.project_dir,
            "synthesis",
            "synthesis.json",
            synthesis_payload,
        )
        state.metadata["synthesis_path"] = str(synthesis_path)
        save_run_state(state)
        return set_phase(
            state,
            PipelinePhase.EVIDENCE_GATHERING,
            "Synthesis prepared",
        )

    async def _handle_evidence_gathering_phase(self, state: Any) -> Any:
        synthesis_path = state.metadata.get("synthesis_path")
        synthesis_payload: Dict[str, Any] = {}
        if synthesis_path and Path(synthesis_path).exists():
            try:
                synthesis_payload = json.loads(
                    Path(synthesis_path).read_text(encoding="utf-8")
                )
            except Exception:
                synthesis_payload = {}

        image_queries = self._normalize_query_list(
            synthesis_payload.get("image_queries"), minimum_count=0
        )
        video_queries = self._normalize_query_list(
            synthesis_payload.get("video_queries"), minimum_count=0
        )

        # ── Agentic Research Loop ────────────────────────────────────────
        # Run autonomous deep research: LLM-driven search + evaluation loop
        # to find verified facts with exact quotes and URLs.
        topic = synthesis_payload.get("chosen_angle", "")
        evidence_questions = synthesis_payload.get("evidence_questions", [])
        verified_evidence: List[Dict[str, str]] = []
        if topic and evidence_questions:
            exa_api_key = (os.getenv("EXA_API_KEY") or "").strip()
            if exa_api_key:
                self.logger.info(
                    "Starting agentic research for topic='%s' with %d claims",
                    topic,
                    len(evidence_questions),
                )
                search_client = ExaSearchClient(api_key=exa_api_key)
                researcher = AgenticExaResearcher(
                    search_client,
                    max_iterations=4,
                )
                try:
                    research_report = await researcher.conduct_deep_research(
                        topic=topic,
                        initial_claims=evidence_questions,
                    )
                    verified_evidence = research_report.get("verified_evidence", [])
                    if verified_evidence:
                        self.logger.info(
                            "Agentic research verified %d facts",
                            len(verified_evidence),
                        )
                        save_finding(
                            state.project_dir,
                            "evidence",
                            "verified_evidence.json",
                            research_report,
                        )
                    else:
                        self.logger.info(
                            "Agentic research found no verified evidence "
                            "(results evaluated: %d)",
                            research_report.get("total_results_evaluated", 0),
                        )
                except Exception as exc:
                    self.logger.error("Agentic research failed: %s", exc)
                finally:
                    await search_client.close()
            else:
                self.logger.info("EXA_API_KEY not set; skipping agentic research")
        else:
            self.logger.info(
                "No chosen_angle or evidence_questions; skipping agentic research"
            )
        # ── End Agentic Research Loop ────────────────────────────────────

        report_text = self._read_hybrid_artifact_text(state.gemini_report_path)
        evidence_context_parts = [
            json.dumps(synthesis_payload, indent=2, ensure_ascii=True)
            if synthesis_payload
            else "",
            report_text,
        ]
        evidence_context = "\n\n".join(
            [part for part in evidence_context_parts if str(part or "").strip()]
        ).strip()
        evidence_payload = await self._generate_hybrid_phase_payload(
            state,
            PipelinePhase.EVIDENCE_GATHERING,
            evidence_context or "Gather evidence for the chosen documentary angle.",
        )
        media_queries = self._normalize_query_list(
            evidence_payload.get("media_queries"), minimum_count=0
        )
        search_image_queries = self._normalize_query_list(
            image_queries + media_queries,
            minimum_count=0,
        )
        search_video_queries = self._normalize_query_list(
            video_queries + media_queries,
            minimum_count=0,
        )

        print("[Hybrid] Searching for images and videos...", flush=True)
        search_payload = await self._run_hybrid_media_queries(
            image_queries=search_image_queries,
            video_queries=search_video_queries,
            count=6,
        )
        search_path = save_finding(
            state.project_dir,
            "evidence",
            "media_search_results.json",
            search_payload,
        )

        downloaded_media = await self._download_hybrid_media_assets(
            state,
            search_payload,
        )

        # Index downloaded assets into local vector store for future dedup + retrieval
        if getattr(self, "asset_store", None) is not None:
            await self.asset_store.initialize()
            raw_media_dir = Path(state.project_dir) / "raw_media"
            if raw_media_dir.exists():
                indexed = await self.asset_store.scan_directory(raw_media_dir)
                self.logger.info(
                    "Indexed %d downloaded assets into local vector store", indexed
                )

        transcripts = await self._transcribe_hybrid_media_assets(
            state,
            downloaded_media,
        )

        evidence_index = {
            "phase": PipelinePhase.EVIDENCE_GATHERING.value,
            "created_at": datetime.now().isoformat(),
            "evidence_plan": evidence_payload.get("evidence_plan", []),
            "verified_evidence": verified_evidence,
            "media_queries": media_queries,
            "image_queries": search_image_queries,
            "video_queries": search_video_queries,
            "search_results_path": str(search_path),
            "downloaded_media": downloaded_media,
            "transcripts": transcripts,
        }
        evidence_path = save_finding(
            state.project_dir,
            "evidence",
            "evidence_index.json",
            evidence_index,
        )
        state.metadata["evidence_index_path"] = str(evidence_path)
        state.metadata["evidence_search_path"] = str(search_path)
        state.metadata["raw_media_dir"] = str(Path(state.project_dir) / "raw_media")
        save_run_state(state)
        return set_phase(
            state,
            PipelinePhase.SCRIPTING,
            "Evidence artifacts captured",
        )

    async def _handle_scripting_phase(self, state: Any, phase: Any) -> Any:
        context_parts: List[str] = []
        for key in (
            "idea_generation_path",
            "synthesis_path",
            "evidence_search_path",
            "evidence_index_path",
        ):
            context_parts.append(
                self._read_hybrid_artifact_text(state.metadata.get(key))
            )
        if state.gemini_report_path:
            context_parts.append(
                self._read_hybrid_artifact_text(state.gemini_report_path)
            )
        raw_context = "\n\n".join([item for item in context_parts if item]).strip()

        workspace_assets = self._collect_hybrid_workspace_assets(state)
        assets_json = json.dumps(workspace_assets, indent=2, ensure_ascii=True)
        raw_context = (
            f"{raw_context}\n\n[AVAILABLE_WORKSPACE_ASSETS]\n{assets_json}"
            if raw_context
            else f"[AVAILABLE_WORKSPACE_ASSETS]\n{assets_json}"
        )

        max_tokens = int(os.getenv("HYBRID_SCRIPT_CONTEXT_MAX_TOKENS", "120000"))

        summary_result = await summarize_if_needed(
            raw_context,
            max_tokens=max_tokens,
            api_key=getattr(self.config.api, "nvidia_nim_api_key", None),
            base_url=getattr(
                self.config.api,
                "nvidia_nim_base_url",
                "https://integrate.api.nvidia.com/v1",
            ),
        )
        summary_path = save_finding(
            state.project_dir,
            "summaries",
            "script_context.txt",
            summary_result.context,
        )
        state.metadata["script_context_path"] = str(summary_path)

        script_payload = await self._run_agentic_scripting_with_visual_loop(
            state,
            phase,
            summary_result.context or raw_context,
        )
        final_script_path = save_finding(
            state.project_dir,
            "scripts",
            "final_script.json",
            script_payload,
        )
        state.metadata["final_script_path"] = str(final_script_path)
        state = set_phase(
            state,
            PipelinePhase.VIDEO_RENDER,
            "Script completed",
        )
        state.status = "active"
        save_run_state(state)
        print(f"[Hybrid] Script saved to: {final_script_path}", flush=True)
        return state

    async def _handle_video_render_phase(
        self, state: Any
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        final_script_path = str(state.metadata.get("final_script_path", "") or "")
        if not final_script_path:
            state.status = "paused_render_failed"
            save_run_state(state)
            return state, self._phase_result(
                state,
                error="Missing final_script_path in state metadata. Run SCRIPTING phase first to generate a script, then resume with --resume --phase VIDEO_RENDER",
            )

        script_path = Path(final_script_path)
        if not script_path.exists() or not script_path.is_file():
            state.status = "paused_render_failed"
            save_run_state(state)
            return state, self._phase_result(
                state,
                final_script_path=final_script_path,
                error=f"Final script not found: {script_path}",
            )

        try:
            script_payload = json.loads(script_path.read_text(encoding="utf-8"))
        except Exception as exc:
            state.status = "paused_render_failed"
            save_run_state(state)
            return state, self._phase_result(
                state,
                final_script_path=final_script_path,
                error=f"Failed to parse final script JSON: {exc}",
            )

        max_render_retries = int(os.getenv("HYBRID_RENDER_MAX_RETRIES", "2"))
        render_result = None
        last_error = ""

        for attempt in range(max_render_retries + 1):
            try:
                render_result = await self._render_hybrid_local_video(
                    state, script_payload, render_attempt=attempt
                )
            except TypeError:
                render_result = await self._render_hybrid_local_video(
                    state, script_payload
                )

            if render_result.get("success"):
                break
            last_error = str(render_result.get("error", "Unknown render error"))
            self.logger.warning("Render attempt %d failed: %s", attempt + 1, last_error)

        if not render_result or not render_result.get("success"):
            state.status = "paused_render_failed"
            save_run_state(state)
            return state, self._phase_result(
                state,
                final_script_path=final_script_path,
                error=f"Hybrid render failed after {max_render_retries + 1} attempts: {last_error}",
            )

        final_video_path = str(render_result.get("video_path", "") or "")
        render_manifest_path = str(render_result.get("render_manifest_path", "") or "")
        state.metadata["final_video_path"] = final_video_path
        if render_manifest_path:
            state.metadata["render_manifest_path"] = render_manifest_path
        state.status = "completed"
        save_run_state(state)

        return state, {
            "success": True,
            "paused": False,
            "status": state.status,
            "current_phase": PipelinePhase.VIDEO_RENDER.value,
            "project_name": state.project_name,
            "pipeline": "hybrid_documentary_studio",
            "no_upload": bool(state.metadata.get("hybrid_no_upload", False)),
            "final_script_path": final_script_path,
            "final_video_path": final_video_path,
            "render_manifest_path": render_manifest_path,
        }

    async def _run_hybrid_state_machine(
        self,
        state,
        gemini_report_path: Optional[str] = None,
        no_auto_research: bool = False,
    ) -> Dict[str, Any]:
        while True:
            phase = self._coerce_hybrid_phase(state.current_phase)
            if phase is None:
                return {
                    "success": False,
                    "error": f"Unsupported phase: {state.current_phase}",
                    "current_phase": str(state.current_phase),
                    "project_name": state.project_name,
                    "pipeline": "hybrid_documentary_studio",
                }

            self.logger.info("Hybrid phase start: %s", phase.value)
            print(f"[Hybrid] Phase: {phase.value}", flush=True)

            if phase == PipelinePhase.IDEA_GENERATION:
                state = await self._handle_idea_generation_phase(state)
                continue

            if phase == PipelinePhase.WAIT_FOR_GEMINI_REPORT:
                state, result = await self._handle_wait_for_gemini_report_phase(
                    state, gemini_report_path, no_auto_research
                )
                if result is not None:
                    return result
                gemini_report_path = None
                continue

            if phase == PipelinePhase.SYNTHESIS:
                state = await self._handle_synthesis_phase(state, phase)
                continue

            if phase == PipelinePhase.EVIDENCE_GATHERING:
                state = await self._handle_evidence_gathering_phase(state)
                continue

            if phase == PipelinePhase.SCRIPTING:
                state = await self._handle_scripting_phase(state, phase)
                continue

            if phase == PipelinePhase.VIDEO_RENDER:
                state, result = await self._handle_video_render_phase(state)
                if result is not None:
                    return result
                # Just in case, though it should always return from video render phase
                break

        return {"success": False, "error": "Exited state machine unexpectedly"}

    # Curated scouting queries for search-first idea discovery (avoid LLM inventing topics)
    _SCOUTING_QUERIES = [
        "removed speedrun world record controversy",
        "deleted forum thread internet mystery",
        "site:reddit.com obscure lore scandal",
        "gaming community deleted evidence",
        "internet mystery primary sources",
        "controversy timeline key events",
        "deleted world record receipts",
    ]

    async def _generate_scouting_queries(self, state) -> list[str]:
        """Generate scouting queries via LLM for better topic diversity, falls back to hardcoded."""
        try:
            prompt = (
                f"Generate 6 niche search queries for a documentary about '{state.project_name}'. "
                "Focus on internet mysteries, gaming lore, deleted content, and controversies. "
                'Return JSON {"queries": ["..."]} only.'
            )
            resp = await self.ai_client.active_client._chat_completion_with_fallback(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            queries = json.loads(resp.choices[0].message.content).get("queries", [])
            if queries and len(queries) >= 3:
                return queries[:8]
        except Exception as exc:
            self.logger.warning("LLM query generation failed, using defaults: %s", exc)
        return list(self._SCOUTING_QUERIES)

    async def _run_idea_generation_search_first(self, state) -> tuple:
        """Search-first IDEA_GENERATION: run searches, then LLM extracts angles with source_urls."""
        api_key = (
            os.getenv("EXA_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY") or ""
        ).strip()

        logs_dir = Path(state.project_dir) / "research" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = logs_dir / f"search_audit_{timestamp}.jsonl"
        audit_logger = SearchAuditLogger(audit_path) if api_key else None

        if not api_key:
            self.logger.warning(
                "No EXA_API_KEY or BRAVE_SEARCH_API_KEY. "
                "Falling back to LLM-only idea generation (may invent content)."
            )
            idea_payload = await self._generate_hybrid_phase_payload(
                state,
                PipelinePhase.IDEA_GENERATION,
                state.context_snapshot
                or "Generate idea angles for a documentary short.",
            )
            self._ensure_source_urls_on_angles(idea_payload)
            return idea_payload, None

        print("[Hybrid] Searching for ideas (5-8 scouting queries)...", flush=True)
        search_context = ""
        async with ExaSearchClient(audit_logger=audit_logger) as client:
            scouting_queries = await self._generate_scouting_queries(state)
            for q in scouting_queries:
                hits = await client.search_web(q, count=4)
                search_context += f"\n--- RESULTS FOR QUERY: {q} ---\n"
                for hit in hits:
                    search_context += f"TITLE: {hit.title}\nURL: {hit.url}\nSNIPPET: {hit.description}\n\n"

        if not search_context.strip():
            self.logger.warning("Search returned no results. Falling back to LLM-only.")
            idea_payload = await self._generate_hybrid_phase_payload(
                state,
                PipelinePhase.IDEA_GENERATION,
                state.context_snapshot
                or "Generate idea angles for a documentary short.",
            )
            self._ensure_source_urls_on_angles(idea_payload)
            return idea_payload, str(audit_path) if audit_logger else None

        save_finding(
            state.project_dir,
            "ideas",
            "raw_scout_data.txt",
            search_context,
        )

        processed = (
            await summarize_if_needed(
                search_context,
                max_tokens=120000,
            )
        ).context

        idea_payload = await self._extract_angles_from_search(processed, state)
        self._validate_source_urls(idea_payload, search_context)
        return idea_payload, str(audit_path) if audit_logger else None

    def _ensure_source_urls_on_angles(self, payload: Dict[str, Any]) -> None:
        """Ensure each angle has source_urls list (empty if none)."""
        for angle in payload.get("angles", []):
            if isinstance(angle, dict) and "source_urls" not in angle:
                angle["source_urls"] = []

    def _validate_source_urls(
        self, payload: Dict[str, Any], raw_search_context: str
    ) -> None:
        """Flag angles whose source_urls are not present in raw search results."""
        seen_urls = set()
        for line in raw_search_context.split("\n"):
            if line.startswith("URL: "):
                seen_urls.add(line[5:].strip())
        for angle in payload.get("angles", []):
            if not isinstance(angle, dict):
                continue
            urls = angle.get("source_urls", [])
            for u in urls:
                if u and u not in seen_urls:
                    self.logger.warning(
                        "Angle '%s' cites URL not in search results: %s",
                        angle.get("title", "?"),
                        u[:80],
                    )

    async def _extract_angles_from_search(
        self, search_context: str, state
    ) -> Dict[str, Any]:
        """LLM extraction: angles with source_urls from search results only."""
        extract_prompt = f"""Below are search results. Extract up to 3 story angles that have concrete evidence in these results.

CRITICAL: Do NOT invent any facts, dates, or sources. Every angle MUST cite source_urls (exact URLs from the results). If something is not in the results, omit it.

Return strict JSON:
{{
  "phase": "IDEA_GENERATION",
  "angles": [
    {{"id": "A1", "title": "...", "hook": "...", "viability_score": 85, "source_urls": ["https://..."]}},
    ...
  ],
  "gemini_deep_research_prompt": "Search for primary sources, court filings, exact dates, and archived links for the chosen angle. Omit anything not found.",
  "next_phase": "WAIT_FOR_GEMINI_REPORT"
}}

Search results:
{search_context}
"""
        active_client = getattr(self.ai_client, "active_client", None)
        if active_client and hasattr(active_client, "_chat_completion_with_fallback"):
            try:
                response = await active_client._chat_completion_with_fallback(
                    messages=[
                        {
                            "role": "system",
                            "content": "Return strict JSON only. Extract from search results. Every angle must have source_urls from the results. Never invent.",
                        },
                        {"role": "user", "content": extract_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1800,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
                if (
                    isinstance(parsed, dict)
                    and parsed.get("phase") == PipelinePhase.IDEA_GENERATION.value
                    and isinstance(parsed.get("angles"), list)
                ):
                    self._ensure_source_urls_on_angles(parsed)
                    return self._prepare_idea_generation_payload(
                        parsed,
                        project_name=getattr(state, "project_name", ""),
                    )
            except Exception as exc:
                self.logger.warning("Extract angles failed: %s", exc)

        fallback = self._fallback_hybrid_phase_payload(
            PipelinePhase.IDEA_GENERATION, search_context
        )
        self._ensure_source_urls_on_angles(fallback)
        return self._prepare_idea_generation_payload(
            fallback,
            project_name=getattr(state, "project_name", ""),
        )

    async def _run_auto_gemini_research(self, state) -> tuple:
        """Run AgenticResearcher or DeepResearchClient to produce gemini report."""
        api_key_brave = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        api_key_exa = (os.getenv("EXA_API_KEY") or "").strip()

        idea_path = Path(state.metadata.get("idea_generation_path", ""))
        if not idea_path.exists():
            self.logger.warning("idea_generation.json not found; cannot auto-research")
            return None, None

        try:
            idea_payload = json.loads(idea_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Failed to read idea_generation.json: %s", exc)
            return None, None

        idea_payload = self._prepare_idea_generation_payload(
            idea_payload,
            project_name=getattr(state, "project_name", ""),
        )
        prompt_path = self._save_deep_research_prompt_artifact(state, idea_payload)
        if prompt_path:
            state.metadata["deep_research_prompt_path"] = str(prompt_path)

        try:
            idea_path.write_text(
                json.dumps(idea_payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("Failed to persist enriched idea payload: %s", exc)

        prompt = str(idea_payload.get("gemini_deep_research_prompt", "") or "").strip()
        angles = idea_payload.get("angles", [])
        topic = str(angles[0].get("title", "") or "").strip() if angles else ""

        if not topic:
            topic = str(state.project_name or "research topic").strip()

        research_query = self._build_gemini_research_query(topic, prompt)

        logs_dir = Path(state.project_dir) / "research" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = logs_dir / f"search_audit_{timestamp}.jsonl"
        audit_logger = SearchAuditLogger(audit_path)

        if api_key_brave:
            researcher = AgenticResearcher(audit_logger=audit_logger)
            report_text = await researcher.deep_dive(
                topic=topic,
                initial_context=research_query,
                max_turns=3,
            )
        elif api_key_exa:
            client = DeepResearchClient(audit_logger=audit_logger)
            report_text = await client.conduct_deep_research(research_query)
            await client.close()
        else:
            self.logger.warning(
                "No BRAVE_SEARCH_API_KEY or EXA_API_KEY for auto research"
            )
            return None, None

        if not self._is_research_report_relevant(report_text, topic, angles):
            self.logger.warning(
                "Auto research output appears off-topic for '%s'; waiting for manual report",
                topic,
            )
            return None, str(audit_path)

        report_path = save_finding(
            state.project_dir,
            "reports",
            "gemini_report.txt",
            report_text,
        )
        self.logger.info("Auto research report saved: %s", report_path)
        return str(report_path), str(audit_path)

    @staticmethod
    def _build_gemini_research_query(topic: str, prompt: str) -> str:
        topic_text = str(topic or "research topic").strip()
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return f"Search for primary sources and key facts: {topic_text}"

        lowered = prompt_text.lower()
        keyword_map = {
            "primary": "primary sources",
            "timeline": "timeline",
            "date": "exact dates",
            "official": "official statements",
            "archive": "archived links",
            "forum": "forum posts",
            "leaderboard": "leaderboard snapshots",
            "court": "court filings",
        }
        inferred_terms: List[str] = []
        for token, term in keyword_map.items():
            if token in lowered and term not in inferred_terms:
                inferred_terms.append(term)

        source_urls = re.findall(r"https?://[^\s)]+", prompt_text)
        site_filters: List[str] = []
        for source_url in source_urls:
            host = urlparse(source_url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            if host and host not in site_filters:
                site_filters.append(host)
            if len(site_filters) >= 2:
                break

        query_parts = [topic_text, "fact check"]
        if inferred_terms:
            query_parts.append(" ".join(inferred_terms[:4]))
        if site_filters:
            query_parts.append(" ".join(f"site:{host}" for host in site_filters))

        return " | ".join(part for part in query_parts if part).strip()

    @staticmethod
    def _extract_topic_keywords(topic: str) -> List[str]:
        words = re.findall(r"[a-z0-9]+", str(topic or "").lower())
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "how",
            "after",
            "when",
            "over",
            "under",
            "about",
            "what",
            "which",
            "where",
            "world",
            "record",
            "records",
            "controversy",
            "history",
            "research",
        }
        filtered = [w for w in words if len(w) >= 4 and w not in stop_words]
        deduped: List[str] = []
        for token in filtered:
            if token not in deduped:
                deduped.append(token)
        return deduped

    def _is_research_report_relevant(
        self,
        report_text: Any,
        topic: str,
        angles: Any,
    ) -> bool:
        text = str(report_text or "").strip().lower()
        if not text:
            return False

        keywords = self._extract_topic_keywords(topic)
        if not keywords:
            return True

        False_POSITIVE_WORDS = {"speedrun", "forum", "world", "record", "history"}
        relevant_keywords = [k for k in keywords if k not in False_POSITIVE_WORDS]
        if not relevant_keywords:
            relevant_keywords = keywords

        keyword_hits = sum(1 for token in relevant_keywords if token in text)
        if keyword_hits >= 3:
            return True

        if isinstance(angles, list):
            for angle in angles:
                if not isinstance(angle, dict):
                    continue
                for source_url in angle.get("source_urls", []) or []:
                    if not isinstance(source_url, str):
                        continue
                    host = urlparse(source_url).netloc.lower()
                    if host and host in text:
                        return True

        return False

    @staticmethod
    def _safe_parse_phase_json(content: str) -> Optional[Dict[str, Any]]:
        """Extract a valid JSON object from possibly malformed LLM output."""
        text = (content or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start < 0:
            return None
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        end = text.rfind("}")
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return None

    async def _generate_hybrid_phase_payload(
        self,
        state,
        phase: PipelinePhase,
        context: str,
    ) -> Dict[str, Any]:
        print(
            f"[Hybrid] Calling AI for {phase.value}... (may take 1-2 min)", flush=True
        )
        max_retries = int(os.getenv("HYBRID_AI_VALIDATION_MAX_RETRIES", "2"))
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            if attempt > 0 and last_error:
                print(
                    f"[Hybrid] Retrying {phase.value} (attempt {attempt + 1}/{max_retries + 1})...",
                    flush=True,
                )
                retry_context = context + build_validation_feedback_prompt(
                    phase.value, {}, last_error
                )
            else:
                retry_context = context

            prompt = build_state_machine_prompt(state, retry_context)
            active_client = getattr(self.ai_client, "active_client", None)

            if not active_client or not hasattr(
                active_client, "_chat_completion_with_fallback"
            ):
                raise RuntimeError(
                    f"No AI client available for hybrid phase {phase.value}"
                )

            max_tokens = 4000 if phase == PipelinePhase.SCRIPTING else 1800
            response = await active_client._chat_completion_with_fallback(
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON for the provided phase contract only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = self._safe_parse_phase_json(content)
            if not isinstance(parsed, dict):
                if attempt < max_retries:
                    last_error = "Unparseable JSON response"
                    continue
                self.logger.warning(
                    "Hybrid phase %s returned unparseable JSON; using fallback payload",
                    phase.value,
                )
                return self._fallback_hybrid_phase_payload(phase, context)

            validated, validation_error = validate_phase_output(phase.value, parsed)
            if validation_error and attempt < max_retries:
                last_error = validation_error
                continue

            normalized = self._coerce_hybrid_phase_payload(phase, parsed, context)
            if not self._is_valid_hybrid_phase_payload(phase, normalized):
                if attempt < max_retries:
                    last_error = "Payload failed key validation"
                    continue
                self.logger.warning(
                    "Hybrid phase %s payload failed validation after normalization; using fallback payload",
                    phase.value,
                )
                return self._fallback_hybrid_phase_payload(phase, context)

            if validated is not None:
                return validated
            return normalized

        return self._fallback_hybrid_phase_payload(phase, context)

    def _coerce_hybrid_phase_payload(
        self,
        phase: PipelinePhase,
        payload: Dict[str, Any],
        context: str,
    ) -> Dict[str, Any]:
        """Drop unknown keys and backfill missing required keys from fallback payloads."""
        fallback = self._fallback_hybrid_phase_payload(phase, context)
        normalized: Dict[str, Any] = {"phase": phase.value}

        for key, fallback_value in fallback.items():
            if key == "phase":
                continue

            value = payload.get(key, fallback_value)
            if isinstance(fallback_value, list) and not isinstance(value, list):
                value = fallback_value
            elif isinstance(fallback_value, str) and value is None:
                value = fallback_value
            normalized[key] = value

        return normalized

    async def _generate_supplementary_image_queries(
        self,
        state: Any,
        script_payload: Dict[str, Any],
        *,
        segment_indexes: List[int],
        existing_queries: Optional[List[str]] = None,
        max_per_segment: int = 3,
    ) -> List[str]:
        segments = script_payload.get("segments")
        if not isinstance(segments, list) or not segments:
            return []

        requested_indexes = [
            idx
            for idx in segment_indexes
            if isinstance(idx, int) and 0 <= idx < len(segments)
        ]
        if not requested_indexes:
            return []

        max_per_segment = max(1, min(6, int(max_per_segment or 3)))
        existing_set = {
            str(item).strip().lower()
            for item in (existing_queries or [])
            if str(item).strip()
        }

        segment_blocks: List[str] = []
        for idx in requested_indexes:
            segment = segments[idx]
            if not isinstance(segment, dict):
                continue
            narration = str(segment.get("narration", "") or "").strip()
            visual_directive = str(segment.get("visual_directive", "") or "").strip()
            evidence_refs = segment.get("evidence_refs", [])
            if isinstance(evidence_refs, list):
                evidence_text = ", ".join(
                    [str(item).strip() for item in evidence_refs if str(item).strip()]
                )
            else:
                evidence_text = str(evidence_refs or "").strip()

            segment_blocks.append(
                "\n".join(
                    [
                        f"SEGMENT_INDEX: {idx}",
                        f"NARRATION: {narration}",
                        f"VISUAL_DIRECTIVE: {visual_directive}",
                        f"EVIDENCE_REFS: {evidence_text}",
                    ]
                )
            )

        if not segment_blocks:
            return []

        prompt = (
            "You are generating Brave/Hackclub image search queries to find documentary-grade visual receipts.\n"
            f"Generate up to {max_per_segment} image search queries per segment below.\n\n"
            "Rules:\n"
            "- Queries must be specific (names, dates, handles, document titles, leaderboard names).\n"
            "- Prefer evidence visuals: screenshots, PDFs, tweets, statements, forum posts, leaderboards.\n"
            "- Avoid generic b-roll queries like 'minecraft gameplay' unless the segment is explicitly about generic gameplay.\n"
            "- Do not include any query already present in the existing set.\n"
            '- Return strict JSON only: {"image_queries": ["..."]}.\n\n'
            f"EXISTING_QUERIES (do not repeat): {json.dumps(sorted(existing_set)[:80], ensure_ascii=True)}\n\n"
            "SEGMENTS:\n" + "\n\n".join(segment_blocks)
        )

        active_client = getattr(self.ai_client, "active_client", None)
        if active_client and hasattr(active_client, "_chat_completion_with_fallback"):
            try:
                response = await active_client._chat_completion_with_fallback(
                    messages=[
                        {
                            "role": "system",
                            "content": "Return strict JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                parsed = self._safe_parse_phase_json(content) or {}
                raw_queries = parsed.get("image_queries", [])
                if isinstance(raw_queries, list):
                    cleaned: List[str] = []
                    for item in raw_queries:
                        text = str(item or "").strip()
                        lowered = text.lower()
                        if not text or lowered in existing_set:
                            continue
                        if len(text) < 4 or len(text) > 140:
                            continue
                        if text not in cleaned:
                            cleaned.append(text)
                    if cleaned:
                        return cleaned[: max_per_segment * len(requested_indexes)]
            except Exception:
                pass

        # Heuristic fallback: build evidence-oriented variants from segment tokens.
        fallback_queries: List[str] = []
        for idx in requested_indexes:
            segment = segments[idx]
            if not isinstance(segment, dict):
                continue
            narration = str(segment.get("narration", "") or "").strip()
            visual_directive = str(segment.get("visual_directive", "") or "").strip()
            tokens = self._tokenize_relevance_text(f"{narration} {visual_directive}")
            core = " ".join(tokens[:8]).strip()
            if not core:
                continue
            variants = [
                f"{core} screenshot",
                f"{core} leaderboard",
                f"{core} pdf",
                f"{core} tweet",
                f"{core} forum post",
            ]
            for query in variants:
                lowered = query.lower().strip()
                if lowered in existing_set:
                    continue
                if query not in fallback_queries:
                    fallback_queries.append(query)
        return fallback_queries[: max_per_segment * len(requested_indexes)]

    async def _run_visual_feedback_agent(
        self,
        state: Any,
        script_payload: Dict[str, Any],
        visual_catalog: Dict[str, Any],
    ) -> Dict[str, Any]:
        segments = script_payload.get("segments")
        if not isinstance(segments, list) or not segments:
            return {}

        active_client = getattr(self.ai_client, "active_client", None)
        if not active_client or not hasattr(
            active_client, "_chat_completion_with_fallback"
        ):
            return {}

        segments_json = json.dumps(segments, ensure_ascii=True)
        visuals_json = json.dumps(visual_catalog, ensure_ascii=True)

        prompt = (
            "You are a visual supervisor for a short documentary script.\n"
            "You will receive:\n"
            "1) narrative script segments (with index, narration, visual_directive)\n"
            "2) a catalog of available images (with local_path, query, title, source_url)\n\n"
            "For each segment, decide whether its visuals are satisfactory, need more images, or need narration revision to better match available visuals.\n\n"
            'Return strict JSON ONLY. The top-level object must have a key "segments" '
            "mapping to a list of objects with fields: "
            '"segment_index" (int), "status" ("satisfied" | "needs_images" | "needs_revision"), '
            '"preferred_assets" (list of strings), "new_image_queries" (list of strings), '
            'and "revised_narration" (string).\n\n'
            "Constraints:\n"
            "- Maintain all factual details (names, dates, URLs) in any revised_narration.\n"
            "- Use preferred_assets only from the provided visual catalog local_path values.\n"
            '- new_image_queries should be specific evidence-style queries when status == "needs_images".\n'
            '- If a segment is already well-served visually, mark it as "satisfied" and leave revised_narration empty.\n\n'
            f"SCRIPT_SEGMENTS_JSON:\n{segments_json}\n\n"
            f"VISUAL_CATALOG_JSON:\n{visuals_json}\n"
        )

        try:
            response = await active_client._chat_completion_with_fallback(
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
        except Exception:
            return {}

        content = response.choices[0].message.content or "{}"
        parsed = self._safe_parse_phase_json(content) or {}
        if not isinstance(parsed, dict):
            return {}
        segments_feedback = parsed.get("segments")
        if not isinstance(segments_feedback, list):
            return {}
        return parsed

    @staticmethod
    def _apply_visual_feedback_to_script(
        script_payload: Dict[str, Any],
        feedback: Dict[str, Any],
        *,
        allow_revisions: bool,
    ) -> None:
        segments = script_payload.get("segments")
        if not isinstance(segments, list) or not segments:
            return
        fb_segments = feedback.get("segments")
        if not isinstance(fb_segments, list):
            return

        for entry in fb_segments:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("segment_index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(segments):
                continue

            segment = segments[idx]

            if allow_revisions:
                revised = str(entry.get("revised_narration", "") or "").strip()
                # Require a bit of substance to avoid trivial/no-op changes.
                if revised and len(revised.split()) >= 3:
                    segment["narration"] = revised

            preferred_assets = entry.get("preferred_assets") or []
            if isinstance(preferred_assets, list) and preferred_assets:
                first_path = str(preferred_assets[0] or "").strip()
                if first_path:
                    segment["visual_asset_path"] = first_path

    @staticmethod
    def _is_valid_hybrid_phase_payload(
        phase: PipelinePhase,
        payload: Dict[str, Any],
    ) -> bool:
        required_keys_map = {
            PipelinePhase.IDEA_GENERATION: {
                "phase",
                "angles",
                "gemini_deep_research_prompt",
                "next_phase",
            },
            PipelinePhase.SYNTHESIS: {
                "phase",
                "chosen_angle",
                "reasoning",
                "image_queries",
                "video_queries",
                "evidence_questions",
                "next_phase",
            },
            PipelinePhase.EVIDENCE_GATHERING: {
                "phase",
                "evidence_plan",
                "media_queries",
                "next_phase",
            },
            PipelinePhase.SCRIPTING: {
                "phase",
                "title",
                "hook",
                "loop_bridge",
                "segments",
                "sources_to_check",
                "hashtags",
            },
        }

        required_keys = required_keys_map.get(phase)
        if not required_keys:
            return False
        if payload.get("phase") != phase.value:
            return False

        actual_keys = set(payload.keys())
        if not required_keys.issubset(actual_keys):
            return False

        if phase == PipelinePhase.IDEA_GENERATION:
            prompt_text = str(
                payload.get("gemini_deep_research_prompt", "") or ""
            ).strip()
            if len(prompt_text.split()) < 8:
                return False
            if not isinstance(payload.get("angles"), list):
                return False

        if phase == PipelinePhase.SYNTHESIS:
            if not isinstance(payload.get("image_queries"), list):
                return False
            if not isinstance(payload.get("video_queries"), list):
                return False
            if not isinstance(payload.get("evidence_questions"), list):
                return False

        if phase == PipelinePhase.EVIDENCE_GATHERING:
            if not isinstance(payload.get("evidence_plan"), list):
                return False
            if not isinstance(payload.get("media_queries"), list):
                return False

        if phase == PipelinePhase.SCRIPTING:
            if not isinstance(payload.get("segments"), list):
                return False
            if not isinstance(payload.get("sources_to_check"), list):
                return False
            if not isinstance(payload.get("hashtags"), list):
                return False

        return True

    def _fallback_hybrid_phase_payload(
        self,
        phase: PipelinePhase,
        context: str,
    ) -> Dict[str, Any]:
        _ = context
        if phase == PipelinePhase.IDEA_GENERATION:
            return {
                "phase": PipelinePhase.IDEA_GENERATION.value,
                "angles": [
                    {
                        "id": "A1",
                        "title": "The deleted leaderboard record",
                        "hook": "A run disappeared after being called impossible.",
                        "viability_score": 86,
                        "source_urls": [],
                    },
                    {
                        "id": "A2",
                        "title": "The forum post that changed a category",
                        "hook": "One rules thread rewrote years of speedrun history.",
                        "viability_score": 81,
                        "source_urls": [],
                    },
                    {
                        "id": "A3",
                        "title": "The split-time contradiction",
                        "hook": "A timestamp mismatch exposed a hidden timing exploit.",
                        "viability_score": 79,
                        "source_urls": [],
                    },
                ],
                "gemini_deep_research_prompt": (
                    "Act as a forensic web researcher. Investigate all three angles, "
                    "and return exact dates, primary-source URLs, archived forum posts, "
                    "leaderboard snapshots, and video evidence with timestamps."
                ),
                "next_phase": PipelinePhase.WAIT_FOR_GEMINI_REPORT.value,
            }
        if phase == PipelinePhase.SYNTHESIS:
            return {
                "phase": PipelinePhase.SYNTHESIS.value,
                "chosen_angle": "The deleted leaderboard record",
                "reasoning": "It has the strongest source density and verifiable timeline.",
                "image_queries": [
                    "site:speedrun.com archived leaderboard history",
                    "web archive speedrunning forum thread removed",
                    "speedrun rules change screenshot date",
                ],
                "video_queries": [
                    "speedrun world record progression with timestamps",
                    "category dispute vod evidence",
                    "community investigation video source",
                ],
                "evidence_questions": [
                    "Who posted the first verified dispute?",
                    "Which date proves the leaderboard deletion happened?",
                    "Which clip contains direct quote evidence?",
                ],
                "next_phase": PipelinePhase.EVIDENCE_GATHERING.value,
            }
        if phase == PipelinePhase.EVIDENCE_GATHERING:
            return {
                "phase": PipelinePhase.EVIDENCE_GATHERING.value,
                "evidence_plan": [
                    {
                        "claim": "Main narrative has unresolved factual gaps.",
                        "evidence_needed": [
                            "Primary source citation",
                            "Timeline artifact",
                        ],
                        "priority": "high",
                    }
                ],
                "media_queries": [
                    "event timeline document",
                    "supporting visual evidence",
                ],
                "next_phase": PipelinePhase.SCRIPTING.value,
            }
        if phase == PipelinePhase.SCRIPTING:
            return {
                "phase": PipelinePhase.SCRIPTING.value,
                "title": "The Record That Vanished",
                "hook": "A world record was deleted, but the receipts stayed online.",
                "loop_bridge": "...which is exactly why that deleted record still matters.",
                "segments": [
                    {
                        "time_seconds": 0.0,
                        "intended_duration_seconds": 8.0,
                        "narration": "In 2014, a top run vanished from the leaderboard overnight.",
                        "visual_asset_path": "research/evidence/media_search_results.json",
                        "visual_directive": "Zoom into the date and highlight the removed entry.",
                        "text_overlay": "RECORD REMOVED",
                        "evidence_refs": [
                            "research/evidence/media_search_results.json"
                        ],
                        "pace": "fast",
                        "emotion": "dramatic",
                    },
                    {
                        "time_seconds": 8.0,
                        "intended_duration_seconds": 8.0,
                        "narration": "Then a forum thread surfaced with exact split-time contradictions.",
                        "visual_asset_path": "research/evidence/evidence_index.json",
                        "visual_directive": "Highlight timestamp row and pan to contradiction.",
                        "text_overlay": "TIMESTAMP MISMATCH",
                        "evidence_refs": ["research/evidence/evidence_index.json"],
                        "pace": "fast",
                        "emotion": "dramatic",
                    },
                ],
                "sources_to_check": [
                    "Archive snapshots",
                    "Forum URLs",
                    "Video timestamps",
                ],
                "hashtags": ["#speedrun", "#gaminghistory", "#documentary"],
            }
        raise ValueError(f"No fallback payload available for phase {phase.value}")

    @staticmethod
    def _coerce_hybrid_phase(phase_value: Any) -> Optional[PipelinePhase]:
        if isinstance(phase_value, PipelinePhase):
            return phase_value
        if isinstance(phase_value, str):
            try:
                return PipelinePhase(phase_value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _read_hybrid_artifact_text(path_value: Optional[str]) -> str:
        if not path_value:
            return ""
        path = Path(path_value)
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _normalize_query_list(value: Any, minimum_count: int = 0) -> List[str]:
        normalized: List[str] = []
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text and text not in normalized:
                    normalized.append(text)

        while len(normalized) < minimum_count:
            normalized.append(f"primary source evidence {len(normalized) + 1}")
        return normalized

    @staticmethod
    def _collect_source_urls_from_angles(angles: Any, limit: int = 8) -> List[str]:
        urls: List[str] = []
        if not isinstance(angles, list):
            return urls

        for angle in angles:
            if not isinstance(angle, dict):
                continue
            for candidate in angle.get("source_urls", []) or []:
                url = str(candidate or "").strip()
                if not url:
                    continue
                if not urlparse(url).scheme:
                    continue
                if url in urls:
                    continue
                urls.append(url)
                if len(urls) >= max(1, limit):
                    return urls
        return urls

    @staticmethod
    def _build_deep_research_prompt(
        topic: str,
        hook: str,
        source_urls: List[str],
        existing_prompt: str,
    ) -> str:
        topic_text = str(topic or "research topic").strip() or "research topic"
        hook_text = str(hook or "").strip()
        existing_text = " ".join(str(existing_prompt or "").split())

        lines = [
            f"Topic: {topic_text}",
            "Goal: build a verifiable fact dossier for a short documentary script.",
            "Research tasks:",
            "1) Build a dated timeline of key events using primary or near-primary sources.",
            "2) Verify the strongest quantitative claims and quote exact numbers with context.",
            "3) Capture direct statements from involved parties and moderators/official bodies.",
            "4) Find archived links or snapshots for deleted/edited pages when possible.",
            "5) Flag uncertain or conflicting claims instead of filling gaps.",
            "Output format:",
            "- Chronology: YYYY-MM-DD | event | citation URL",
            "- Claims table: claim | evidence | confidence(high/medium/low) | citation URL",
            "- Open questions: unresolved facts that still need receipts",
        ]

        if hook_text:
            lines.append(f"Narrative hook to verify: {hook_text}")

        if source_urls:
            lines.append("Seed sources to verify (do not trust blindly):")
            for url in source_urls[:8]:
                lines.append(f"- {url}")

        if existing_text:
            lines.append(f"Additional directive: {existing_text}")

        return "\n".join(lines)

    def _prepare_idea_generation_payload(
        self,
        payload: Dict[str, Any],
        project_name: str,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}

        prepared: Dict[str, Any] = dict(payload)
        prepared["phase"] = PipelinePhase.IDEA_GENERATION.value

        raw_angles = prepared.get("angles")
        angles: List[Dict[str, Any]] = []
        if isinstance(raw_angles, list):
            for angle in raw_angles:
                if not isinstance(angle, dict):
                    continue
                cleaned = dict(angle)
                source_urls = cleaned.get("source_urls")
                if not isinstance(source_urls, list):
                    source_urls = []
                cleaned["source_urls"] = [
                    str(url).strip()
                    for url in source_urls
                    if str(url or "").strip().startswith(("http://", "https://"))
                ]
                angles.append(cleaned)
        prepared["angles"] = angles

        best_angle: Dict[str, Any] = angles[0] if angles else {}
        topic = (
            str(best_angle.get("title", "") or "").strip()
            or str(project_name or "research topic").strip()
        )
        hook = str(best_angle.get("hook", "") or "").strip()
        source_urls = self._collect_source_urls_from_angles(angles, limit=8)

        prepared["gemini_deep_research_prompt"] = self._build_deep_research_prompt(
            topic=topic,
            hook=hook,
            source_urls=source_urls,
            existing_prompt=str(prepared.get("gemini_deep_research_prompt", "") or ""),
        )
        prepared["next_phase"] = PipelinePhase.WAIT_FOR_GEMINI_REPORT.value
        return prepared

    def _save_deep_research_prompt_artifact(
        self,
        state: Any,
        idea_payload: Dict[str, Any],
    ) -> Optional[Path]:
        prompt_text = str(
            idea_payload.get("gemini_deep_research_prompt", "") or ""
        ).strip()
        if not prompt_text:
            return None
        return save_finding(
            state.project_dir,
            "reports",
            "deep_research_prompt.txt",
            prompt_text,
        )

    async def _run_hybrid_media_queries(
        self,
        image_queries: List[str],
        video_queries: List[str],
        count: int = 6,
    ) -> Dict[str, Any]:
        exa_api_key = (os.getenv("EXA_API_KEY") or "").strip()
        brave_api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        results: Dict[str, Any] = {"image": {}, "video": {}, "web": {}}

        if exa_api_key:
            async with ExaSearchClient(api_key=exa_api_key) as client:
                for query in image_queries:
                    web_hits = await client.search_web(query, count=min(count, 4))
                    results["web"][query] = [hit.__dict__ for hit in web_hits]
                for query in video_queries:
                    web_hits = await client.search_web(query, count=min(count, 4))
                    existing = results["web"].get(query, [])
                    existing.extend([hit.__dict__ for hit in web_hits])
                    results["web"][query] = existing

        if brave_api_key and (image_queries or video_queries):
            brave_results = await self._run_brave_media_queries(
                image_queries=image_queries,
                video_queries=video_queries,
                count=count,
                api_key=brave_api_key,
            )
            for media_type in ("image", "video"):
                results[media_type] = brave_results.get(media_type, {})
            if not exa_api_key:
                results["web"] = brave_results.get("web", {})

        return results

    async def _run_brave_media_queries(
        self,
        *,
        image_queries: List[str],
        video_queries: List[str],
        count: int,
        api_key: str,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"image": {}, "video": {}, "web": {}}
        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        timeout = aiohttp.ClientTimeout(total=20)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for query in image_queries:
                image_hits = await self._search_brave_endpoint(
                    session,
                    headers,
                    endpoint="images/search",
                    query=query,
                    count=count,
                )
                web_hits = await self._search_brave_endpoint(
                    session,
                    headers,
                    endpoint="web/search",
                    query=query,
                    count=min(count, 4),
                )
                results["image"][query] = image_hits
                results["web"][query] = web_hits

            for query in video_queries:
                video_hits = await self._search_brave_endpoint(
                    session,
                    headers,
                    endpoint="videos/search",
                    query=query,
                    count=count,
                )
                web_hits = await self._search_brave_endpoint(
                    session,
                    headers,
                    endpoint="web/search",
                    query=query,
                    count=min(count, 4),
                )
                results["video"][query] = video_hits
                existing = results["web"].get(query, [])
                existing.extend(web_hits)
                results["web"][query] = existing

        return results

    async def _search_brave_endpoint(
        self,
        session: aiohttp.ClientSession,
        headers: Dict[str, str],
        *,
        endpoint: str,
        query: str,
        count: int,
    ) -> List[Dict[str, str]]:
        url = f"https://api.search.brave.com/res/v1/{endpoint.lstrip('/')}"
        params = {"q": query, "count": max(1, min(int(count or 1), 20))}

        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status >= 400:
                    self.logger.warning(
                        "Brave %s failed status=%s for query=%s",
                        endpoint,
                        response.status,
                        query,
                    )
                    return []
                payload = await response.json(content_type=None)
        except Exception as exc:
            self.logger.warning(
                "Brave %s request failed for query=%s: %s",
                endpoint,
                query,
                exc,
            )
            return []

        return self._parse_brave_search_results(endpoint, payload)

    @staticmethod
    def _parse_brave_search_results(
        endpoint: str,
        payload: Any,
    ) -> List[Dict[str, str]]:
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

        parsed: List[Dict[str, str]] = []
        for item in candidates:
            url = (
                item.get("url")
                or item.get("link")
                or item.get("contentUrl")
                or item.get("content_url")
                or item.get("page_url")
                or ""
            )
            if not isinstance(url, str) or not url:
                continue

            thumbnail_url = (
                item.get("thumbnail")
                or item.get("thumbnail_url")
                or item.get("image")
                or ""
            )
            if isinstance(thumbnail_url, dict):
                thumbnail_url = (
                    thumbnail_url.get("url")
                    or thumbnail_url.get("src")
                    or str(thumbnail_url)
                )

            parsed.append(
                {
                    "title": str(item.get("title") or item.get("name") or ""),
                    "url": url,
                    "description": str(
                        item.get("description")
                        or item.get("snippet")
                        or item.get("text")
                        or ""
                    ),
                    "source": str(
                        item.get("source")
                        or item.get("domain")
                        or item.get("hostname")
                        or "unknown"
                    ),
                    "thumbnail_url": str(thumbnail_url or ""),
                }
            )

        return parsed

    async def _download_hybrid_media_assets(
        self,
        state,
        search_payload: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        project_dir = Path(state.project_dir)
        raw_media_dir = project_dir / "raw_media"
        image_dir = project_dir / "research" / "media_images"
        raw_media_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        max_video_downloads = int(os.getenv("HYBRID_MAX_VIDEO_DOWNLOADS", "8"))
        max_image_downloads = int(os.getenv("HYBRID_MAX_IMAGE_DOWNLOADS", "10"))

        def _next_index(folder: Path, pattern: str) -> int:
            """Return next 1-based numeric index for files matching regex pattern."""
            max_seen = 0
            try:
                for path in folder.glob("*"):
                    if not path.is_file():
                        continue
                    match = re.search(pattern, path.name)
                    if not match:
                        continue
                    try:
                        max_seen = max(max_seen, int(match.group(1)))
                    except (TypeError, ValueError):
                        continue
            except Exception:
                return 1
            return max_seen + 1

        next_video_index = _next_index(raw_media_dir, r"^video_(\d{3})_")
        next_image_index = _next_index(image_dir, r"^image_(\d{3})\.")

        downloaded: List[Dict[str, str]] = []
        seen_source_urls = self._load_existing_downloaded_source_urls(state)
        reserved_source_urls: set[str] = set()
        video_targets: List[Dict[str, str]] = []
        image_targets: List[Dict[str, Any]] = []

        for query, hits in (search_payload.get("video", {}) or {}).items():
            if not isinstance(hits, list):
                continue
            ranked_hits = sorted(
                [item for item in hits if isinstance(item, dict)],
                key=lambda item: self._video_source_rank(str(item.get("url", ""))),
            )
            for item in ranked_hits[:2]:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                normalized_url = self._normalize_media_source_url(url)
                if normalized_url and (
                    normalized_url in seen_source_urls
                    or normalized_url in reserved_source_urls
                ):
                    continue
                video_targets.append(
                    {
                        "query": query,
                        "url": url,
                        "title": str(item.get("title") or "video_evidence"),
                    }
                )
                if normalized_url:
                    reserved_source_urls.add(normalized_url)

        for query, hits in (search_payload.get("image", {}) or {}).items():
            if not isinstance(hits, list):
                continue
            for item in hits[:2]:
                if not isinstance(item, dict):
                    continue
                candidate_urls = self._extract_image_candidate_urls(item)
                if not candidate_urls:
                    continue

                filtered_candidates: List[str] = []
                filtered_normalized: List[str] = []
                for candidate_url in candidate_urls:
                    candidate_text = str(candidate_url).strip()
                    if not candidate_text:
                        continue
                    normalized_url = self._normalize_media_source_url(candidate_text)
                    if normalized_url and (
                        normalized_url in seen_source_urls
                        or normalized_url in reserved_source_urls
                    ):
                        continue
                    filtered_candidates.append(candidate_text)
                    filtered_normalized.append(normalized_url)

                if not filtered_candidates:
                    continue

                image_targets.append(
                    {
                        "query": query,
                        "url": filtered_candidates[0],
                        "url_candidates": filtered_candidates,
                        "title": str(item.get("title") or "image_evidence"),
                    }
                )
                for normalized_url in filtered_normalized:
                    if normalized_url:
                        reserved_source_urls.add(normalized_url)

        video_batch = video_targets[:max_video_downloads]
        video_total = len(video_batch)

        async def _download_video_task(
            b_pos: int, tgt: Dict[str, Any]
        ) -> Optional[Dict[str, str]]:
            idx = next_video_index + b_pos - 1
            d_path = await asyncio.to_thread(
                self._download_hybrid_video_hit,
                tgt["url"],
                raw_media_dir,
                idx,
            )
            if d_path is None:
                return None
            return {
                "media_type": "video",
                "source_url": tgt["url"],
                "query": tgt["query"],
                "title": tgt["title"],
                "local_path": str(d_path.relative_to(project_dir)),
            }

        video_tasks = [
            _download_video_task(batch_pos, target)
            for batch_pos, target in enumerate(video_batch, start=1)
        ]

        video_results = await asyncio.gather(*video_tasks, return_exceptions=True)
        for b_pos, res in enumerate(video_results, start=1):
            print(f"[Hybrid] Downloading video {b_pos}/{video_total}...", flush=True)
            if isinstance(res, Exception):
                print(
                    f"[Hybrid] Warning: Video download failed with exception: {res}",
                    flush=True,
                )
            elif res is not None:
                downloaded.append(res)

        image_batch = image_targets[:max_image_downloads]
        if image_batch:
            image_total = len(image_batch)
            timeout = aiohttp.ClientTimeout(total=25)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for batch_pos, target in enumerate(image_batch, start=1):
                    print(
                        f"[Hybrid] Downloading image {batch_pos}/{image_total}...",
                        flush=True,
                    )
                    index = next_image_index + batch_pos - 1
                    downloaded_path: Optional[Path] = None
                    selected_source_url = str(target.get("url", ""))
                    candidate_urls = target.get(
                        "url_candidates", [target.get("url", "")]
                    )
                    if not isinstance(candidate_urls, list):
                        candidate_urls = [target.get("url", "")]

                    for candidate_url in candidate_urls:
                        candidate_text = str(candidate_url).strip()
                        if not candidate_text:
                            continue
                        downloaded_path = await self._download_hybrid_image_hit(
                            session,
                            candidate_text,
                            image_dir,
                            index,
                        )
                        if downloaded_path is not None:
                            selected_source_url = candidate_text
                            break

                    if downloaded_path is None:
                        continue
                    downloaded.append(
                        {
                            "media_type": "image",
                            "source_url": selected_source_url,
                            "query": target["query"],
                            "title": target["title"],
                            "local_path": str(downloaded_path.relative_to(project_dir)),
                        }
                    )

        return downloaded

    @staticmethod
    def _normalize_media_source_url(url: str) -> str:
        text = str(url or "").strip()
        if not text:
            return ""
        parsed = urlparse(text)
        if not parsed.scheme or not parsed.netloc:
            return text.lower().rstrip("/")
        normalized_path = parsed.path.rstrip("/")
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                normalized_path,
                "",
                parsed.query,
                "",
            )
        )

    def _load_existing_downloaded_source_urls(self, state: Any) -> set[str]:
        state_meta = getattr(state, "metadata", {})
        evidence_index_path = str(
            state_meta.get("evidence_index_path", "") or ""
        ).strip()
        if not evidence_index_path:
            return set()

        evidence_path = Path(evidence_index_path)
        if not evidence_path.exists():
            return set()

        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception:
            return set()

        downloaded_media = payload.get("downloaded_media", [])
        if not isinstance(downloaded_media, list):
            return set()

        urls: set[str] = set()
        for item in downloaded_media:
            if not isinstance(item, dict):
                continue
            normalized_url = self._normalize_media_source_url(
                str(item.get("source_url", "") or "")
            )
            if normalized_url:
                urls.add(normalized_url)
        return urls

    def _downloaded_media_entry_key(self, entry: Any) -> str:
        if not isinstance(entry, dict):
            return ""
        normalized_source = self._normalize_media_source_url(
            str(entry.get("source_url", "") or "")
        )
        if normalized_source:
            return f"url:{normalized_source}"
        local_path = str(entry.get("local_path", "") or "").strip()
        if local_path:
            return f"path:{local_path}"
        return ""

    def _merge_downloaded_media_entries(
        self,
        existing_entries: Any,
        new_entries: Any,
    ) -> List[Dict[str, str]]:
        merged: List[Dict[str, str]] = []
        seen_keys: set[str] = set()

        existing_list = existing_entries if isinstance(existing_entries, list) else []
        for item in existing_list:
            if not isinstance(item, dict):
                continue
            key = self._downloaded_media_entry_key(item)
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            merged.append(item)

        new_list = new_entries if isinstance(new_entries, list) else []
        for item in new_list:
            if not isinstance(item, dict):
                continue
            key = self._downloaded_media_entry_key(item)
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            merged.append(item)

        return merged

    def _download_hybrid_video_hit(
        self,
        url: str,
        raw_media_dir: Path,
        index: int,
    ) -> Optional[Path]:
        safe_host = self._slugify_for_filename(urlparse(url).netloc or "web")
        output_base = raw_media_dir / f"video_{index:03d}_{safe_host}"
        success = self.video_processor.downloader.download_video(url, output_base)
        if not success:
            return None

        for candidate in output_base.parent.glob(f"{output_base.stem}.*"):
            if candidate.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
                return candidate
        return None

    @staticmethod
    def _video_source_rank(url: str) -> int:
        host = (urlparse(str(url or "")).netloc or "").lower()
        if "youtube.com" in host or "youtu.be" in host:
            return 2
        if "reddit.com" in host or "v.redd.it" in host:
            return 1
        return 0

    def _extract_image_candidate_urls(self, item: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []

        thumbnail_raw = item.get("thumbnail_url")
        candidates.extend(self._parse_thumbnail_candidate_urls(thumbnail_raw))

        primary_url = str(item.get("url", "") or "").strip()
        if primary_url:
            if self._is_probable_image_url(primary_url):
                candidates.insert(0, primary_url)
            else:
                candidates.append(primary_url)

        deduped: List[str] = []
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text or text in deduped:
                continue
            if text.startswith("http://") or text.startswith("https://"):
                deduped.append(text)
        return deduped

    @staticmethod
    def _parse_thumbnail_candidate_urls(thumbnail_raw: Any) -> List[str]:
        candidates: List[str] = []
        if isinstance(thumbnail_raw, dict):
            src = thumbnail_raw.get("src") or thumbnail_raw.get("url")
            if isinstance(src, str) and src.strip():
                candidates.append(src.strip())
            return candidates

        if not isinstance(thumbnail_raw, str):
            return candidates

        text = thumbnail_raw.strip()
        if not text:
            return candidates

        if text.startswith("http://") or text.startswith("https://"):
            candidates.append(text)
            return candidates

        if not text.startswith("{"):
            return candidates

        try:
            payload = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return candidates

        if isinstance(payload, dict):
            src = payload.get("src") or payload.get("url")
            if isinstance(src, str) and src.strip():
                candidates.append(src.strip())
        return candidates

    @staticmethod
    def _is_probable_image_url(url: str) -> bool:
        text = str(url or "").strip().lower()
        if not text:
            return False
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".svg"):
            if ext in text:
                return True
        host = (urlparse(text).netloc or "").lower()
        return "img" in host or "image" in host or "ytimg" in host

    async def _download_hybrid_image_hit(
        self,
        session: aiohttp.ClientSession,
        url: str,
        image_dir: Path,
        index: int,
    ) -> Optional[Path]:
        try:
            async with session.get(url) as response:
                if response.status >= 400:
                    return None
                content_type = str(response.headers.get("content-type", "")).lower()
                if not content_type.startswith("image/"):
                    return None
                payload = await response.read()
                if not payload:
                    return None

                extension = ".jpg"
                if "png" in content_type:
                    extension = ".png"
                elif "webp" in content_type:
                    extension = ".webp"
                elif "gif" in content_type:
                    extension = ".gif"

                filename = f"image_{index:03d}{extension}"
                out_path = image_dir / filename
                out_path.write_bytes(payload)
                return out_path
        except Exception:
            return None

    async def _transcribe_hybrid_media_assets(
        self,
        state,
        downloaded_media: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        api_key = (
            getattr(self.config.api, "nvidia_nim_api_key", None)
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("NVIDIA_NIM_API_KEY")
        )
        if not api_key:
            return []

        max_files = int(os.getenv("HYBRID_MAX_TRANSCRIBE_FILES", "6"))
        outputs: List[Dict[str, str]] = []
        project_dir = Path(state.project_dir)

        eligible = [
            item
            for item in downloaded_media
            if item.get("media_type") == "video"
            and item.get("local_path", "")
            .lower()
            .endswith((".mp4", ".webm", ".mkv", ".mov", ".avi"))
        ]
        transcribe_total = len(eligible[:max_files])

        for idx, item in enumerate(eligible[:max_files], start=1):
            print(f"[Hybrid] Transcribing {idx}/{transcribe_total}...", flush=True)
            local_path = project_dir / item["local_path"]
            try:
                transcript = await asyncio.to_thread(
                    transcribe_media_file,
                    local_path,
                    state.project_dir,
                    api_key=api_key,
                )
            except Exception:
                continue

            outputs.append(
                {
                    "media": item["local_path"],
                    "transcript_text_path": str(
                        Path(transcript.transcript_text_path).relative_to(project_dir)
                    ),
                    "transcript_json_path": str(
                        Path(transcript.transcript_json_path).relative_to(project_dir)
                    ),
                    "transcript_srt_path": (
                        str(
                            Path(transcript.transcript_srt_path).relative_to(
                                project_dir
                            )
                        )
                        if transcript.transcript_srt_path
                        else ""
                    ),
                }
            )

        return outputs

    def _collect_hybrid_workspace_assets(self, state) -> List[str]:
        base = Path(state.project_dir)
        assets: List[str] = []
        for subdir in [
            "raw_media",
            "research/media_images",
            "research/media_videos",
            "research/media/images",
            "research/media/videos",
            "research/transcripts",
            "research/evidence",
        ]:
            root = base / subdir
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    assets.append(str(path.relative_to(base)))
        assets.sort()
        return assets

    async def _apply_hybrid_image_relevance_mapping(
        self,
        state: Any,
        script_payload: Dict[str, Any],
        assets: List[str],
    ) -> Dict[str, Any]:
        self._enforce_script_asset_mapping(script_payload, assets)

        segments = script_payload.get("segments")
        if not isinstance(segments, list) or not segments:
            return {}

        visual_assets = self._filter_visual_assets(assets)
        if not visual_assets:
            return {}

        min_score = self._read_env_int(
            "HYBRID_IMAGE_RELEVANCE_MIN_SCORE",
            default=70,
            minimum=1,
            maximum=100,
        )
        top_k = self._read_env_int(
            "HYBRID_IMAGE_RELEVANCE_TOP_K",
            default=3,
            minimum=1,
            maximum=12,
        )
        max_calls = self._read_env_int(
            "HYBRID_IMAGE_RELEVANCE_MAX_CALLS",
            default=40,
            minimum=1,
            maximum=120,
        )
        min_approved_images = self._read_env_int(
            "HYBRID_IMAGE_RELEVANCE_MIN_APPROVED_PER_SEGMENT",
            default=1,
            minimum=1,
            maximum=12,
        )
        remaining_calls = max_calls

        media_metadata = self._load_hybrid_media_metadata(state)
        project_dir = Path(getattr(state, "project_dir", ""))
        if not project_dir:
            return {}

        report_segments: List[Dict[str, Any]] = []

        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue

            ranked_candidates = self._rank_visual_candidates_for_segment(
                segment,
                visual_assets,
                media_metadata,
                top_k=top_k,
            )
            current_asset = str(segment.get("visual_asset_path", "") or "").strip()
            if current_asset and current_asset not in ranked_candidates:
                ranked_candidates = [current_asset, *ranked_candidates]

            candidate_scores: List[Dict[str, Any]] = []
            approved_candidates: List[Dict[str, Any]] = []
            approved_image_sources: set[str] = set()
            scanned_assets: set[str] = set()

            for candidate_asset in ranked_candidates:
                if remaining_calls <= 0:
                    break
                candidate_path = project_dir / candidate_asset
                if not candidate_path.exists() or not candidate_path.is_file():
                    continue

                score_result = await self._score_hybrid_visual_relevance(
                    candidate_path,
                    segment,
                    media_metadata.get(candidate_asset, {}),
                    project_dir=project_dir,
                )
                remaining_calls -= 1
                scanned_assets.add(candidate_asset)

                score = int(score_result.get("score", 0) or 0)
                relevant = bool(score_result.get("relevant", score >= min_score))
                reason = str(score_result.get("reason", "") or "")
                row = {
                    "asset_path": candidate_asset,
                    "media_type": self._media_type_for_asset(candidate_asset),
                    "score": score,
                    "relevant": relevant,
                    "reason": reason,
                    "source_url": str(
                        media_metadata.get(candidate_asset, {}).get("source_url", "")
                        or ""
                    ),
                }
                candidate_scores.append(row)
                if score >= min_score and relevant:
                    approved_candidates.append(row)
                    if row["media_type"] == "image":
                        image_key = self._normalize_media_source_url(
                            str(row.get("source_url", "") or "")
                        )
                        approved_image_sources.add(image_key or candidate_asset)

            selected_asset = str(segment.get("visual_asset_path", "") or "")
            selected_score: Optional[int] = None

            if approved_candidates:
                best = max(approved_candidates, key=lambda item: int(item["score"]))
                selected_asset = str(best["asset_path"])
                selected_score = int(best["score"])
                segment["visual_asset_path"] = selected_asset
            elif selected_asset and selected_asset in scanned_assets:
                selected_asset = ""
                segment["visual_asset_path"] = ""

            report_segments.append(
                {
                    "segment_index": idx,
                    "narration": str(segment.get("narration", "") or ""),
                    "min_score": min_score,
                    "min_approved_images": min_approved_images,
                    "selected_asset_path": selected_asset,
                    "selected_score": selected_score,
                    "approved_count_total": len(approved_candidates),
                    "approved_count_images": len(approved_image_sources),
                    "candidates": candidate_scores,
                }
            )

        report_payload = {
            "phase": PipelinePhase.SCRIPTING.value,
            "created_at": datetime.now().isoformat(),
            "min_score": min_score,
            "top_k": top_k,
            "max_calls": max_calls,
            "remaining_calls": remaining_calls,
            "segments": report_segments,
        }
        save_finding(
            state.project_dir,
            "evidence",
            "media_relevance_report.json",
            report_payload,
        )
        return report_payload

    async def _ensure_min_approved_images_per_segment(
        self,
        state: Any,
        script_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Iteratively search/download more images until each segment has enough approved images."""
        min_approved_images = self._read_env_int(
            "HYBRID_IMAGE_RELEVANCE_MIN_APPROVED_PER_SEGMENT",
            default=1,
            minimum=1,
            maximum=12,
        )
        max_retries = self._read_env_int(
            "HYBRID_IMAGE_RELEVANCE_MAX_RETRIES",
            default=1,
            minimum=0,
            maximum=8,
        )

        evidence_index_path = str(
            getattr(state, "metadata", {}).get("evidence_index_path", "") or ""
        ).strip()
        evidence_payload: Dict[str, Any] = {}
        if evidence_index_path and Path(evidence_index_path).exists():
            try:
                evidence_payload = json.loads(
                    Path(evidence_index_path).read_text(encoding="utf-8")
                )
            except Exception:
                evidence_payload = {}

        def _list_from_payload(key: str) -> List[str]:
            raw = evidence_payload.get(key, [])
            if not isinstance(raw, list):
                return []
            return [str(item).strip() for item in raw if str(item).strip()]

        existing_queries = _list_from_payload("image_queries")
        existing_queries.extend(_list_from_payload("supplementary_image_queries"))

        last_report: Dict[str, Any] = {}

        for attempt in range(max_retries + 1):
            workspace_assets = self._collect_hybrid_workspace_assets(state)
            last_report = await self._apply_hybrid_image_relevance_mapping(
                state,
                script_payload,
                workspace_assets,
            )
            segments_report = last_report.get("segments", [])
            if not isinstance(segments_report, list):
                return last_report

            under_served: List[int] = []
            for row in segments_report:
                if not isinstance(row, dict):
                    continue
                idx = row.get("segment_index")
                if not isinstance(idx, int) or idx < 0:
                    continue
                approved_images = int(row.get("approved_count_images", 0) or 0)
                if approved_images < min_approved_images:
                    under_served.append(idx)
            if not under_served:
                return last_report

            if attempt >= max_retries:
                return last_report

            supplementary_queries = await self._generate_supplementary_image_queries(
                state,
                script_payload,
                segment_indexes=sorted(set(under_served)),
                existing_queries=existing_queries,
                max_per_segment=3,
            )
            existing_lower = {x.lower() for x in existing_queries}
            supplementary_queries = [
                str(q).strip()
                for q in supplementary_queries
                if str(q).strip() and str(q).strip().lower() not in existing_lower
            ]
            if not supplementary_queries:
                return last_report

            search_payload = await self._run_hybrid_media_queries(
                image_queries=supplementary_queries,
                video_queries=[],
                count=6,
            )
            if not isinstance(search_payload, dict) or not any(
                (search_payload.get("image") or {}).values()
            ):
                return last_report

            retry_search_path = save_finding(
                state.project_dir,
                "evidence",
                f"media_search_results_retry_{attempt + 1}.json",
                search_payload,
            )

            downloaded_media = await self._download_hybrid_media_assets(
                state,
                search_payload,
            )
            if not downloaded_media:
                return last_report

            evidence_payload["downloaded_media"] = self._merge_downloaded_media_entries(
                evidence_payload.get("downloaded_media", []),
                downloaded_media,
            )

            # Track supplementary queries + retry artifacts for traceability
            supp_list = evidence_payload.get("supplementary_image_queries", [])
            if not isinstance(supp_list, list):
                supp_list = []
            for q in supplementary_queries:
                if q not in supp_list:
                    supp_list.append(q)
            evidence_payload["supplementary_image_queries"] = supp_list

            retry_paths = evidence_payload.get("supplementary_search_results_paths", [])
            if not isinstance(retry_paths, list):
                retry_paths = []
            retry_paths.append(
                str(Path(retry_search_path).relative_to(Path(state.project_dir)))
            )
            evidence_payload["supplementary_search_results_paths"] = retry_paths

            existing_queries.extend([q for q in supplementary_queries if q])
            evidence_payload["updated_at"] = datetime.now().isoformat()

            evidence_path = save_finding(
                state.project_dir,
                "evidence",
                "evidence_index.json",
                evidence_payload,
            )
            if not hasattr(state, "metadata") or not isinstance(state.metadata, dict):
                state.metadata = {}
            state.metadata["evidence_index_path"] = str(evidence_path)
            save_run_state(state)

        return last_report

    async def _judge_script_quality(
        self,
        script_payload: Dict[str, Any],
        max_judge_retries: int = 1,
    ) -> Dict[str, Any]:
        """LLM Judge: Evaluate script quality, auto-rewrite if score < 8."""
        segments = script_payload.get("segments", [])
        if not isinstance(segments, list) or not segments:
            return script_payload

        ai_client = getattr(self, "ai_client", None)
        active_client = getattr(ai_client, "active_client", None) if ai_client else None
        if not active_client or not hasattr(
            active_client, "_chat_completion_with_fallback"
        ):
            return script_payload

        judge_prompt = """You are a strict YouTube Shorts script judge. Evaluate this script.
Rate it 1-10 based on:
1. Hook is under 3 seconds (first segment must grab attention immediately)
2. Sentences are short and punchy (max 10-15 words per segment)
3. Sounds natural/conversational (not like a robot)
4. No forbidden phrases: "Did you know?", "In today's video", "Wait for it", "Let's dive in", "Mind-blowing"
5. The "Perfect Loop" mechanic: last sentence flows back into the hook

Return strict JSON:
{
  "score": <1-10>,
  "hook_under_3_seconds": true/false,
  "sounds_natural": true/false,
  "sentences_too_long": true/false,
  "has_forbidden_phrases": true/false,
  "feedback": "Brief explanation of issues",
  "passes_quality_bar": true/false
}

Script to judge:
"""
        for attempt in range(max_judge_retries + 1):
            try:
                import json as _json

                script_json = _json.dumps(script_payload, indent=2, ensure_ascii=True)[
                    :4000
                ]
                full_prompt = judge_prompt + script_json

                response = await active_client._chat_completion_with_fallback(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict script quality judge. Return JSON only.",
                        },
                        {"role": "user", "content": full_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                parsed = _json.loads(content)
                score = int(parsed.get("score", 0))
                passes = bool(parsed.get("passes_quality_bar", score >= 8))

                print(
                    f"[Hybrid] Script Judge score: {score}/10 (passes={passes})",
                    flush=True,
                )
                if parsed.get("feedback"):
                    print(f"[Hybrid] Judge feedback: {parsed['feedback']}", flush=True)

                if passes or attempt >= max_judge_retries:
                    return script_payload

                rewrite_prompt = (
                    f"The script judge gave a score of {score}/10 and said: {parsed.get('feedback', '')}\n\n"
                    f"Rewrite the script to address these issues. Keep all factual details intact. "
                    f"Make the hook grab attention in the first 3 seconds. Shorten long sentences. "
                    f"Ensure the last sentence loops back to the hook. "
                    f"Remove any forbidden phrases. Return the full corrected script JSON."
                )

                print(
                    f"[Hybrid] Judge score < 8, rewriting script (attempt {attempt + 1})...",
                    flush=True,
                )
                response = await active_client._chat_completion_with_fallback(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a top YouTube Shorts scriptwriter. Rewrite to fix issues. Return JSON only.",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Original script:\n{script_json}\n\n"
                                f"Rewrite instructions:\n{rewrite_prompt}"
                            ),
                        },
                    ],
                    temperature=0.4,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                rewritten = self._safe_parse_phase_json(content)
                if (
                    rewritten
                    and isinstance(rewritten, dict)
                    and rewritten.get("segments")
                ):
                    script_payload = rewritten
                    print("[Hybrid] Script rewritten by judge feedback.", flush=True)
                else:
                    break
            except Exception as exc:
                self.logger.warning("Script judge failed: %s", exc)
                break

        return script_payload

    async def _run_agentic_scripting_with_visual_loop(
        self,
        state: Any,
        phase: PipelinePhase,
        context: str,
    ) -> Dict[str, Any]:
        """Run SCRIPTING with a visual feedback loop before final relevance mapping."""
        script_payload = await self._generate_hybrid_phase_payload(
            state,
            phase,
            context,
        )

        max_rounds = self._read_env_int(
            "HYBRID_VISUAL_FEEDBACK_MAX_ROUNDS",
            default=0,
            minimum=0,
            maximum=8,
        )
        if max_rounds <= 0:
            await self._ensure_min_approved_images_per_segment(state, script_payload)
            judge_enabled = (
                str(os.getenv("HYBRID_SCRIPT_JUDGE_ENABLED", "true")).strip().lower()
            )
            if judge_enabled not in {"0", "false", "no", "off"}:
                script_payload = await self._judge_script_quality(script_payload)
            return script_payload

        flag_raw = (
            str(os.getenv("HYBRID_VISUAL_REVISION_ENABLED", "true")).strip().lower()
        )
        allow_revisions = flag_raw not in {"0", "false", "no", "off"}
        issued_image_queries: set[str] = set()

        for _ in range(max_rounds):
            visual_catalog = self._build_visual_catalog_for_script(state)
            images = visual_catalog.get("images", [])
            if not isinstance(images, list) or not images:
                break

            feedback = await self._run_visual_feedback_agent(
                state,
                script_payload,
                visual_catalog,
            )
            if not isinstance(feedback, dict):
                break
            fb_segments = feedback.get("segments")
            if not isinstance(fb_segments, list) or not fb_segments:
                break

            # Apply revisions and collect new image queries.
            self._apply_visual_feedback_to_script(
                script_payload,
                feedback,
                allow_revisions=allow_revisions,
            )

            new_image_queries: List[str] = []
            needs_images = False
            for entry in fb_segments:
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status", "") or "").strip().lower()
                if status == "needs_images":
                    needs_images = True
                queries = entry.get("new_image_queries") or []
                if isinstance(queries, list):
                    for q in queries:
                        text = str(q or "").strip()
                        if text:
                            new_image_queries.append(text)

            if not needs_images or not new_image_queries:
                break

            # Deduplicate queries.
            seen_queries: set[str] = set()
            deduped_queries: List[str] = []
            for q in new_image_queries:
                lowered = q.lower()
                if lowered in seen_queries or lowered in issued_image_queries:
                    continue
                seen_queries.add(lowered)
                deduped_queries.append(q)
            if not deduped_queries:
                break

            issued_image_queries.update([q.lower() for q in deduped_queries])

            search_payload = await self._run_hybrid_media_queries(
                image_queries=deduped_queries,
                video_queries=[],
                count=6,
            )
            if not isinstance(search_payload, dict) or not any(
                (search_payload.get("image") or {}).values()
            ):
                break

            evidence_index_path = str(
                getattr(state, "metadata", {}).get("evidence_index_path", "") or ""
            ).strip()
            evidence_payload: Dict[str, Any] = {}
            if evidence_index_path and Path(evidence_index_path).exists():
                try:
                    evidence_payload = json.loads(
                        Path(evidence_index_path).read_text(encoding="utf-8")
                    )
                except Exception:
                    evidence_payload = {}
            if "downloaded_media" not in evidence_payload or not isinstance(
                evidence_payload.get("downloaded_media"), list
            ):
                evidence_payload["downloaded_media"] = []

            retry_search_path = save_finding(
                state.project_dir,
                "evidence",
                "media_search_results_visual_feedback.json",
                search_payload,
            )

            downloaded_media = await self._download_hybrid_media_assets(
                state,
                search_payload,
            )
            if not downloaded_media:
                break

            evidence_payload["downloaded_media"] = self._merge_downloaded_media_entries(
                evidence_payload.get("downloaded_media", []),
                downloaded_media,
            )

            # Track feedback-driven queries and search artifacts.
            supp_list = evidence_payload.get("supplementary_image_queries", [])
            if not isinstance(supp_list, list):
                supp_list = []
            for q in deduped_queries:
                if q not in supp_list:
                    supp_list.append(q)
            evidence_payload["supplementary_image_queries"] = supp_list

            retry_paths = evidence_payload.get("supplementary_search_results_paths", [])
            if not isinstance(retry_paths, list):
                retry_paths = []
            retry_paths.append(
                str(Path(retry_search_path).relative_to(Path(state.project_dir)))
            )
            evidence_payload["supplementary_search_results_paths"] = retry_paths

            evidence_payload["updated_at"] = datetime.now().isoformat()

            evidence_path = save_finding(
                state.project_dir,
                "evidence",
                "evidence_index.json",
                evidence_payload,
            )
            if not hasattr(state, "metadata") or not isinstance(state.metadata, dict):
                state.metadata = {}
            state.metadata["evidence_index_path"] = str(evidence_path)
            save_run_state(state)

        await self._ensure_min_approved_images_per_segment(state, script_payload)

        judge_enabled = (
            str(os.getenv("HYBRID_SCRIPT_JUDGE_ENABLED", "true")).strip().lower()
        )
        if judge_enabled not in {"0", "false", "no", "off"}:
            script_payload = await self._judge_script_quality(script_payload)

        return script_payload

    async def _render_hybrid_local_video(
        self,
        state: Any,
        script_payload: Dict[str, Any],
        render_attempt: int = 0,
    ) -> Dict[str, Any]:
        project_dir = Path(str(getattr(state, "project_dir", "") or "")).resolve()
        if not project_dir.exists():
            return {
                "success": False,
                "error": f"Project directory not found: {project_dir}",
            }

        segments = script_payload.get("segments")
        if not isinstance(segments, list) or not segments:
            return {
                "success": False,
                "error": "No script segments available for hybrid render",
            }

        # Apply cinematic editor pacing adjustment
        cinematic_editor = getattr(self, "cinematic_editor", None)
        if cinematic_editor is not None and hasattr(cinematic_editor, "adjust_pacing"):
            try:
                segments = cinematic_editor.adjust_pacing(segments)
            except Exception as exc:
                self.logger.warning("CinematicEditor pacing adjustment failed: %s", exc)

        min_segment_duration = self._read_env_float(
            "HYBRID_MIN_SEGMENT_DURATION_SECONDS",
            default=2.0,
            minimum=0.8,
            maximum=30.0,
        )
        max_segment_duration = self._read_env_float(
            "HYBRID_MAX_SEGMENT_DURATION_SECONDS",
            default=14.0,
            minimum=3.0,
            maximum=60.0,
        )
        narration_multiplier = self._read_env_float(
            "HYBRID_NARRATION_DURATION_MULTIPLIER",
            default=2.0,
            minimum=1.0,
            maximum=4.0,
        )

        created_clips: List[Any] = []
        timeline_clips: List[Any] = []
        overlays: List[TextOverlay] = []
        timeline_manifest: List[Dict[str, Any]] = []
        tts_jobs: List[Dict[str, Any]] = []

        timeline_clip: Optional[Any] = None
        rendered_clip: Optional[Any] = None
        composite_audio: Optional[Any] = None

        def _register_clip(clip: Any) -> None:
            if clip is not None:
                created_clips.append(clip)

        print("[Hybrid] Rendering final video locally...", flush=True)

        try:
            # ──────────────────────────────────────────────────────────
            # PASS 1: Build narrative segments, generate TTS, measure real durations
            # ──────────────────────────────────────────────────────────
            segment_meta: List[Dict[str, Any]] = []
            for idx, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    continue

                estimate = self._normalize_segment_duration(
                    segment,
                    min_segment_duration=min_segment_duration,
                    max_segment_duration=max_segment_duration,
                    narration_multiplier=narration_multiplier,
                )

                narration = str(segment.get("narration", "") or "").strip()
                asset_rel = str(segment.get("visual_asset_path", "") or "").strip()
                asset_abs = (project_dir / asset_rel) if asset_rel else None
                source_type = "fallback"
                if asset_abs and asset_abs.exists() and asset_abs.is_file():
                    media_type = self._media_type_for_asset(asset_rel)
                    source_type = media_type if media_type != "unknown" else "fallback"

                emotion_raw = str(segment.get("emotion", "") or "").strip().lower()
                if emotion_raw not in {
                    EmotionType.EXCITED.value,
                    EmotionType.CALM.value,
                    EmotionType.DRAMATIC.value,
                    EmotionType.NEUTRAL.value,
                }:
                    emotion_raw = EmotionType.NEUTRAL.value

                pace_raw = (
                    str(segment.get("pace", segment.get("pacing", "")) or "")
                    .strip()
                    .lower()
                )
                if pace_raw not in {
                    PacingType.FAST.value,
                    PacingType.NORMAL.value,
                    PacingType.SLOW.value,
                }:
                    pace_raw = PacingType.NORMAL.value

                emotion_value = EmotionType.NEUTRAL
                if emotion_raw == EmotionType.EXCITED.value:
                    emotion_value = EmotionType.EXCITED
                elif emotion_raw == EmotionType.CALM.value:
                    emotion_value = EmotionType.CALM
                elif emotion_raw == EmotionType.DRAMATIC.value:
                    emotion_value = EmotionType.DRAMATIC

                pace_value = PacingType.NORMAL
                if pace_raw == PacingType.FAST.value:
                    pace_value = PacingType.FAST
                elif pace_raw == PacingType.SLOW.value:
                    pace_value = PacingType.SLOW

                expression_cue = (
                    str(segment.get("expression_cue", "") or "").strip() or None
                )

                text_overlay = str(segment.get("text_overlay", "") or "").strip()
                if text_overlay:
                    try:
                        overlays.append(
                            TextOverlay(
                                text=text_overlay[:200],
                                timestamp_seconds=0.0,  # will patch after real timing
                                duration=max(0.8, min(estimate, 4.0)),
                                position=PositionType.CENTER,
                                style=TextStyle.HIGHLIGHT,
                            )
                        )
                    except Exception:
                        self.logger.debug(
                            "Skipping invalid text overlay at segment index %s", idx
                        )

                meta_entry = {
                    "idx": idx,
                    "narration": narration,
                    "asset_rel": asset_rel,
                    "asset_abs": asset_abs,
                    "source_type": source_type,
                    "estimate": estimate,
                    "emotion_value": emotion_value,
                    "pace_value": pace_value,
                    "text_overlay": text_overlay,
                }
                segment_meta.append(meta_entry)

                if narration:
                    try:
                        tts_segment = NarrativeSegment(
                            text=narration,
                            time_seconds=0.0,
                            intended_duration_seconds=estimate,
                            emotion=emotion_value,
                            pacing=pace_value,
                            expression_cue=expression_cue,
                        )
                        tts_jobs.append(
                            {
                                "index": idx,
                                "segment": tts_segment,
                            }
                        )
                    except Exception:
                        self.logger.debug(
                            "Skipping invalid narration segment at index %s", idx
                        )

            # Generate TTS audio for ALL segments
            actual_durations: Dict[int, float] = {}
            tts_audio_map: Dict[int, Any] = {}
            if tts_jobs:
                tts_segments_list = [job["segment"] for job in tts_jobs]
                tts_results = await asyncio.to_thread(
                    self.advanced_audio_processor.tts_service.generate_multiple_segments,
                    tts_segments_list,
                )
                for job, tts_result in zip(tts_jobs, tts_results):
                    idx = job["index"]
                    if not isinstance(tts_result, dict) or not tts_result.get(
                        "success"
                    ):
                        meta = segment_meta[idx]
                        actual_durations[idx] = max(meta["estimate"], 0.5)
                        continue
                    audio_path = tts_result.get("audio_path")
                    if not audio_path:
                        meta = segment_meta[idx]
                        actual_durations[idx] = max(meta["estimate"], 0.5)
                        continue
                    try:
                        clip = AudioFileClip(str(audio_path))
                        _register_clip(clip)
                        actual = clip.duration + 0.2  # breathing room
                        actual_durations[idx] = max(actual, 0.5)
                        tts_audio_map[idx] = clip
                    except Exception as exc:
                        self.logger.warning(
                            "Failed to load TTS audio for segment %s: %s", idx, exc
                        )
                        meta = segment_meta[idx]
                        actual_durations[idx] = max(meta["estimate"], 0.5)

            # Fallback: any segment_meta without a duration gets its estimate
            for m in segment_meta:
                if m["idx"] not in actual_durations:
                    actual_durations[m["idx"]] = max(m["estimate"], 0.5)

            # ──────────────────────────────────────────────────────────
            # PASS 2: Build video timeline matching real audio durations
            # ──────────────────────────────────────────────────────────
            cursor = 0.0
            for m in segment_meta:
                idx = m["idx"]
                duration = actual_durations[idx]
                start_time = round(cursor, 3)
                cursor += duration

                # Update overlay timestamp
                for ov in overlays:
                    if ov.timestamp_seconds == 0.0 and idx == m["idx"]:
                        ov.timestamp_seconds = start_time
                        break

                segment_clip: Optional[Any] = None
                source_type = m["source_type"]
                asset_abs = m["asset_abs"]
                asset_rel = m["asset_rel"]

                if asset_abs and asset_abs.exists() and asset_abs.is_file():
                    media_type = self._media_type_for_asset(asset_rel)
                    if media_type == "video":
                        try:
                            source_video = VideoFileClip(str(asset_abs))
                            _register_clip(source_video)
                            clip_duration = float(
                                getattr(source_video, "duration", 0.0) or 0.0
                            )
                            if clip_duration > 0 and clip_duration >= duration:
                                segment_clip = MoviePyCompat.subclip(
                                    source_video, 0, duration
                                )
                            elif clip_duration > 0:
                                loop_fn = getattr(source_video, "loop", None)
                                if callable(loop_fn):
                                    segment_clip = loop_fn(duration=duration)
                                else:
                                    segment_clip = MoviePyCompat.with_duration(
                                        source_video, duration
                                    )
                            if segment_clip is not None:
                                source_type = "video"
                        except Exception as exc:
                            self.logger.warning(
                                "Hybrid render video segment failed for %s: %s",
                                asset_abs,
                                exc,
                            )
                    elif media_type == "image":
                        try:
                            segment_clip = ImageClip(str(asset_abs))
                            segment_clip = MoviePyCompat.with_duration(
                                segment_clip, duration
                            )
                            # Ken Burns subtle zoom: 1.0x -> 1.05x over duration
                            if duration > 0:
                                zoom = lambda t, dur=duration: 1 + 0.05 * min(
                                    t / dur, 1.0
                                )
                                segment_clip = segment_clip.resized(zoom)
                            source_type = "image"
                        except Exception as exc:
                            self.logger.warning(
                                "Hybrid render image segment failed for %s: %s",
                                asset_abs,
                                exc,
                            )

                if segment_clip is None:
                    try:
                        background_clip = BackgroundManager().get_sliced_background(
                            target_duration=duration,
                            subreddit="hybrid",
                            text_content=m["narration"],
                        )
                        from moviepy import ColorClip as _ColorClip

                        overlay = _ColorClip(
                            size=(1080, 1920),
                            color=(0, 0, 0),
                            duration=duration,
                        )
                        from moviepy import CompositeVideoClip as _CompositeVideoClip

                        segment_clip = _CompositeVideoClip([background_clip, overlay])
                        source_type = "background"
                    except Exception as exc:
                        self.logger.warning(
                            "Hybrid render background fallback failed for segment %s: %s",
                            idx,
                            exc,
                        )
                        segment_clip = ColorClip(
                            size=(1080, 1920),
                            color=(18, 18, 18),
                            duration=duration,
                        )
                        source_type = "color"

                segment_clip = ensure_shorts_format(
                    segment_clip, target_duration=duration
                )
                _register_clip(segment_clip)
                timeline_clips.append(segment_clip)

                timeline_manifest.append(
                    {
                        "segment_index": idx,
                        "start_time": start_time,
                        "duration": duration,
                        "source_type": source_type,
                        "visual_asset_path": asset_rel,
                    }
                )

            if not timeline_clips:
                return {
                    "success": False,
                    "error": "No valid timeline clips built from script segments",
                }

            timeline_clip = concatenate_videoclips(timeline_clips, method="compose")
            _register_clip(timeline_clip)
            rendered_clip = timeline_clip
            if rendered_clip is None:
                return {
                    "success": False,
                    "error": "Failed to build concatenated render timeline",
                }

            # ──────────────────────────────────────────────────────────
            # Audio assembly: layer TTS at real start times + SFX
            # ──────────────────────────────────────────────────────────
            from src.processing.sound_effects_manager import SoundEffectsManager

            sfx_mgr = SoundEffectsManager()
            audio_layers: List[Any] = []
            cum_time = 0.0
            for m in segment_meta:
                idx = m["idx"]
                duration = actual_durations[idx]
                start_time = cum_time
                cum_time += duration

                # Place TTS audio
                clip = tts_audio_map.get(idx)
                if clip is not None:
                    positioned = MoviePyCompat.with_start(clip, float(start_time))
                    audio_layers.append(positioned)

                # SFX: whoosh at b-roll transitions (image/video appearing)
                if m["source_type"] in ("image", "video") and start_time > 0:
                    whoosh_path = sfx_mgr.get_whoosh_sound()
                    if whoosh_path:
                        try:
                            whoosh = AudioFileClip(str(whoosh_path))
                            whoosh = whoosh.with_start(start_time).with_volume_scaled(
                                0.3
                            )
                            audio_layers.append(whoosh)
                            _register_clip(whoosh)
                        except Exception:
                            pass

            # Boom at hook (0.0) for first segment
            boom_path = sfx_mgr.get_boom_sound()
            if boom_path:
                try:
                    boom = AudioFileClip(str(boom_path))
                    boom = boom.with_start(0.0).with_volume_scaled(0.6)
                    audio_layers.append(boom)
                    _register_clip(boom)
                except Exception:
                    pass

            if audio_layers:
                composite_audio = CompositeAudioClip(audio_layers)
                _register_clip(composite_audio)
                # Perfect loop: crossfade tail into head
                loopable = EnhancedVideoOrchestrator.make_loopable_audio(
                    composite_audio
                )
                rendered_clip = MoviePyCompat.with_audio(rendered_clip, loopable)

            if rendered_clip is None:
                return {
                    "success": False,
                    "error": "Render clip unexpectedly unavailable before overlays",
                }

            if overlays:
                try:
                    rendered_clip = (
                        self.video_processor.text_processor.add_text_overlays(
                            rendered_clip,
                            overlays,
                        )
                    )
                    _register_clip(rendered_clip)
                except Exception as exc:
                    self.logger.warning(
                        "Hybrid render text overlay pass failed: %s", exc
                    )
            if rendered_clip is None:
                return {
                    "success": False,
                    "error": "Render clip unexpectedly unavailable",
                }

            output_dir = Path(self.config.paths.processed_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            project_slug = self._slugify_for_filename(
                str(getattr(state, "project_name", "hybrid"))
            )
            output_path = output_dir / (
                f"hybrid_{project_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )

            quality_profiles = [
                {"fps": 30, "codec": "libx264", "preset": "medium", "bitrate": "10M"},
                {"fps": 24, "codec": "libx264", "preset": "fast", "bitrate": "5M"},
                {"fps": 20, "codec": "libx264", "preset": "veryfast", "bitrate": "3M"},
            ]
            profile_idx = min(render_attempt, len(quality_profiles) - 1)
            profile = quality_profiles[profile_idx]

            write_kwargs: Dict[str, Any] = {
                "fps": profile["fps"],
                "codec": profile["codec"],
                "preset": profile["preset"],
                "bitrate": profile["bitrate"],
            }
            if getattr(rendered_clip, "audio", None) is not None:
                write_kwargs["audio_codec"] = "aac"
            else:
                write_kwargs["audio"] = False

            if profile_idx > 0:
                print(
                    f"[Hybrid] Render quality profile: FPS={profile['fps']}, "
                    f"preset={profile['preset']}, bitrate={profile['bitrate']}",
                    flush=True,
                )

            await asyncio.to_thread(
                rendered_clip.write_videofile,
                str(output_path),
                **write_kwargs,
            )

            manifest_payload = {
                "phase": PipelinePhase.VIDEO_RENDER.value,
                "created_at": datetime.now().isoformat(),
                "project_name": str(getattr(state, "project_name", "")),
                "output_video_path": str(output_path),
                "segments": timeline_manifest,
                "narration_segments_requested": len(tts_jobs),
                "text_overlay_count": len(overlays),
            }
            manifest_path = save_finding(
                state.project_dir,
                "scripts",
                "render_manifest.json",
                manifest_payload,
            )

            return {
                "success": True,
                "video_path": str(output_path),
                "render_manifest_path": str(manifest_path),
            }
        except Exception as exc:
            self.logger.error("Hybrid local render failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": str(exc),
            }
        finally:
            seen_ids: set[int] = set()
            for clip in reversed(created_clips):
                clip_id = id(clip)
                if clip_id in seen_ids:
                    continue
                seen_ids.add(clip_id)
                close_method = getattr(clip, "close", None)
                if callable(close_method):
                    try:
                        close_method()
                    except Exception:
                        pass

    @staticmethod
    def _slugify_for_filename(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
        cleaned = cleaned.strip("-._")
        return cleaned or "asset"

    def _load_hybrid_media_metadata(self, state: Any) -> Dict[str, Dict[str, str]]:
        metadata: Dict[str, Dict[str, str]] = {}
        state_meta = getattr(state, "metadata", {})
        evidence_index_path = str(
            state_meta.get("evidence_index_path", "") or ""
        ).strip()
        if not evidence_index_path:
            return metadata

        path = Path(evidence_index_path)
        if not path.exists() or not path.is_file():
            return metadata

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return metadata

        downloaded_media = payload.get("downloaded_media")
        if not isinstance(downloaded_media, list):
            return metadata

        for entry in downloaded_media:
            if not isinstance(entry, dict):
                continue
            media_type = str(entry.get("media_type", "") or "").strip().lower()
            if media_type not in {"image", "video"}:
                media_type = self._media_type_for_asset(
                    str(entry.get("local_path", "") or "")
                )
                if media_type not in {"image", "video"}:
                    continue
            local_path = str(entry.get("local_path", "") or "").strip()
            if not local_path:
                continue
            metadata[local_path] = {
                "media_type": media_type,
                "query": str(entry.get("query", "") or ""),
                "title": str(entry.get("title", "") or ""),
                "source_url": str(entry.get("source_url", "") or ""),
            }
        return metadata

    def _build_visual_catalog_for_script(self, state: Any) -> Dict[str, Any]:
        """Build a lightweight catalog of available images for SCRIPTING/feedback agents."""
        media_metadata = self._load_hybrid_media_metadata(state)
        images: List[Dict[str, Any]] = []
        for idx, (local_path, meta) in enumerate(sorted(media_metadata.items())):
            if not isinstance(meta, dict):
                continue
            if str(meta.get("media_type", "") or "").lower() != "image":
                continue
            images.append(
                {
                    "id": idx,
                    "local_path": local_path,
                    "query": str(meta.get("query", "") or ""),
                    "title": str(meta.get("title", "") or ""),
                    "source_url": str(meta.get("source_url", "") or ""),
                }
            )
        return {"images": images}

    @staticmethod
    def _media_type_for_asset(asset_path: str) -> str:
        suffix = Path(str(asset_path or "")).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return "image"
        if suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
            return "video"
        return "unknown"

    @staticmethod
    def _tokenize_relevance_text(text: str) -> List[str]:
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "into",
            "then",
            "about",
            "your",
            "show",
            "video",
            "image",
            "dream",
            "minecraft",
        }
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return [
            token for token in tokens if len(token) >= 3 and token not in stop_words
        ]

    def _rank_visual_candidates_for_segment(
        self,
        segment: Dict[str, Any],
        visual_assets: List[str],
        media_metadata: Dict[str, Dict[str, str]],
        *,
        top_k: int,
    ) -> List[str]:
        narration = str(segment.get("narration", "") or "")
        visual_directive = str(segment.get("visual_directive", "") or "")
        segment_tokens = set(
            self._tokenize_relevance_text(f"{narration} {visual_directive}")
        )

        def _score(asset: str) -> int:
            entry = media_metadata.get(asset, {})
            filename = Path(asset).stem.replace("_", " ")
            candidate_text = " ".join(
                [
                    str(entry.get("media_type", "") or ""),
                    str(entry.get("query", "") or ""),
                    str(entry.get("title", "") or ""),
                    str(entry.get("source_url", "") or ""),
                    filename,
                ]
            )
            candidate_tokens = set(self._tokenize_relevance_text(candidate_text))
            if not segment_tokens or not candidate_tokens:
                return 1 if self._media_type_for_asset(asset) == "video" else 0
            overlap = segment_tokens.intersection(candidate_tokens)
            base = len(overlap) * 2
            if self._media_type_for_asset(asset) == "video":
                base += 1
            return base

        ranked = sorted(
            visual_assets, key=lambda asset: (_score(asset), asset), reverse=True
        )
        return ranked[: max(1, top_k)]

    async def _score_hybrid_visual_relevance(
        self,
        media_path: Path,
        segment: Dict[str, Any],
        media_meta: Dict[str, str],
        *,
        project_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        media_type = str(media_meta.get("media_type", "") or "").strip().lower()
        if media_type not in {"image", "video"}:
            media_type = self._media_type_for_asset(str(media_path))

        if media_type == "image":
            return await self._score_hybrid_image_relevance(
                media_path, segment, media_meta
            )

        if media_type == "video":
            return await self._score_hybrid_video_relevance(
                media_path,
                segment,
                media_meta,
            )

        return {
            "score": 0,
            "relevant": False,
            "reason": "Unsupported media type",
        }

    async def _score_hybrid_image_relevance(
        self,
        image_path: Path,
        segment: Dict[str, Any],
        image_meta: Dict[str, str],
    ) -> Dict[str, Any]:
        if not image_path.exists() or not image_path.is_file():
            return {
                "score": 0,
                "relevant": False,
                "reason": "Image file missing",
            }

        narration = str(segment.get("narration", "") or "").strip()
        visual_directive = str(segment.get("visual_directive", "") or "").strip()
        evidence_refs = segment.get("evidence_refs", [])
        if isinstance(evidence_refs, list):
            evidence_text = ", ".join(
                [str(item).strip() for item in evidence_refs if str(item).strip()]
            )
        else:
            evidence_text = str(evidence_refs or "").strip()

        metadata_text = json.dumps(image_meta, ensure_ascii=True)
        prompt = (
            "Evaluate whether this image is suitable for a documentary script segment. "
            "Focus on semantic relevance to the narration and visual directive. "
            "Return strict JSON only with keys: score (0-100 integer), relevant (boolean), reason (short string).\n\n"
            f"Segment narration: {narration}\n"
            f"Visual directive: {visual_directive}\n"
            f"Evidence refs: {evidence_text}\n"
            f"Image metadata: {metadata_text}\n"
        )

        score_result = await self._score_with_multimodal_payload(
            media_type="image",
            media_path=image_path,
            prompt_text=prompt,
        )
        if score_result is not None:
            return score_result

        heuristic_score = self._estimate_image_relevance_score(
            image_path=image_path,
            segment=segment,
            image_meta=image_meta,
        )
        return {
            "score": heuristic_score,
            "relevant": heuristic_score >= 70,
            "reason": "Heuristic fallback score",
        }

    async def _score_hybrid_video_relevance(
        self,
        video_path: Path,
        segment: Dict[str, Any],
        video_meta: Dict[str, str],
    ) -> Dict[str, Any]:
        if not video_path.exists() or not video_path.is_file():
            return {
                "score": 0,
                "relevant": False,
                "reason": "Video file missing",
            }

        narration = str(segment.get("narration", "") or "").strip()
        visual_directive = str(segment.get("visual_directive", "") or "").strip()
        evidence_refs = segment.get("evidence_refs", [])
        if isinstance(evidence_refs, list):
            evidence_text = ", ".join(
                [str(item).strip() for item in evidence_refs if str(item).strip()]
            )
        else:
            evidence_text = str(evidence_refs or "").strip()

        metadata_text = json.dumps(video_meta, ensure_ascii=True)
        prompt = (
            "Evaluate whether this entire video clip is suitable for a documentary script segment. "
            "Focus on semantic relevance to narration and visual directive, not just generic gameplay visuals. "
            "Return strict JSON only with keys: score (0-100 integer), relevant (boolean), reason (short string).\n\n"
            f"Segment narration: {narration}\n"
            f"Visual directive: {visual_directive}\n"
            f"Evidence refs: {evidence_text}\n"
            f"Video metadata: {metadata_text}\n"
        )

        score_result = await self._score_with_multimodal_payload(
            media_type="video",
            media_path=video_path,
            prompt_text=prompt,
        )
        if score_result is not None:
            return score_result

        heuristic_score = self._estimate_image_relevance_score(
            image_path=video_path,
            segment=segment,
            image_meta=video_meta,
        )
        return {
            "score": heuristic_score,
            "relevant": heuristic_score >= 70,
            "reason": "Video metadata heuristic fallback",
        }

    async def _score_with_multimodal_payload(
        self,
        *,
        media_type: str,
        media_path: Path,
        prompt_text: str,
    ) -> Optional[Dict[str, Any]]:
        active_client = getattr(getattr(self, "ai_client", None), "active_client", None)
        if not (
            active_client and hasattr(active_client, "_chat_completion_with_fallback")
        ):
            return None

        try:
            if media_type == "video":
                max_video_mb = self._read_env_int(
                    "HYBRID_MULTIMODAL_VIDEO_MAX_MB",
                    default=8,
                    minimum=1,
                    maximum=128,
                )
                max_video_bytes = max_video_mb * 1024 * 1024
                if media_path.stat().st_size > max_video_bytes:
                    self.logger.debug(
                        "Skipping multimodal video scoring for %s: file exceeds %s MB",
                        media_path,
                        max_video_mb,
                    )
                    return None
                data_url = self._encode_video_as_data_url(media_path)
                user_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "video_url", "video_url": {"url": data_url}},
                ]
            else:
                data_url = self._encode_image_as_data_url(media_path)
                user_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]

            response = await active_client._chat_completion_with_fallback(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict visual relevance grader for documentary editing. "
                            "Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self.logger.debug(
                "%s relevance AI scoring failed for %s: %s",
                media_type.capitalize(),
                media_path,
                exc,
            )
            return None

        return self._parse_relevance_json_response(response)

    @staticmethod
    def _parse_relevance_json_response(response: Any) -> Dict[str, Any]:
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        score = int(parsed.get("score", 0) or 0)
        score = max(0, min(100, score))
        relevant = bool(parsed.get("relevant", score >= 70))
        reason = str(parsed.get("reason", "") or "")
        return {"score": score, "relevant": relevant, "reason": reason}

    def _estimate_image_relevance_score(
        self,
        image_path: Path,
        segment: Dict[str, Any],
        image_meta: Dict[str, str],
    ) -> int:
        segment_text = " ".join(
            [
                str(segment.get("narration", "") or ""),
                str(segment.get("visual_directive", "") or ""),
            ]
        )
        segment_tokens = set(self._tokenize_relevance_text(segment_text))

        metadata_text = " ".join(
            [
                str(image_meta.get("query", "") or ""),
                str(image_meta.get("title", "") or ""),
                str(image_meta.get("source_url", "") or ""),
                image_path.stem.replace("_", " "),
            ]
        )
        metadata_tokens = set(self._tokenize_relevance_text(metadata_text))

        if not segment_tokens or not metadata_tokens:
            return 45

        overlap_count = len(segment_tokens.intersection(metadata_tokens))
        if overlap_count <= 0:
            return 30

        return max(35, min(95, 35 + (overlap_count * 15)))

    @staticmethod
    def _encode_image_as_data_url(image_path: Path) -> str:
        raw = image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/jpeg"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _encode_video_as_data_url(video_path: Path) -> str:
        raw = video_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type:
            mime_type = "video/mp4"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _enforce_script_asset_mapping(
        self,
        script_payload: Dict[str, Any],
        assets: List[str],
    ) -> None:
        script_payload["phase"] = PipelinePhase.SCRIPTING.value
        segments = script_payload.get("segments")
        if not isinstance(segments, list):
            script_payload["segments"] = []
            segments = script_payload["segments"]

        visual_assets = self._filter_visual_assets(assets)
        target_duration = self._read_env_float(
            "HYBRID_TARGET_DURATION_SECONDS", default=45.0, minimum=15.0, maximum=180.0
        )
        min_segment_duration = self._read_env_float(
            "HYBRID_MIN_SEGMENT_DURATION_SECONDS",
            default=2.0,
            minimum=0.8,
            maximum=30.0,
        )
        max_segment_duration = self._read_env_float(
            "HYBRID_MAX_SEGMENT_DURATION_SECONDS",
            default=14.0,
            minimum=3.0,
            maximum=60.0,
        )
        narration_multiplier = self._read_env_float(
            "HYBRID_NARRATION_DURATION_MULTIPLIER",
            default=2.0,
            minimum=1.0,
            maximum=4.0,
        )

        total_duration = 0.0

        if not visual_assets:
            for segment in segments:
                if isinstance(segment, dict):
                    segment["visual_asset_path"] = ""
                    segment["intended_duration_seconds"] = (
                        self._normalize_segment_duration(
                            segment,
                            min_segment_duration=min_segment_duration,
                            max_segment_duration=max_segment_duration,
                            narration_multiplier=narration_multiplier,
                        )
                    )
                duration = float(segment.get("intended_duration_seconds", 0.0) or 0.0)
                total_duration += max(0.0, duration)
            if total_duration < target_duration and segments:
                self._distribute_segment_padding(
                    segments,
                    pad_seconds=target_duration - total_duration,
                    min_segment_duration=min_segment_duration,
                    max_segment_duration=max_segment_duration,
                )
                self._recompute_segment_timestamps(segments)
            return

        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            segment.setdefault("time_seconds", total_duration)
            segment.setdefault("intended_duration_seconds", 6.0)
            segment.setdefault("narration", "")
            segment.setdefault("visual_directive", "")
            segment.setdefault("text_overlay", "")
            segment.setdefault("evidence_refs", [])
            segment.setdefault("pace", "fast")
            segment.setdefault("emotion", "dramatic")

            duration = self._normalize_segment_duration(
                segment,
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
                narration_multiplier=narration_multiplier,
            )
            segment["intended_duration_seconds"] = duration
            total_duration += max(0.0, duration)

            existing = str(segment.get("visual_asset_path", "")).strip()
            if existing and existing in visual_assets:
                continue
            segment["visual_asset_path"] = visual_assets[idx % len(visual_assets)]

        if total_duration < target_duration and segments:
            self._distribute_segment_padding(
                segments,
                pad_seconds=target_duration - total_duration,
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
            )

        self._recompute_segment_timestamps(segments)

    @staticmethod
    def _filter_visual_assets(assets: List[str]) -> List[str]:
        visual_extensions = {
            ".mp4",
            ".webm",
            ".mkv",
            ".mov",
            ".avi",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
        }
        return [
            asset for asset in assets if Path(asset).suffix.lower() in visual_extensions
        ]

    @staticmethod
    def _normalize_segment_duration(
        segment: Dict[str, Any],
        *,
        min_segment_duration: float,
        max_segment_duration: float,
        narration_multiplier: float,
    ) -> float:
        # Prefer actual TTS duration when available (eliminates audio drift)
        actual = segment.get("actual_duration")
        if actual is not None and isinstance(actual, (int, float)) and actual > 0:
            clamped = max(
                min_segment_duration, min(max_segment_duration, float(actual) + 0.2)
            )
            return clamped

        narration = str(segment.get("narration", "") or "").strip()
        words = len(re.findall(r"\w+", narration))

        # Dynamic audio pacing: enforce word-count limits per segment
        # Fast pace: 2.5 words/sec, Normal: 2.0 words/sec, Slow: 1.5 words/sec
        pace = str(
            segment.get("pace", segment.get("pacing", "normal")) or "normal"
        ).lower()
        words_per_sec = {"fast": 2.5, "normal": 2.0, "slow": 1.5}.get(pace, 2.0)

        estimated = 2.5 if words == 0 else max(2.0, min(12.0, words / words_per_sec))

        # Word-count limit: if narration exceeds ~12 words at fast pace, flag it
        max_words_for_pace = {"fast": 12, "normal": 15, "slow": 20}.get(pace, 15)
        if words > max_words_for_pace and words > 0:
            segment["_word_count_warning"] = (
                f"{words} words exceeds {max_words_for_pace} for pace '{pace}'"
            )

        requested = float(segment.get("intended_duration_seconds", 0.0) or 0.0)
        baseline = requested if requested > 0 else estimated
        cap = min(max_segment_duration, estimated * narration_multiplier)
        duration = min(baseline, max(min_segment_duration, cap))
        return max(min_segment_duration, duration)

    @staticmethod
    def make_loopable_audio(audio_clip: Any, crossfade: float = 0.2) -> Any:
        """Crossfade the tail into the head for seamless loop playback."""
        from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips

        duration = audio_clip.duration
        if duration < crossfade * 3:
            return audio_clip
        head = audio_clip.subclipped(0, crossfade).with_start(0)
        tail = audio_clip.subclipped(duration - crossfade, duration).with_start(0)
        crossfaded_head = CompositeAudioClip(
            [
                head,
                tail.with_volume_scaled(0.3),
            ]
        )
        body = audio_clip.subclipped(crossfade, duration - crossfade)
        result = concatenate_audioclips([crossfaded_head, body.with_start(crossfade)])
        # Close intermediate clips to prevent memory leaks
        for clip in (head, tail, crossfaded_head, body):
            try:
                clip.close()
            except Exception:
                pass
        return result if result.duration > 0 else audio_clip

    @staticmethod
    def _distribute_segment_padding(
        segments: List[Any],
        *,
        pad_seconds: float,
        min_segment_duration: float,
        max_segment_duration: float,
    ) -> None:
        remaining = max(0.0, pad_seconds)
        editable_segments = [seg for seg in segments if isinstance(seg, dict)]
        if not editable_segments or remaining <= 0:
            return

        while remaining > 1e-6:
            room_segments = [
                seg
                for seg in editable_segments
                if float(seg.get("intended_duration_seconds", 0.0) or 0.0)
                < max_segment_duration - 1e-6
            ]
            if not room_segments:
                template = editable_segments[-1]
                chunk = min(max_segment_duration, remaining)
                if chunk <= 0:
                    break
                filler_segment = {
                    "time_seconds": 0.0,
                    "intended_duration_seconds": max(min_segment_duration, chunk),
                    "narration": "Add one more concrete receipt beat with source and timestamp.",
                    "visual_asset_path": str(
                        template.get("visual_asset_path", "") or ""
                    ),
                    "visual_directive": str(template.get("visual_directive", "") or ""),
                    "text_overlay": "",
                    "evidence_refs": template.get("evidence_refs", []) or [],
                    "pace": str(template.get("pace", "normal") or "normal"),
                    "emotion": str(template.get("emotion", "neutral") or "neutral"),
                }
                segments.append(filler_segment)
                editable_segments.append(filler_segment)
                remaining -= chunk
                continue

            per_segment = remaining / len(room_segments)
            progressed = False
            for seg in room_segments:
                current = float(seg.get("intended_duration_seconds", 0.0) or 0.0)
                room = max(0.0, max_segment_duration - current)
                add = min(per_segment, room)
                if add <= 0:
                    continue
                seg["intended_duration_seconds"] = current + add
                remaining -= add
                progressed = True

            if not progressed:
                break

    @staticmethod
    def _recompute_segment_timestamps(segments: List[Any]) -> None:
        cursor = 0.0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment["time_seconds"] = round(cursor, 3)
            duration = float(segment.get("intended_duration_seconds", 0.0) or 0.0)
            cursor += max(0.0, duration)

    @staticmethod
    def _read_env_float(
        key: str,
        *,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = str(os.getenv(key, "")).strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _read_env_int(
        key: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw = str(os.getenv(key, "")).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(minimum, min(maximum, value))
